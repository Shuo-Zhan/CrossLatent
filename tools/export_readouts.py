"""Export the 60 protein and 40 RNA frozen predictors from saved study records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export(data_root: Path, output: Path, catalog: dict) -> dict:
    source = data_root / "inputs/fig/paper_figures/fig3/biological100"
    index = []
    for suite, evaluation, fitting, expected, width, checkpoint_key in [
        ("protein60", "protein60_valid_test", "protein_domain_latent10", 60, 10240, "protein_sae_sha256"),
        ("rna40", "rna40_valid_test", "rna40_latent10", 40, 9216, "rna_sae_sha256"),
    ]:
        summaries = sorted((source / "evaluation" / evaluation).glob("*/summary.json"))
        assert len(summaries) == expected, (suite, len(summaries))
        for summary_path in summaries:
            task = summary_path.parent.name
            summary = json.loads(summary_path.read_text())
            assert summary["status"] == "complete"
            curve_path = summary_path.with_name("selection_curve.csv")
            with curve_path.open() as stream:
                curve = [row for row in csv.DictReader(stream) if row["method"] == "CrossLatent"]
            winner = min(curve, key=lambda x: (-float(x["valid_macro_f1"]), int(x["K"]), float(x["threshold"])))
            assert int(winner["K"]) == summary["selected_k"]
            assert float(winner["threshold"]) == summary["selected_threshold"]
            assert math.isclose(float(winner["test_macro_f1"]), summary["test_macro_f1"], abs_tol=1e-12)
            label = "mask_target_domain" if suite == "protein60" else summary["label"]
            dictionary_id = f"study:{suite}:sha256:{catalog[checkpoint_key]}"
            models = []
            for k in range(1, 11):
                path = source / "results" / fitting / task / f"{task}_{label}_latent10_{k}.json"
                original = json.loads(path.read_text())["features"][label]
                combo = original["combination"]
                ids = [int(n) for n in combo["neuron_ids"]]
                weights = [float(w) for w in combo["coefficients"]]
                assert len(ids) == len(weights) == len(set(ids)) == k
                assert all(0 <= n < width for n in ids)
                assert all(math.isfinite(x) for x in [*weights, float(combo["intercept_b"])] )
                chosen = min((x for x in curve if int(x["K"]) == k), key=lambda x: (-float(x["valid_macro_f1"]), float(x["threshold"])))
                metadata = {
                    "suite": suite, "task_id": task, "position_annotation": label,
                    "feature_index_base": 0, "feature_count": k,
                    "sae_checkpoint_sha256": catalog[checkpoint_key],
                    "threshold_source": "validation selection_curve.csv; never the training JSON best_threshold",
                    "validation_mean_sequence_macro_f1": float(chosen["valid_macro_f1"]),
                    "test_mean_sequence_macro_f1": float(chosen["test_macro_f1"]),
                    "source_json": path.relative_to(data_root).as_posix(),
                    "source_json_sha256": digest(path),
                    "summary_json": summary_path.relative_to(data_root).as_posix(),
                    "summary_sha256": digest(summary_path),
                    "selection_curve_sha256": digest(curve_path),
                    "prediction": "intercept + activations[:, latent_ids] @ coefficients; positive if score >= threshold",
                }
                model = {"schema": "latent-readout/v1", "latent_ids": ids, "coefficients": weights,
                         "intercept": float(combo["intercept_b"]), "n_features": width,
                         "dictionary_id": dictionary_id, "task": "binary",
                         "threshold": float(chosen["threshold"]), "metadata": metadata}
                models.append(model)
            selected = models[int(summary["selected_k"])-1]
            assert selected["threshold"] == summary["selected_threshold"]
            assert math.isclose(selected["metadata"]["validation_mean_sequence_macro_f1"], summary["valid_macro_f1"], abs_tol=1e-12)
            selected["metadata"]["selection_rule"] = summary["selection_rule"]
            selected["metadata"]["best_single_baseline"] = {
                "latent_id": int(summary["best_single_neuron"]), "coefficient": 1.0, "intercept": 0.0,
                "threshold": float(summary["best_single_valid_threshold"]),
                "test_mean_sequence_macro_f1": float(summary["best_single_test_macro_f1"]),
            }
            model_path = Path(suite) / f"{task}.json"
            trajectory_path = Path("trajectories") / suite / f"{task}.json"
            save(output / model_path, selected)
            save(output / trajectory_path, {"schema": "crosslatent-fitted-budgets/v1", "suite":suite,
                 "task_id":task,"selected_k":summary["selected_k"], "budgets":models,
                 "note":"Each K has its independently refitted coefficients and validation-selected threshold. This is distinct from truncating the final model with frozen coefficients."})
            index.append({"suite":suite,"task_id":task,"model":model_path.as_posix(),
                          "trajectory":trajectory_path.as_posix(),"selected_k":summary["selected_k"],
                          "threshold":selected["threshold"],"latent_ids":selected["latent_ids"],
                          "test_macro_f1":summary["test_macro_f1"],"model_sha256":digest(output/model_path)})
    assert len(index) == 100
    save(output / "index.json", {"schema":"crosslatent-study-readouts/v1", 
         "tasks":index,"task_count":100,"protein_tasks":60,"rna_tasks":40,"fitted_budgets":1000,
         "feature_index_base":0,"selection":"All operating points selected by validation performance."})
    with (output / "index.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["suite","task_id","selected_k","threshold","latent_ids","test_macro_f1","model","trajectory","model_sha256"])
        writer.writeheader()
        for row in index:writer.writerow(dict(row,latent_ids=";".join(map(str,row["latent_ids"]))))
    report={"tasks":len(index),"budgets":1000,"validation_choices_verified":True,
            "protein_mean_f1":sum(x["test_macro_f1"] for x in index if x["suite"]=="protein60")/60,
            "rna_mean_f1":sum(x["test_macro_f1"] for x in index if x["suite"]=="rna40")/40,
            "json_bytes":sum(p.stat().st_size for p in output.rglob("*.json"))}
    save(output / "export_validation.json",report)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model-catalog",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.data_root.resolve(),args.output.resolve(),json.loads(args.model_catalog.read_text())),indent=2))


if __name__ == "__main__":
    main()
