"""Recompute every frozen test metric from the delivered predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from grabspark.raw_probes import confusion
from grabspark.run_raw_probe_experiment import load_frame, ordered_hash, sha256, specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    completion = json.loads((root / "completion.json").read_text())
    if completion["classification_completed"] != 101 or completion["classification_missing"]:
        raise ValueError("All 101 classification tasks must complete before delivery verification")
    models = sorted((root / "models").glob("*/*/complete.json"))
    if len(models) != 101:
        raise ValueError("Expected 101 completed model directories")
    alignment_checks = 0
    for spec in specs():
        for split, csv_path in spec["splits"].items():
            frame, _, offsets = load_frame(Path(csv_path), spec)
            cache = root / f"cache/{spec['suite']}/{spec['task']}"
            meta = json.loads((cache / f"{split}.metadata.json").read_text())
            mapping = pd.read_csv(cache / f"{split}.sequence_mapping.csv", dtype={"sequence_id": str})
            embedding = np.load(meta["embedding_path"], mmap_mode="r")
            if (meta["csv_sha256"] != sha256(Path(csv_path)) or meta["ordered_sequence_sha256"] != ordered_hash(frame)
                    or embedding.shape != (int(offsets[-1]), spec["dim"]) or str(embedding.dtype) != meta["dtype"]
                    or not np.array_equal(mapping.source_row, frame.source_row)
                    or not np.array_equal(mapping.sequence_id, frame.sequence_id)
                    or not np.array_equal(mapping.sequence_sha256, frame.sequence_sha256)
                    or not np.array_equal(mapping.token_start, offsets[:-1])
                    or not np.array_equal(mapping.token_end, offsets[1:])):
                raise ValueError(f"{spec['suite']}/{spec['task']}/{split}: input/token mapping differs")
            alignment_checks += 1
    checks = []
    for path in models:
        folder = path.parent
        suite, task = folder.parent.name, folder.name
        payload = json.loads((folder / "all_candidates.json").read_text())
        candidates = payload["models"]
        if len(candidates) != 23 or [m["K"] for m in candidates[:10]] != list(range(1, 11)):
            raise ValueError(f"{suite}/{task}: missing candidates")
        if [m["lambda"] for m in candidates[10:]] != [10.0**e for e in range(-8, 5)]:
            raise ValueError(f"{suite}/{task}: ridge grid differs")
        settings_path = folder / "frozen_settings.json"
        freeze_hash = hashlib.sha256(settings_path.read_bytes()).hexdigest()
        settings = json.loads(settings_path.read_text())["selected"]
        summary = pd.read_csv(folder / "test_summary.csv")
        per_sequence = pd.read_csv(folder / "test_per_sequence.csv", dtype={"sequence_id": str, "sequence_sha256": str})
        for split in summary.split.unique():
            with np.load(root / f"predictions/{suite}/{task}/{split}.npz", allow_pickle=False) as data:
                scores, labels, offsets = data["scores"], data["labels"], data["offsets"]
                ids, hashes = data["sequence_ids"], data["sequence_sha256"]
                if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
                    raise ValueError("Nonfinite prediction or invalid binary labels")
                if scores.shape != (len(labels), 23) or offsets[-1] != len(labels) or not np.all(np.diff(offsets) > 0):
                    raise ValueError("Prediction shape/token boundary mismatch")
                for setting in settings:
                    row = summary[(summary.split == split) & (summary.method == setting["method"])].iloc[0]
                    if row.freeze_sha256 != freeze_hash:
                        raise ValueError("Freeze hash mismatch")
                    selected = per_sequence[(per_sequence.split == split) & (per_sequence.method == setting["method"])]
                    if not np.array_equal(selected.sequence_id.to_numpy(), ids) or not np.array_equal(selected.sequence_sha256.to_numpy(), hashes):
                        raise ValueError("Delivered prediction sequence identities differ")
                    metrics = []
                    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                        result = confusion(labels[start:end], scores[start:end, setting["model_index"]], setting["threshold"])
                        metrics.append(result["macro_f1"])
                        delivered = selected.iloc[i]
                        for key in ("tp", "tn", "fp", "fn"):
                            if int(delivered[key]) != result[key]:
                                raise ValueError("Confusion count does not reproduce")
                    recomputed = float(np.mean(metrics))
                    if abs(recomputed-row.macro_f1) > 1e-10:
                        raise ValueError("Frozen test metric does not reproduce")
                    checks.append({"suite": suite, "task": task, "split": split, "method": setting["method"],
                                   "reported_f1": row.macro_f1, "recomputed_f1": recomputed,
                                   "absolute_error": abs(recomputed-row.macro_f1), "n_sequences": len(ids), "n_tokens": len(labels)})
    grid = pd.read_csv(root / "validation_grid.csv")
    frozen = pd.read_csv(root / "frozen_settings.csv")
    budget = pd.read_csv(root / "budget_curves.csv")
    comparisons = pd.read_csv(root / "all_method_comparison.csv")
    if len(comparisons) != 103 or comparisons[["CrossLatent", "Raw-OLS", "Raw-Ridge", "best_single"]].isna().any().any():
        raise ValueError("All-method comparison is incomplete")
    if len(grid) != 101*23*21 or len(frozen) != 202 or len(checks) != 206 or len(budget) != 2030:
        raise ValueError("Delivery candidate/freeze/evaluation/budget row counts are incomplete")
    report = {"status": "pass", "tasks": len(models), "test_metrics": len(checks),
              "input_alignment_checks": alignment_checks,
              "validation_rows": len(grid), "frozen_models": len(frozen), "budget_rows": len(budget),
              "max_absolute_f1_error": max(c["absolute_error"] for c in checks), "checks": checks}
    (root / "delivery_verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS: 101 tasks, 206 frozen test metrics, maximum F1 error {report['max_absolute_f1_error']:.3g}")


if __name__ == "__main__":
    main()
