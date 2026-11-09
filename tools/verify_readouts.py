"""Compare all released weights and validation choices against saved source JSONs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def verify(readouts: Path, data: Path | None = None) -> dict:
    from crosslatent import LinearReadout
    import numpy as np

    catalog=json.loads((readouts/"index.json").read_text())
    assert len(catalog["tasks"])==100
    models=budgets=0
    rng=np.random.default_rng(20261109)
    for row in catalog["tasks"]:
        path=readouts/row["model"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["model_sha256"]
        model=LinearReadout.load(path)
        assert len(model.latent_ids)==row["selected_k"] and model.threshold==row["threshold"]
        candidates=json.loads((readouts/row["trajectory"]).read_text())["budgets"]
        assert len(candidates)==10
        if data is not None:
            curve_path=(data/model.metadata["summary_json"]).with_name("selection_curve.csv")
            with curve_path.open() as stream:validation_rows=[r for r in csv.DictReader(stream) if r["method"]=="CrossLatent"]
        for k,candidate in enumerate(candidates,1):
            assert len(candidate["latent_ids"])==len(candidate["coefficients"])==k
            if data is not None:
                source=data/candidate["metadata"]["source_json"]
                assert hashlib.sha256(source.read_bytes()).hexdigest()==candidate["metadata"]["source_json_sha256"]
                payload=json.loads(source.read_text())["features"][candidate["metadata"]["position_annotation"]]["combination"]
                assert candidate["latent_ids"]==payload["neuron_ids"]
                assert candidate["coefficients"]==payload["coefficients"]
                assert candidate["intercept"]==payload["intercept_b"]
                best_k=min((r for r in validation_rows if int(r["K"])==k),key=lambda r:(-float(r["valid_macro_f1"]),float(r["threshold"])))
                assert candidate["threshold"]==float(best_k["threshold"])
            budgets+=1
        x=np.zeros((7,model.n_features),dtype=np.float64)
        response=rng.normal(size=(7,len(model.latent_ids)))
        x[:,model.latent_ids]=response
        np.testing.assert_allclose(model.predict(x),model.intercept+response@np.array(model.coefficients),rtol=1e-12,atol=1e-12)
        if data is not None:
            s=json.loads((data/model.metadata["summary_json"]).read_text())
            assert s["selected_k"]==len(model.latent_ids)
            assert s["selected_threshold"]==model.threshold
            curve_path=(data/model.metadata["summary_json"]).with_name("selection_curve.csv")
            with curve_path.open() as stream:rows=[r for r in csv.DictReader(stream) if r["method"]=="CrossLatent"]
            best=min(rows,key=lambda r:(-float(r["valid_macro_f1"]),int(r["K"]),float(r["threshold"])))
            assert int(best["K"])==len(model.latent_ids) and float(best["threshold"])==model.threshold
        models+=1
    return {"saved_models_loaded":models,"fitted_budgets_verified":budgets,
            "weights_equal_original_json":data is not None,"validation_operating_points_verified":data is not None,
            "prediction_formula_checks":models}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[1]/"crosslatent/pretrained_readouts"
    parser.add_argument("--readouts",type=Path,default=default)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    report=verify(args.readouts.resolve(),args.data_root.resolve() if args.data_root else None)
    text=json.dumps(report,indent=2)+"\n"
    if args.output:args.output.write_text(text)
    print(text)


if __name__=="__main__":main()
