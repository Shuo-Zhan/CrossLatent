"""Portable command line; paths are explicit or relative to the run configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._identity import DISPLAY_NAME, PACKAGE_NAME, VERSION
from .api import ActivationDataset, LinearReadout, evaluate, fit, refit_leave_one_out


def _write(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_config(config_path: str | Path) -> dict:
    """Paths resolve against the JSON location, independent of current directory."""
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    allowed = {"train", "valid", "test", "output", "max_latents", "task", "tie_break", "method", "fixed_budget"}
    if set(config) - allowed:
        raise ValueError(f"unknown run settings: {sorted(set(config) - allowed)}")

    def resolve(key: str) -> Path:
        return (config_path.parent / Path(config[key]).expanduser()).resolve()

    train, valid = ActivationDataset.load(resolve("train")), ActivationDataset.load(resolve("valid"))
    model = fit(train, valid, **{k: v for k, v in config.items() if k in {
        "max_latents", "task", "tie_break", "method", "fixed_budget"}})
    output = resolve("output")
    model.save(output / "model.json")
    report = {"validation": evaluate(model, valid), "model": "model.json"}
    if "test" in config:
        from .api import _compatible

        test = ActivationDataset.load(resolve("test"))
        _compatible(train, valid, test)
        report["test"] = evaluate(model, test)
    _write(output / "metrics.json", report)
    # Preserve portable relative inputs rather than the machine's absolute paths.
    _write(output / "run_settings.json", config)
    return report


def demo(output: str | Path) -> dict:
    """Small seeded synthetic complementarity example; not biological evidence."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for split, sequences in [("train", 30), ("valid", 12), ("test", 12)]:
        x = (rng.random((sequences * 40, 16)) < 0.22).astype(np.float32)
        y = ((x[:, 0] + x[:, 1]) > 0).astype(float)
        data = ActivationDataset(x, y, np.full(sequences, 40), "synthetic-fixed-dictionary-v1",
                                 np.array([f"{split}-{i}" for i in range(sequences)]))
        data.save(root / f"{split}.npz")
    config = {"train": "train.npz", "valid": "valid.npz", "test": "test.npz",
              "output": "result", "max_latents": 4, "task": "binary", "tie_break": "lower"}
    _write(root / "run.json", config)
    report = run_config(root / "run.json")
    single = fit(ActivationDataset.load(root / "train.npz"), ActivationDataset.load(root / "valid.npz"),
                 method="best-single", max_latents=1)
    single.save(root / "result/single.json")
    report["single_test"] = evaluate(single, ActivationDataset.load(root / "test.npz"))
    report["scope"] = "synthetic example only"
    _write(root / "result/metrics.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description=f"{DISPLAY_NAME}: fixed-dictionary compositional interpretation")
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("demo", help="run a small synthetic example without model checkpoints")
    sample.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="fit/validate/test using paths relative to a JSON configuration")
    run.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("fit", help="select on train and freeze K/threshold on validation")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--valid", type=Path, required=True)
    train.add_argument("--max-latents", type=int, default=10)
    train.add_argument("--task", choices=["binary", "continuous"], default="binary")
    train.add_argument("--tie-break", choices=["lower", "closest-half"], default="lower")
    train.add_argument("--method", choices=["residual", "topk", "best-single"], default="residual")
    train.add_argument("--fixed-budget", action="store_true", help="keep the final prefix for shared-set analysis; tune only threshold")
    train.add_argument("--output", type=Path, required=True)
    for name in ["evaluate", "refit"]:
        command = commands.add_parser(name, help="frozen evaluation" if name == "evaluate" else "fixed-set leave-one-out OLS refitting")
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--test", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "refit":
            command.add_argument("--train", type=Path, required=True)
            command.add_argument("--valid", type=Path, required=True)
    predict = commands.add_parser("predict", help="raw frozen predictions from a full activation NPY")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--activations", type=Path, required=True)
    predict.add_argument("--dictionary-id", required=True)
    predict.add_argument("--output", type=Path, required=True)
    steps = commands.add_parser("predict-steps", help="export final-coefficient cumulative prediction trajectories")
    steps.add_argument("--model", type=Path, required=True)
    steps.add_argument("--activations", type=Path, required=True)
    steps.add_argument("--dictionary-id", required=True)
    steps.add_argument("--lengths", type=Path, help="integer NPY token counts per sequence; default: one sequence")
    steps.add_argument("--sequence-ids", type=Path, help="JSON list of sequence identifiers")
    steps.add_argument("--output", type=Path, required=True, help="new output directory")
    legacy = commands.add_parser("import-legacy", help="convert an explicitly chosen historical combination JSON")
    legacy.add_argument("--input", type=Path, required=True)
    legacy.add_argument("--feature", required=True)
    legacy.add_argument("--n-features", type=int, required=True)
    legacy.add_argument("--dictionary-id", required=True)
    legacy.add_argument("--task", choices=["binary", "continuous"], required=True)
    legacy.add_argument("--threshold", type=float)
    legacy.add_argument("--output", type=Path, required=True)
    shared = commands.add_parser("shared", help="intersect latent sets from independently fitted models")
    shared.add_argument("--models", type=Path, nargs="+", required=True)
    shared.add_argument("--output", type=Path, required=True)
    encode = commands.add_parser("encode", help="encode existing embeddings with an explicit fixed SAE checkpoint")
    encode.add_argument("--embeddings", type=Path, required=True)
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--kind", choices=["protein", "rna", "dna"], required=True)
    encode.add_argument("--batch-tokens", type=int, required=True)
    encode.add_argument("--device", default="cpu")
    encode.add_argument("--dtype", choices=["float32", "bfloat16"])
    encode.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "demo":
        payload = demo(args.output)
    elif args.command == "run":
        payload = run_config(args.config)
    elif args.command == "fit":
        model = fit(ActivationDataset.load(args.train), ActivationDataset.load(args.valid),
                    max_latents=args.max_latents, task=args.task, tie_break=args.tie_break, method=args.method,
                    fixed_budget=args.fixed_budget)
        model.save(args.output)
        payload = {"model": str(args.output), "latent_ids": model.latent_ids, "threshold": model.threshold}
    elif args.command in {"evaluate", "refit"}:
        model, test = LinearReadout.load(args.model), ActivationDataset.load(args.test)
        payload = evaluate(model, test) if args.command == "evaluate" else refit_leave_one_out(
            model, ActivationDataset.load(args.train), ActivationDataset.load(args.valid), test)
        _write(args.output, payload)
    elif args.command == "predict":
        model = LinearReadout.load(args.model)
        if model.dictionary_id != args.dictionary_id:
            raise ValueError("dictionary_id does not match model")
        prediction = model.predict(np.load(args.activations, mmap_mode="r", allow_pickle=False))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as handle:
            np.save(handle, prediction)
        payload = {"tokens": len(prediction), "output": str(args.output)}
    elif args.command == "predict-steps":
        from .trajectory import export_steps
        payload = export_steps(
            LinearReadout.load(args.model), np.load(args.activations, mmap_mode="r", allow_pickle=False),
            args.output, dictionary_id=args.dictionary_id,
            lengths=np.load(args.lengths, allow_pickle=False) if args.lengths else None,
            sequence_ids=json.loads(args.sequence_ids.read_text()) if args.sequence_ids else None)
    elif args.command == "import-legacy":
        if args.task == "binary" and args.threshold is None:
            raise ValueError("supply the validation-frozen --threshold; training best_threshold is not used")
        original = json.loads(args.input.read_text())
        combo = original["features"][args.feature]["combination"]
        model = LinearReadout(combo["neuron_ids"], combo["coefficients"], combo["intercept_b"],
                              args.n_features, args.dictionary_id, args.task, args.threshold,
                              {"source_filename": args.input.name, "feature": args.feature,
                               "scope": "explicit historical model; no source selection or threshold tuning"})
        model.save(args.output)
        payload = {"model": str(args.output), "latent_ids": model.latent_ids}
    elif args.command == "shared":
        from .interpretation import shared_latents

        ids = shared_latents([LinearReadout.load(path) for path in args.models])
        payload = {"shared_latents": ids, "count": len(ids)}
        _write(args.output, payload)
    elif args.command == "encode":
        from .adapters import encode_embeddings

        payload = encode_embeddings(args.embeddings, args.output, kind=args.kind, checkpoint=args.checkpoint,
                                    batch_tokens=args.batch_tokens, device=args.device, dtype=args.dtype)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
