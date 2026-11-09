#!/usr/bin/env python3
"""
Evaluate all Evo2 boosting JSON checkpoints in one forward pass.

Specify a results name (no suffix), e.g.
  Evo2_7b_262k_layer26_atac_K562_neuron_parameters
then load every matching file:
  {name}_0.json, {name}_1.json, ... {name}_N.json

For each sequence: Evo2 embed + SAE encode once, then apply every checkpoint's
combination coefficients. Saves per-checkpoint overall / per-sequence R² and
all continuous predictions.

Example:
    CUDA_VISIBLE_DEVICES=2 python Evo2_eval_neuron_boosting_seq.py \\
        --results_name Evo2_7b_262k_layer26_atac_K562_neuron_parameters \\
        --results_dir data/test_sample_3000_a/results \\
        --fasta_path data/test_sample_3000_b/sequences.fasta \\
        --label_npy data/test_sample_3000_b/atac_K562.npy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_parser_early = argparse.ArgumentParser(add_help=False)
_parser_early.add_argument("--cuda_visible_devices", type=str, default=None)
_early_args, _ = _parser_early.parse_known_args()
if _early_args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.cuda_visible_devices
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from neuron_boosting_seq import _encode_batch, build_streaming_context
from evo2_sae_utils import (
    DEFAULT_EVO2_CKPT,
    DEFAULT_LAYER,
    DEFAULT_SAE_CKPT,
    build_combination_prediction,
    load_evo2_and_sae,
    load_fasta_continuous_labels,
    sequence_r2,
)

DEFAULT_RESULTS_DIR = (
    SCRIPT_DIR / "data" / "test_sample_3000_a" / "results"
)
DEFAULT_RESULTS_NAME = "Evo2_7b_262k_layer26_phyloP_neuron_parameters"
DEFAULT_FASTA = SCRIPT_DIR / "data" / "test_sample_3000_b" / "sequences.fasta"
DEFAULT_LABEL_NPY = SCRIPT_DIR / "data" / "test_sample_3000_b" / "phyloP.npy"
DEFAULT_OUTPUT_JSON = (
    SCRIPT_DIR
    / "data"
    / "test_sample_3000_b"
    / "results"
    / "Evo2_7b_262k_layer26_phyloP_evaluation_all_iters.json"
)

_SUFFIXED_JSON_RE = re.compile(r"^(?P<name>.+)_(?P<iter>\d+)\.json$")


def load_results_json(path: str | Path, feature: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "features" not in payload or feature not in payload["features"]:
        raise KeyError(f"Feature '{feature}' not found in {path}")
    return payload


def extract_combination_cfg(payload: dict[str, Any], feature: str) -> dict[str, Any]:
    combo = payload["features"][feature]["combination"]
    return {
        "neuron_ids": [int(x) for x in combo["neuron_ids"]],
        "coefficients": [float(x) for x in combo["coefficients"]],
        "intercept_b": float(combo["intercept_b"]),
        "composite_r2": float(combo.get("composite_r2", float("nan"))),
        "best_threshold": combo.get("best_threshold"),
        "best_macro_f1": combo.get("best_macro_f1"),
        "iteration": int(payload.get("iteration", -1)),
        "num_boosting_steps": int(payload.get("num_boosting_steps", 0)),
    }


def normalize_results_name(results_name: str) -> str:
    """Strip optional .json / trailing _N so user can pass a stem or a file path."""
    name = Path(results_name).name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    match = _SUFFIXED_JSON_RE.match(f"{name}.json")
    if match is not None:
        # User passed ..._5 or ..._5.json → keep base without iteration suffix.
        return match.group("name")
    return name


def discover_checkpoint_jsons(
    results_dir: str | Path,
    results_name: str,
) -> list[Path]:
    """
    Load all JSONs named `{results_name}_{N}.json` under results_dir,
    sorted by N. Does not include the unsuffixed `{results_name}.json`.
    """
    root = Path(results_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"results_dir not found: {root}")

    base = normalize_results_name(results_name)
    pattern = f"{base}_*.json"
    candidates: list[tuple[int, Path]] = []
    for path in sorted(root.glob(pattern)):
        match = _SUFFIXED_JSON_RE.match(path.name)
        if match is None:
            continue
        if match.group("name") != base:
            # Avoid matching longer prefixes, e.g. name_foo_1 when base=name.
            continue
        candidates.append((int(match.group("iter")), path.resolve()))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint JSONs matching '{base}_*.json' under {root}"
        )
    candidates.sort(key=lambda x: x[0])
    return [path for _, path in candidates]


def load_all_checkpoints(
    paths: list[Path],
    feature: str,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for path in paths:
        payload = load_results_json(path, feature)
        cfg = extract_combination_cfg(payload, feature)
        match = _SUFFIXED_JSON_RE.match(path.name)
        iter_id = int(match.group("iter")) if match else int(cfg.get("iteration", -1))
        models.append(
            {
                "key": f"iter_{iter_id}",
                "iteration": iter_id,
                "path": str(path),
                "cfg": cfg,
                "payload_config": payload.get("config", {}),
            }
        )
    models.sort(key=lambda m: m["iteration"])
    return models


@torch.no_grad()
def encode_and_evaluate_all_models(
    ctx,
    y: np.ndarray,
    seq_lengths: list[int],
    sequence_ids: list[str],
    sequences: list[str],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    One Evo2+SAE forward per sequence; apply every checkpoint combination.
    Retains per-model predictions and per-sequence R².
    """
    n_seq = len(seq_lengths)
    n_models = len(models)
    max_len = max(seq_lengths) if seq_lengths else 0
    uniform_len = len(set(seq_lengths)) == 1
    if not uniform_len:
        raise ValueError(
            "Variable-length sequences are not supported in multi-checkpoint "
            "prediction dump; pad or filter to a fixed length."
        )

    labels = np.zeros((n_seq, max_len), dtype=np.float32)
    # predictions[m, i, t]
    predictions = np.zeros((n_models, n_seq, max_len), dtype=np.float32)
    r2_matrix = np.zeros((n_models, n_seq), dtype=np.float64)
    y_pred_flat = [np.zeros_like(y, dtype=np.float64) for _ in range(n_models)]

    print(f"Per-sequence eval: {n_seq} sequences × {n_models} checkpoints")
    for m in models:
        cfg = m["cfg"]
        print(
            f"  {m['key']}: neurons={cfg['neuron_ids']}, "
            f"b={cfg['intercept_b']:.6f}, train_r2={cfg.get('composite_r2')}"
        )

    for row_idx in tqdm(range(n_seq), desc="Per-sequence all-checkpoint R²"):
        seq_len = seq_lengths[row_idx]
        sequence = sequences[row_idx]
        if len(sequence) != seq_len:
            raise ValueError(
                f"Row {row_idx}: sequence len={len(sequence)} != seq_len={seq_len}"
            )

        f_seq = _encode_batch(ctx, [sequence])
        if f_seq.shape[0] != seq_len:
            raise ValueError(
                f"Row {row_idx}: SAE activations={f_seq.shape[0]} != seq_len={seq_len}"
            )

        tok_start = int(ctx.token_offsets[row_idx])
        tok_end = int(ctx.token_offsets[row_idx + 1])
        y_seq = y[tok_start:tok_end]
        labels[row_idx] = y_seq.astype(np.float32)

        for m_idx, model in enumerate(models):
            cfg = model["cfg"]
            pred = build_combination_prediction(
                f_seq=f_seq,
                neuron_ids=cfg["neuron_ids"],
                coefficients=cfg["coefficients"],
                intercept_b=cfg["intercept_b"],
            )
            predictions[m_idx, row_idx] = pred.astype(np.float32)
            y_pred_flat[m_idx][tok_start:tok_end] = pred
            r2_matrix[m_idx, row_idx] = float(sequence_r2(y_seq, pred))

    overall_r2 = np.asarray(
        [sequence_r2(y, y_pred_flat[m]) for m in range(n_models)],
        dtype=np.float64,
    )
    mean_per_seq_r2 = r2_matrix.mean(axis=1)

    per_sequence: list[dict[str, Any]] = []
    for row_idx in range(n_seq):
        row: dict[str, Any] = {
            "row_index": int(row_idx),
            "sequence_id": sequence_ids[row_idx],
            "uniprot_id": sequence_ids[row_idx],
            "seq_length": int(seq_lengths[row_idx]),
            "r2_by_checkpoint": {},
        }
        for m_idx, model in enumerate(models):
            row["r2_by_checkpoint"][model["key"]] = float(r2_matrix[m_idx, row_idx])
        per_sequence.append(row)

    checkpoint_summaries = []
    for m_idx, model in enumerate(models):
        best_row = int(np.argmax(r2_matrix[m_idx]))
        checkpoint_summaries.append(
            {
                "key": model["key"],
                "iteration": model["iteration"],
                "path": model["path"],
                "neuron_ids": model["cfg"]["neuron_ids"],
                "coefficients": model["cfg"]["coefficients"],
                "intercept_b": model["cfg"]["intercept_b"],
                "num_neurons": len(model["cfg"]["neuron_ids"]),
                "train_composite_r2": model["cfg"].get("composite_r2"),
                "overall_r2": float(overall_r2[m_idx]),
                "mean_per_sequence_r2": float(mean_per_seq_r2[m_idx]),
                "best_sequence": {
                    "row_index": best_row,
                    "sequence_id": sequence_ids[best_row],
                    "r2": float(r2_matrix[m_idx, best_row]),
                },
            }
        )
        print(
            f"{model['key']}: overall R²={overall_r2[m_idx]:.6f}, "
            f"mean per-seq R²={mean_per_seq_r2[m_idx]:.6f}, "
            f"neurons={model['cfg']['neuron_ids']}"
        )

    return {
        "labels": labels,
        "predictions": predictions,
        "r2_matrix": r2_matrix,
        "overall_r2": overall_r2,
        "mean_per_seq_r2": mean_per_seq_r2,
        "per_sequence": per_sequence,
        "checkpoint_summaries": checkpoint_summaries,
    }


def save_all_outputs(
    *,
    result: dict[str, Any],
    models: list[dict[str, Any]],
    output_json: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stem = output_json.with_suffix("")
    files: dict[str, Any] = {"evaluation_json": str(output_json)}

    r2_path = Path(f"{stem}_r2.npy")
    np.save(r2_path, result["r2_matrix"].astype(np.float32))
    files["r2_matrix"] = str(r2_path)
    files["r2_matrix_shape"] = " [n_checkpoints, n_sequences]"
    files["checkpoint_keys"] = [m["key"] for m in models]

    keys_path = Path(f"{stem}_checkpoint_keys.json")
    with open(keys_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "index": i,
                    "key": m["key"],
                    "iteration": m["iteration"],
                    "path": m["path"],
                    "neuron_ids": m["cfg"]["neuron_ids"],
                }
                for i, m in enumerate(models)
            ],
            handle,
            indent=2,
            ensure_ascii=False,
        )
    files["checkpoint_keys_json"] = str(keys_path)

    if save_predictions:
        labels_path = Path(f"{stem}_labels.npy")
        np.save(labels_path, result["labels"])
        files["labels"] = str(labels_path)

        # One file per checkpoint for easy loading: [n_seq, L]
        pred_dir = Path(f"{stem}_predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_files: dict[str, str] = {}
        for m_idx, model in enumerate(models):
            pred_path = pred_dir / f"{model['key']}_pred.npy"
            np.save(pred_path, result["predictions"][m_idx])
            pred_files[model["key"]] = str(pred_path)
        # Also a stacked array [n_models, n_seq, L]
        stacked_path = Path(f"{stem}_predictions_stacked.npy")
        np.save(stacked_path, result["predictions"])
        files["predictions_dir"] = str(pred_dir)
        files["predictions_by_checkpoint"] = pred_files
        files["predictions_stacked"] = str(stacked_path)

    return files


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    results_name = normalize_results_name(args.results_name)
    paths = discover_checkpoint_jsons(
        results_dir=args.results_dir,
        results_name=results_name,
    )
    models = load_all_checkpoints(paths, args.feature)
    if not models:
        raise RuntimeError("No checkpoints loaded.")

    train_config = models[-1]["payload_config"]
    print(f"Results name: {results_name}")
    print(f"Loaded {len(models)} checkpoints from {args.results_dir}:")
    for m in models:
        print(f"  {m['key']}: {m['path']}")

    df, y = load_fasta_continuous_labels(
        args.fasta_path,
        args.label_npy,
        feature_name=args.feature,
    )
    if args.max_sequences is not None:
        n = int(args.max_sequences)
        df = df.iloc[:n].reset_index(drop=True)
        total = int(df["sequence"].astype(str).str.len().sum())
        y = y[:total]

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    sequence_ids = df["sequence_id"].astype(str).tolist()
    sequences = df["sequence"].astype(str).tolist()
    print(
        f"Evaluating sequences: {len(seq_lengths)}, tokens: {sum(seq_lengths)}, "
        f"label mean={float(y.mean()):.6g}"
    )

    evo2_ckpt = args.evo2_ckpt or train_config.get("evo2_ckpt", DEFAULT_EVO2_CKPT)
    sae_ckpt = args.sae_ckpt or train_config.get("sae_ckpt", DEFAULT_SAE_CKPT)
    layer_name = args.layer_name or train_config.get("layer_name", DEFAULT_LAYER)
    max_seq_len = max(seq_lengths) if seq_lengths else 1024
    max_batch_tokens = int(args.max_batch_tokens or max_seq_len)
    if max_batch_tokens < max_seq_len:
        raise ValueError(
            f"max_batch_tokens={max_batch_tokens} < max sequence length={max_seq_len}"
        )

    embedder, sae, device = load_evo2_and_sae(
        evo2_ckpt=evo2_ckpt,
        sae_ckpt=sae_ckpt,
        layer_name=layer_name,
        device=args.device,
    )
    ctx = build_streaming_context(
        df=df,
        y=y,
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=26,
        max_batch_tokens=max_batch_tokens,
        embed_batch_size=1,
    )

    result = encode_and_evaluate_all_models(
        ctx=ctx,
        y=y,
        seq_lengths=seq_lengths,
        sequence_ids=sequence_ids,
        sequences=sequences,
        models=models,
    )
    del embedder, sae, ctx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_json = Path(args.output_json)
    saved_files = save_all_outputs(
        result=result,
        models=models,
        output_json=out_json,
        save_predictions=bool(args.save_predictions),
    )

    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Sequences: {len(seq_lengths)}, checkpoints: {len(models)}")
    for summary in result["checkpoint_summaries"]:
        print(
            f"  {summary['key']}: overall={summary['overall_r2']:.6f}, "
            f"mean_per_seq={summary['mean_per_sequence_r2']:.6f}, "
            f"n_neurons={summary['num_neurons']}"
        )

    output = {
        "results_dir": str(args.results_dir),
        "results_name": results_name,
        "results_jsons": [m["path"] for m in models],
        "fasta_path": str(args.fasta_path),
        "label_npy": str(args.label_npy),
        "streaming": True,
        "per_sequence_eval": True,
        "continuous_label": True,
        "metric": "r2",
        "one_forward_all_checkpoints": True,
        "evo2_ckpt": evo2_ckpt,
        "sae_ckpt": sae_ckpt,
        "layer_name": layer_name,
        "max_batch_tokens": max_batch_tokens,
        "embed_batch_size": 1,
        "num_sequences": len(df),
        "num_checkpoints": len(models),
        "feature": args.feature,
        "saved_files": saved_files,
        "checkpoints": result["checkpoint_summaries"],
        "per_sequence_r2": result["per_sequence"],
    }

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation JSON: {out_json}")
    if args.save_predictions:
        print(f"Saved R² / predictions under stem: {out_json.with_suffix('')}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One forward per sequence; evaluate all `{results_name}_N.json` "
            "checkpoints and save every R² + prediction."
        )
    )
    parser.add_argument(
        "--results_name",
        type=str,
        default=DEFAULT_RESULTS_NAME,
        help=(
            "Checkpoint basename without iteration suffix, e.g. "
            "Evo2_7b_262k_layer26_phyloP_neuron_parameters "
            "(loads {name}_0.json ... {name}_N.json)."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing `{results_name}_N.json` files.",
    )
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument("--label_npy", type=str, default=str(DEFAULT_LABEL_NPY))
    parser.add_argument("--feature", type=str, default="phyloP")
    parser.add_argument("--evo2_ckpt", type=str, default=None)
    parser.add_argument("--sae_ckpt", type=str, default=None)
    parser.add_argument("--layer_name", type=str, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument(
        "--save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save labels + all checkpoint predictions to npy (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
