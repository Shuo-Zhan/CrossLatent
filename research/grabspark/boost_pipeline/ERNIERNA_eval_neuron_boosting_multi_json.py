#!/usr/bin/env python3
"""
Evaluate a range of ERNIERNA boosting JSONs on one CSV + precomputed embedding.npy.

Given a JSON name stem and an inclusive index range, loads
``{json_dir}/{json_name}_{i}.json`` for each i, runs SAE encode **once** on the
embedding matrix (no sequence→PLM forward), then:

- for each JSON: mean per-sequence combo macro F1
- once for the whole run: mean per-sequence single-neuron macro F1
  (same training series shares ``best_single_neuron`` across checkpoints)

Example:
    python ERNIERNA_eval_neuron_boosting_multi_json.py \\
        --csv_path data/RNA/Rfam/ERNIERNA_embeddings/Rfam_test.csv \\
        --embedding_npy data/RNA/Rfam/ERNIERNA_embeddings/layer_12/Rfam_test.npy \\
        --json_dir data/RNA/Rfam/results/mask_angle \\
        --json_name ERNIERNA_mask_angle_results \\
        --index_start 0 \\
        --index_end 100 \\
        --feature mask_angle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_TRAIN_DIR = SCRIPT_DIR.parent / "sae_train"
DATA_DIR = SCRIPT_DIR.parents[1] / "data"

sys.path.insert(0, str(SAE_TRAIN_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from sae_models import load_sae_from_checkpoint  # pyright: ignore[reportMissingImports]
from ERNIERNA_eval_neuron_boosting import (
    build_combination_prediction_for_sequence,
    load_evaluation_dataframe,
    load_results_json,
    parse_binary_label,
    resolve_single_neuron_config,
    row_sequence_id,
    sequence_macro_f1_from_predictions,
)
from neuron_boosting import (
    build_sequence_batches,
    encode_features,
    validate_csv_embedding_pairing,
    validate_row_labels,
)

RFAM_DIR = DATA_DIR / "RNA" / "Rfam" / "ERNIERNA_embeddings"
DEFAULT_CSV = RFAM_DIR / "Rfam_test.csv"
DEFAULT_EMBEDDING_NPY = RFAM_DIR / "layer_12" / "Rfam_test.npy"
DEFAULT_JSON_DIR = DATA_DIR / "RNA" / "Rfam" / "results" / "mask_angle_8"
DEFAULT_JSON_NAME = "ERNIERNA_mask_angle_results"
# DEFAULT_SAE_CHECKPOINT = (
#     SCRIPT_DIR.parents[1]
#     / "ckpt"
#     / "ERNIE-RNA"
#     / "layer_12_token"
#     / "relu_12x_dim768_20260602_013912"
#     / "best.pt"
# )
DEFAULT_SAE_CHECKPOINT = (
    SCRIPT_DIR.parents[1]
    / "ckpt"
    / "ERNIE-RNA"
    / "layer_12_token"
    / "relu_8x_dim768_20260604_014527"
    / "best.pt"
)

@dataclass
class JsonPredictor:
    """One results JSON combo predictor (single-neuron is shared across the series)."""

    index: int
    path: Path
    combo_neuron_ids: list[int]
    combo_coefficients: list[float]
    combo_intercept: float
    combo_threshold: float
    combo_train_macro_f1: float | None
    combo_f1_sum: float = 0.0
    num_sequences: int = 0


@dataclass
class SharedSingleNeuron:
    """Single-neuron predictor shared by all JSONs in one training series."""

    neuron_id: int
    threshold: float
    train_macro_f1: float | None
    source_json: Path
    f1_sum: float = 0.0
    num_sequences: int = 0


def resolve_json_paths(
    json_dir: Path,
    json_name: str,
    index_start: int,
    index_end: int,
) -> list[tuple[int, Path]]:
    if index_end < index_start:
        raise ValueError(f"index_end ({index_end}) < index_start ({index_start})")

    pairs: list[tuple[int, Path]] = []
    missing: list[str] = []
    for idx in range(index_start, index_end + 1):
        path = json_dir / f"{json_name}_{idx}.json"
        if path.exists():
            pairs.append((idx, path))
        else:
            missing.append(str(path))

    if not pairs:
        raise FileNotFoundError(
            f"No JSON found for {json_name}_{{{index_start}..{index_end}}}.json "
            f"under {json_dir}"
        )
    if missing:
        print(f"Warning: {len(missing)} JSON file(s) missing (skipped):")
        for path in missing[:10]:
            print(f"  - {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    return pairs


def load_predictors_and_shared_single(
    json_pairs: list[tuple[int, Path]],
    feature: str,
) -> tuple[list[JsonPredictor], SharedSingleNeuron | None]:
    """
    Load combo configs for every JSON; take single-neuron once from the first JSON,
    and verify later JSONs match (same training series should be identical).
    """
    predictors: list[JsonPredictor] = []
    shared_single: SharedSingleNeuron | None = None

    for idx, path in json_pairs:
        payload = load_results_json(str(path), feature)
        feature_result = payload["features"][feature]
        combo_cfg = feature_result["combination"]
        single_cfg = resolve_single_neuron_config(
            feature_result, payload.get("config", {})
        )

        predictors.append(
            JsonPredictor(
                index=idx,
                path=path,
                combo_neuron_ids=[int(x) for x in combo_cfg["neuron_ids"]],
                combo_coefficients=[float(x) for x in combo_cfg["coefficients"]],
                combo_intercept=float(combo_cfg["intercept_b"]),
                combo_threshold=float(combo_cfg["best_threshold"]),
                combo_train_macro_f1=(
                    float(combo_cfg["best_macro_f1"])
                    if "best_macro_f1" in combo_cfg
                    else None
                ),
            )
        )

        if single_cfg is None:
            continue

        neuron_id = int(single_cfg["neuron_id"])
        threshold = float(single_cfg["best_threshold"])
        train_f1 = (
            float(single_cfg["best_macro_f1"]) if "best_macro_f1" in single_cfg else None
        )
        if shared_single is None:
            shared_single = SharedSingleNeuron(
                neuron_id=neuron_id,
                threshold=threshold,
                train_macro_f1=train_f1,
                source_json=path,
            )
        elif (
            shared_single.neuron_id != neuron_id
            or not np.isclose(shared_single.threshold, threshold)
        ):
            raise ValueError(
                f"best_single_neuron mismatch across JSON series: "
                f"{shared_single.source_json.name} has "
                f"(id={shared_single.neuron_id}, thr={shared_single.threshold}), but "
                f"{path.name} has (id={neuron_id}, thr={threshold}). "
                "Refusing to treat them as one shared single-neuron predictor."
            )

    return predictors, shared_single


def resolve_sae_defaults_from_json(
    first_json: Path,
    feature: str,
    args: argparse.Namespace,
) -> tuple[str, str | None, int]:
    """Fill SAE / batch defaults from the first JSON config when CLI omits them."""
    with open(first_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})

    if feature not in payload.get("features", {}):
        available = list(payload.get("features", {}).keys())
        raise ValueError(
            f"Feature {feature!r} not in {first_json}. Available: {available}"
        )

    sae_checkpoint = args.sae_checkpoint or config.get(
        "sae_checkpoint", str(DEFAULT_SAE_CHECKPOINT)
    )
    sae_type = args.sae_type or config.get("sae_type", "relu")
    max_batch_tokens = int(
        args.max_batch_tokens
        if args.max_batch_tokens is not None
        else config.get("max_batch_tokens", 128000)
    )
    return str(sae_checkpoint), sae_type, max_batch_tokens


@torch.no_grad()
def encode_once_evaluate_all_jsons(
    model: torch.nn.Module,
    embedding_npy: str,
    df: pd.DataFrame,
    seq_lengths: list[int],
    feature: str,
    predictors: list[JsonPredictor],
    shared_single: SharedSingleNeuron | None,
    device: str,
    max_batch_tokens: int,
) -> None:
    """
    One batched SAE encode pass over precomputed embeddings.

    For each sequence:
      - apply every JSON's combo predictor (per-JSON mean macro F1)
      - apply shared single-neuron predictor once (series-level mean macro F1)
    """
    row_batches = build_sequence_batches(seq_lengths, max_batch_tokens)
    token_offsets = np.cumsum([0] + seq_lengths)
    all_embeddings = np.load(embedding_npy, mmap_mode="r")

    print(
        f"Batched {len(seq_lengths)} sequences into "
        f"{len(row_batches)} SAE forward pass(es); "
        f"evaluating {len(predictors)} combo JSON(s)"
        + (
            f" + shared single neuron {shared_single.neuron_id}"
            if shared_single is not None
            else " (no single-neuron config)"
        )
    )
    print(f"Max batch tokens: {max_batch_tokens}")

    for batch_row_indices in tqdm(
        row_batches,
        desc="SAE encode + multi-JSON eval",
    ):
        batch_start = int(token_offsets[batch_row_indices[0]])
        batch_end = int(token_offsets[batch_row_indices[-1] + 1])
        batch_embedding = np.asarray(
            all_embeddings[batch_start:batch_end], dtype=np.float32
        )
        batch_features = encode_features(model, batch_embedding, device=device)

        offset = 0
        for local_row_idx in batch_row_indices:
            seq_len = seq_lengths[local_row_idx]
            f_seq = batch_features[offset : offset + seq_len]
            offset += seq_len

            if f_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: SAE activations={f_seq.shape[0]} != "
                    f"sequence length={seq_len}"
                )

            y_seq = parse_binary_label(df.iloc[local_row_idx][feature])
            if y_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: label len={y_seq.shape[0]} != seq_len={seq_len}"
                )

            sequence_id = row_sequence_id(df, local_row_idx)

            for pred in predictors:
                combo_continuous = build_combination_prediction_for_sequence(
                    f_seq=f_seq,
                    neuron_ids=pred.combo_neuron_ids,
                    coefficients=pred.combo_coefficients,
                    intercept_b=pred.combo_intercept,
                )
                combo_binary = combo_continuous >= pred.combo_threshold
                combo_item = sequence_macro_f1_from_predictions(
                    y_seq,
                    combo_binary,
                    row_index=local_row_idx,
                    sequence_id=sequence_id,
                    seq_length=seq_len,
                )
                pred.combo_f1_sum += float(combo_item.macro_f1)
                pred.num_sequences += 1

            if shared_single is not None:
                single_binary = (
                    f_seq[:, shared_single.neuron_id] >= shared_single.threshold
                )
                single_item = sequence_macro_f1_from_predictions(
                    y_seq,
                    single_binary,
                    row_index=local_row_idx,
                    sequence_id=sequence_id,
                    seq_length=seq_len,
                )
                shared_single.f1_sum += float(single_item.macro_f1)
                shared_single.num_sequences += 1

    for pred in predictors:
        if pred.num_sequences != len(seq_lengths):
            raise ValueError(
                f"{pred.path.name}: expected {len(seq_lengths)} sequences, "
                f"got {pred.num_sequences}"
            )
    if shared_single is not None and shared_single.num_sequences != len(seq_lengths):
        raise ValueError(
            f"shared single: expected {len(seq_lengths)} sequences, "
            f"got {shared_single.num_sequences}"
        )


def default_output_json(
    json_dir: Path,
    json_name: str,
    index_start: int,
    index_end: int,
    csv_path: Path,
) -> Path:
    csv_stem = csv_path.stem
    return json_dir / (
        f"ERNIERNA_eval_multi_{json_name}_idx{index_start}-{index_end}"
        f"_on_{csv_stem}.json"
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    json_dir = Path(args.json_dir)
    json_pairs = resolve_json_paths(
        json_dir=json_dir,
        json_name=args.json_name,
        index_start=args.index_start,
        index_end=args.index_end,
    )
    print(
        f"Resolved {len(json_pairs)} JSON(s): "
        f"{args.json_name}_{{{json_pairs[0][0]}..{json_pairs[-1][0]}}}.json"
    )

    sae_checkpoint, sae_type, max_batch_tokens = resolve_sae_defaults_from_json(
        json_pairs[0][1], args.feature, args
    )

    predictors, shared_single = load_predictors_and_shared_single(
        json_pairs, args.feature
    )
    print(f"Loaded {len(predictors)} combo predictors for feature={args.feature!r}")
    if shared_single is not None:
        print(
            f"Shared single-neuron: id={shared_single.neuron_id}, "
            f"threshold={shared_single.threshold:.1f} "
            f"(from {shared_single.source_json.name})"
        )
    else:
        print("No single-neuron config found; skipping single-neuron evaluation.")

    df = load_evaluation_dataframe(args.csv_path)
    validate_row_labels(df, args.feature)
    validate_csv_embedding_pairing(df, args.embedding_npy)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Evaluating sequences: {len(seq_lengths)}, "
        f"total tokens: {sum(seq_lengths)}"
    )

    device = torch.device(args.device)
    model = load_sae_from_checkpoint(
        sae_checkpoint,
        sae_type=sae_type,
        device=device,
    )
    model.eval()
    print(
        f"Loaded SAE: checkpoint={sae_checkpoint}, sae_type={sae_type}, "
        f"dict_size={model.dict_size}, activation_dim={model.activation_dim}"
    )
    print(f"SAE device: {device}")

    encode_once_evaluate_all_jsons(
        model=model,
        embedding_npy=args.embedding_npy,
        df=df,
        seq_lengths=seq_lengths,
        feature=args.feature,
        predictors=predictors,
        shared_single=shared_single,
        device=str(device),
        max_batch_tokens=max_batch_tokens,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_json: list[dict[str, Any]] = []
    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Sequences evaluated: {len(seq_lengths)}")
    for pred in predictors:
        combo_mean = pred.combo_f1_sum / max(pred.num_sequences, 1)
        entry = {
            "index": pred.index,
            "results_json": str(pred.path),
            "num_sequences": pred.num_sequences,
            "combination": {
                "neuron_ids": pred.combo_neuron_ids,
                "coefficients": pred.combo_coefficients,
                "intercept_b": pred.combo_intercept,
                "threshold": pred.combo_threshold,
                "train_macro_f1": pred.combo_train_macro_f1,
                "mean_macro_f1": combo_mean,
            },
        }
        per_json.append(entry)
        print(
            f"[{pred.index:>3d}] {pred.path.name}: "
            f"combo mean macro F1={combo_mean:.6f}"
        )

    single_neuron_out: dict[str, Any] | None = None
    if shared_single is not None:
        single_mean = shared_single.f1_sum / max(shared_single.num_sequences, 1)
        single_neuron_out = {
            "neuron_id": shared_single.neuron_id,
            "threshold": shared_single.threshold,
            "train_macro_f1": shared_single.train_macro_f1,
            "mean_macro_f1": single_mean,
            "source_json": str(shared_single.source_json),
            "num_sequences": shared_single.num_sequences,
        }
        print(
            f"Shared single neuron (id={shared_single.neuron_id}, "
            f"threshold={shared_single.threshold:.1f}): "
            f"mean macro F1={single_mean:.6f}"
        )

    output: dict[str, Any] = {
        "csv_path": str(args.csv_path),
        "embedding_npy": str(args.embedding_npy),
        "json_dir": str(json_dir),
        "json_name": args.json_name,
        "index_start": args.index_start,
        "index_end": args.index_end,
        "resolved_indices": [p.index for p in predictors],
        "encode_once": True,
        "input_is_embedding": True,
        "sae_checkpoint": sae_checkpoint,
        "sae_type": sae_type,
        "max_batch_tokens": max_batch_tokens,
        "num_sequences": len(df),
        "feature": args.feature,
        "single_neuron": single_neuron_out,
        "per_json": per_json,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved multi-JSON evaluation: {out_path}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a range of ERNIERNA boosting JSONs on one CSV + "
            "precomputed embedding.npy with a single SAE encode pass."
        )
    )
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help="Precomputed token embedding matrix [num_tokens, hidden_dim].",
    )
    parser.add_argument("--json_dir", type=str, default=str(DEFAULT_JSON_DIR))
    parser.add_argument(
        "--json_name",
        type=str,
        default=DEFAULT_JSON_NAME,
        help=(
            "JSON stem without trailing _{index}.json, e.g. "
            "ERNIERNA_mask_angle_results"
        ),
    )
    parser.add_argument("--index_start", type=int, default=0)
    parser.add_argument(
        "--index_end",
        type=int,
        default=100,
        help="Inclusive end index.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_angle",
        help="Label column / JSON feature key.",
    )
    parser.add_argument(
        "--sae_checkpoint",
        type=str,
        default=None,
        help="Override SAE checkpoint; defaults to value in first results JSON.",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        default=None,
        help="Override SAE type; defaults to value in first results JSON.",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=None,
        help="Max tokens per SAE forward; default from JSON config or 128000.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output path; default derived under json_dir from stem/indices/csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.output_json is None:
        args.output_json = str(
            default_output_json(
                Path(args.json_dir),
                args.json_name,
                args.index_start,
                args.index_end,
                Path(args.csv_path),
            )
        )
    run_evaluation(args)


if __name__ == "__main__":
    main()
