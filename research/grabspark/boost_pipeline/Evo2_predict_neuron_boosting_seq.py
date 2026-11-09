#!/usr/bin/env python3
"""
Generate per-base continuous combo predictions from raw DNA sequences
(streaming Evo2-7B-262k embed + Layer-26 SAE).

Loads combination coefficients from train results JSON, runs online Evo2
forward + SAE encode (no precomputed embedding.npy), and writes
sequence_id / sequence / continuous pred_combo_* to CSV.
best_single_neuron is not used (pred_single_* filled with NaN).

Example:
    CUDA_VISIBLE_DEVICES=2 python Evo2_predict_neuron_boosting_seq.py \\
        --results_json data/test_sample_3000_a/results/Evo2_7b_262k_layer26_atac_K562_neuron_parameters_5.json \\
        --fasta_path data/test_sample_3000_b/sequences.fasta \\
        --feature atac_K562 \\
        --output_csv data/test_sample_3000_b/results/Evo2_pred_atac_K562.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
import pandas as pd
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
    read_fasta,
)

DEFAULT_RESULTS_JSON = (
    SCRIPT_DIR
    / "data"
    / "test_sample_3000_a"
    / "results"
    / "Evo2_7b_262k_layer26_atac_K562_neuron_parameters_5.json"
)
DEFAULT_FASTA = SCRIPT_DIR / "data" / "test_sample_3000_b" / "sequences.fasta"
DEFAULT_OUTPUT_CSV = (
    SCRIPT_DIR
    / "data"
    / "test_sample_3000_b"
    / "results"
    / "Evo2_7b_262k_layer26_atac_K562_pred.csv"
)


def prediction_column_name(feature: str, kind: str) -> str:
    """kind: 'combo' | 'single' → pred_{kind}_..."""
    if feature.startswith("mask_") or feature.startswith("pred_"):
        return f"pred_{kind}_{feature}"
    return f"pred_{kind}_{feature}"


def continuous_array_to_string(values: np.ndarray, precision: int = 6) -> str:
    return " ".join(f"{float(v):.{precision}g}" for v in values)


def load_results_json(path: str, feature: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "features" not in payload or feature not in payload["features"]:
        raise KeyError(f"Feature '{feature}' not found in {path}")
    return payload


def load_input_dataframe(
    *,
    fasta_path: str | None,
    csv_path: str | None,
) -> pd.DataFrame:
    if fasta_path and csv_path:
        raise ValueError("Provide only one of --fasta_path or --csv_path.")
    if fasta_path:
        ids, sequences = read_fasta(fasta_path)
        return pd.DataFrame(
            {
                "sequence_id": ids,
                "uniprot_id": ids,
                "sequence": sequences,
            }
        )
    if csv_path:
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            raise ValueError("CSV must contain a 'sequence' column.")
        df = df.copy()
        if "uniprot_id" not in df.columns:
            if "sequence_id" in df.columns:
                df["uniprot_id"] = df["sequence_id"].astype(str)
            else:
                df["uniprot_id"] = df.index.astype(str)
        if "sequence_id" not in df.columns:
            df["sequence_id"] = df["uniprot_id"].astype(str)
        return df
    raise ValueError("Must provide --fasta_path or --csv_path.")


def filter_dataframe_by_max_length(
    df: pd.DataFrame,
    max_len: int | None,
) -> tuple[pd.DataFrame, int]:
    if max_len is None or max_len <= 0:
        return df.reset_index(drop=True), 0
    seq_lens = df["sequence"].astype(str).str.len()
    keep = seq_lens <= max_len
    skipped = int((~keep).sum())
    if skipped:
        print(f"Skipping {skipped} sequence(s) with length > {max_len}")
    return df.loc[keep].reset_index(drop=True), skipped


@torch.no_grad()
def encode_and_predict_streaming(
    ctx,
    df: pd.DataFrame,
    seq_lengths: list[int],
    combo_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    combo_neuron_ids = [int(x) for x in combo_cfg["neuron_ids"]]
    combo_coefficients = [float(x) for x in combo_cfg["coefficients"]]
    combo_intercept = float(combo_cfg["intercept_b"])

    print(
        f"Batched {len(seq_lengths)} sequences into "
        f"{len(ctx.row_batches)} streaming encode pass(es)"
    )
    print(
        f"Combo neurons={combo_neuron_ids}, intercept={combo_intercept:.6f}, "
        f"coefficients={[round(c, 6) for c in combo_coefficients]}"
    )
    print("Single neuron: skipped (continuous Evo2 SAE predict)")

    rows: list[dict[str, Any]] = []
    all_preds = np.zeros(int(sum(seq_lengths)), dtype=np.float64)
    token_offsets = np.cumsum([0] + seq_lengths)

    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc="Streaming Evo2+SAE predict",
    ):
        batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
        batch_features = _encode_batch(ctx, batch_sequences)

        offset = 0
        for local_row_idx in batch_row_indices:
            seq_len = seq_lengths[local_row_idx]
            f_seq = batch_features[offset : offset + seq_len]
            offset += seq_len

            if f_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: SAE activations={f_seq.shape[0]} != "
                    f"sequence length={seq_len}."
                )

            sequence = str(df.iloc[local_row_idx]["sequence"])
            if len(sequence) != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: sequence len={len(sequence)} != {seq_len}"
                )

            sequence_id = str(
                df.iloc[local_row_idx].get(
                    "sequence_id",
                    df.iloc[local_row_idx].get("uniprot_id", local_row_idx),
                )
            )
            uniprot_id = str(df.iloc[local_row_idx].get("uniprot_id", sequence_id))

            combo_continuous = build_combination_prediction(
                f_seq=f_seq,
                neuron_ids=combo_neuron_ids,
                coefficients=combo_coefficients,
                intercept_b=combo_intercept,
            )
            tok_start = int(token_offsets[local_row_idx])
            tok_end = int(token_offsets[local_row_idx + 1])
            all_preds[tok_start:tok_end] = combo_continuous

            rows.append(
                {
                    "row_index": local_row_idx,
                    "sequence_id": sequence_id,
                    "uniprot_id": uniprot_id,
                    "sequence": sequence,
                    "length": seq_len,
                    "combo_pred": continuous_array_to_string(combo_continuous),
                    "single_pred": float("nan"),
                }
            )

    if len(rows) != len(seq_lengths):
        raise ValueError(
            f"Expected {len(seq_lengths)} prediction rows, got {len(rows)}"
        )
    return rows, all_preds


def run_prediction(args: argparse.Namespace) -> pd.DataFrame:
    payload = load_results_json(args.results_json, args.feature)
    train_config = payload.get("config", {})
    feature_result = payload["features"][args.feature]
    combo_cfg = feature_result["combination"]

    df = load_input_dataframe(
        fasta_path=args.fasta_path,
        csv_path=args.csv_path,
    )
    num_sequences_total = len(df)
    df, num_skipped_long = filter_dataframe_by_max_length(
        df,
        max_len=args.max_seq_len,
    )
    if args.max_sequences is not None:
        df = df.iloc[: int(args.max_sequences)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No sequences left to predict.")

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Predicting sequences: {len(seq_lengths)} "
        f"(input={num_sequences_total}, skipped_long={num_skipped_long}), "
        f"total tokens: {sum(seq_lengths)}"
    )

    evo2_ckpt = args.evo2_ckpt or train_config.get("evo2_ckpt", DEFAULT_EVO2_CKPT)
    sae_ckpt = args.sae_ckpt or train_config.get("sae_ckpt", DEFAULT_SAE_CKPT)
    layer_name = args.layer_name or train_config.get("layer_name", DEFAULT_LAYER)
    max_batch_tokens = int(
        args.max_batch_tokens
        if args.max_batch_tokens is not None
        else train_config.get("max_batch_tokens", 1024)
    )
    embed_batch_size = int(
        args.embed_batch_size
        if args.embed_batch_size is not None
        else train_config.get("embed_batch_size", 1)
    )

    embedder, sae, device = load_evo2_and_sae(
        evo2_ckpt=evo2_ckpt,
        sae_ckpt=sae_ckpt,
        layer_name=layer_name,
        device=args.device,
    )
    print(f"Max batch tokens: {max_batch_tokens}, embed batch size: {embed_batch_size}")

    dummy_y = np.zeros(sum(seq_lengths), dtype=np.float64)
    ctx = build_streaming_context(
        df=df,
        y=dummy_y,
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=26,
        max_batch_tokens=max_batch_tokens,
        embed_batch_size=embed_batch_size,
    )

    pred_rows, all_preds = encode_and_predict_streaming(
        ctx=ctx,
        df=df,
        seq_lengths=seq_lengths,
        combo_cfg=combo_cfg,
    )
    del embedder, sae, ctx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combo_col = prediction_column_name(args.feature, "combo")
    single_col = prediction_column_name(args.feature, "single")
    out_df = pd.DataFrame(pred_rows)[
        [
            "sequence_id",
            "uniprot_id",
            "sequence",
            "length",
            "combo_pred",
            "single_pred",
        ]
    ].rename(
        columns={
            "combo_pred": combo_col,
            "single_pred": single_col,
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    output_npy = None
    if args.output_npy:
        output_npy = Path(args.output_npy)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        # Store as [n_seq, seq_len] when all lengths equal; else 1D flat.
        unique_lens = set(seq_lengths)
        if len(unique_lens) == 1:
            arr = all_preds.reshape(len(seq_lengths), seq_lengths[0])
        else:
            arr = all_preds
        np.save(output_npy, arr.astype(np.float32))

    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Combo column (continuous): {combo_col}")
    print(f"Single column: {single_col} (NaN)")
    print(f"Total input sequences: {num_sequences_total}")
    print(f"Skipped (len>{args.max_seq_len}): {num_skipped_long}")
    print(f"Predicted sequences: {len(out_df)}")
    print(f"Saved CSV: {output_csv}")
    if output_npy is not None:
        print(f"Saved NPY: {output_npy} shape={arr.shape}")

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict continuous combo scores from raw sequences "
            "(streaming Evo2 + Layer-26 SAE). No embedding.npy required."
        )
    )
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional CSV with a 'sequence' column (mutually exclusive with --fasta_path).",
    )
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--output_npy",
        type=str,
        default=None,
        help="Optional path to save flat/2D continuous predictions as .npy.",
    )
    parser.add_argument("--feature", type=str, default="atac_K562")
    parser.add_argument("--evo2_ckpt", type=str, default=None)
    parser.add_argument("--sae_ckpt", type=str, default=None)
    parser.add_argument("--layer_name", type=str, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=1024)
    parser.add_argument("--embed_batch_size", type=int, default=1)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Skip sequences longer than this (default: no length filter).",
    )
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    args = parser.parse_args()

    # argparse default fasta should be cleared if user only passes csv.
    if args.csv_path is not None:
        # If user explicitly set csv, prefer csv and ignore default fasta unless also set.
        # Detect whether fasta was left as default while csv provided.
        if args.fasta_path == str(DEFAULT_FASTA):
            args.fasta_path = None
    return args


def main() -> None:
    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
