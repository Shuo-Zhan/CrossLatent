#!/usr/bin/env python3
"""
Evaluate boosting neuron predictions on held-out test CSV (streaming, no embedding.npy).

Loads best single-neuron and combination settings from train results JSON, runs ESM2
embed + SAE encode per streaming batch (same as ESM2_train_neuron_boosting_seq.py),
and computes macro F1 per sequence.

Example:
    python ESM2_eval_neuron_boosting_seq.py \\
        --results_json data/Protein/clan_CL0023/results/ESM2_650M_boosting_neuron_feature_results.json \\
        --csv_path data/Protein/clan_CL0023/CL0023_test.csv \\
        --feature mask_protein_domains
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from interplm.embedders import get_embedder  # pyright: ignore[reportMissingImports]
from interplm.sae.inference import load_sae_from_hf  # pyright: ignore[reportMissingImports]
from ESM2_eval_neuron_boosting import (
    SequenceMacroF1,
    align_per_sequence_results,
    build_combination_prediction_for_sequence,
    evaluation_to_dict,
    load_results_json,
    parse_binary_label,
    resolve_single_neuron_config,
    sequence_macro_f1_from_predictions,
    summarize_evaluation,
)
from neuron_boosting import load_evaluation_dataframe, validate_row_labels
from neuron_boosting_seq import (  # pyright: ignore[reportMissingImports]
    StreamingContext,
    _encode_batch,
    build_streaming_context,
    compute_all_labels_from_df,
)

DATA_DIR = SCRIPT_DIR.parents[1] / "data"
SAVE_DIR = SCRIPT_DIR.parents[1] / "fig"

DEFAULT_RESULTS_JSON = (
    SAVE_DIR
    / 'protein_domains'
    / "ESM2_650M_33_mask_protein_domains_neuron_parameters_pfam00006_100.json"
)
# /data1/zshuo23/GraBSpArk/data/Protein/top20_pfam_families/Anticodon_binding_domain_2290.csv
# DEFAULT_CSV = DATA_DIR / "Protein" / "clan_CL0023" / "CL0023_test.csv"
DEFAULT_CSV = DATA_DIR / "Protein" / "top20_pfam_families" / "Protein_kinase_domain_2151.csv"
DEFAULT_ESM2_MODEL_DIR = INTERPLM_ROOT / "esm2_model" / "esm2_650M_UR50D"
# DEFAULT_OUTPUT_JSON = (
#     DATA_DIR
#     / "Protein"
#     / "clan_CL0023"
#     / "results"
#     / "ESM2_650M_boosting_Protein_kinase_domain_evaluation.json"
# )
DEFAULT_OUTPUT_JSON = (
    SAVE_DIR
    / 'protein_domains'
    / "ESM2_650M_33_eval_macroF1_pfam00006_100_pfam00069.json"
)
DEFAULT_OUTPUT_PLOT = (
    SAVE_DIR
    / 'protein_domains'
    / "ESM2_650M_33_eval_macroF1_pfam00006_100_pfam00069.png"
)

# ESM2 default max_length=1024 → at most 1022 residue tokens after CLS/EOS.
MAX_SEQ_LEN = 1022


def plot_single_vs_combination_macro_f1(
    single_f1: np.ndarray,
    combo_f1: np.ndarray,
    *,
    output_path: str,
    feature: str,
    single_neuron_id: int,
    single_threshold: float,
    combo_neuron_ids: list[int],
    combo_threshold: float,
) -> str:
    """Scatter plot: x=single-neuron macro F1, y=combination macro F1 per test sequence."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        single_f1,
        combo_f1,
        s=10,
        alpha=0.35,
        edgecolors="none",
        c="#2563eb",
    )

    lo = float(min(0.0, np.min(single_f1), np.min(combo_f1)))
    hi = float(max(1.0, np.max(single_f1), np.max(combo_f1)))
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    ax.plot([lo, hi], [lo, hi], "--", color="#6b7280", linewidth=1.0, label="y = x")

    n = len(single_f1)
    n_combo_better = int(np.sum(combo_f1 > single_f1))
    n_single_better = int(np.sum(single_f1 > combo_f1))
    n_tie = int(np.sum(np.isclose(single_f1, combo_f1)))
    mean_delta = float(np.mean(combo_f1 - single_f1))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Single-neuron macro F1")
    ax.set_ylabel("Combination macro F1")
    ax.set_title(
        f"Per-sequence macro F1 on test set ({feature})\n"
        f"single: neuron {single_neuron_id}, threshold={single_threshold:.1f} | "
        f"combo: {len(combo_neuron_ids)} neurons, threshold={combo_threshold:.1f}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.98,
        (
            f"n={n}\n"
            f"combo > single: {n_combo_better}\n"
            f"single > combo: {n_single_better}\n"
            f"tie: {n_tie}\n"
            f"mean single={single_f1.mean():.3f}\n"
            f"mean combo={combo_f1.mean():.3f}\n"
            f"meanΔ (y-x)={mean_delta:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def filter_dataframe_by_max_length(
    df: pd.DataFrame,
    max_len: int = MAX_SEQ_LEN,
) -> tuple[pd.DataFrame, int]:
    """Keep rows with len(sequence) <= max_len; return filtered df and skip count."""
    seq_lens = df["sequence"].astype(str).str.len()
    keep = seq_lens <= max_len
    skipped = int((~keep).sum())
    if skipped:
        print(
            f"Skipping {skipped} sequence(s) with length > {max_len} "
            f"(ESM2 max effective length)"
        )
    filtered = df.loc[keep].reset_index(drop=True)
    return filtered, skipped


def load_models(
    *,
    esm_model_dir: str,
    plm_model: str,
    plm_layer: int,
    device: torch.device,
) -> tuple[Any, torch.nn.Module]:
    embedder = get_embedder(
        "esm",
        model_name=esm_model_dir,
        device=str(device),
    )
    sae = load_sae_from_hf(
        plm_model=plm_model,
        plm_layer=plm_layer,
    ).to(device)
    sae.eval()
    print(f"Loaded ESM2 embedder from: {esm_model_dir}")
    print(
        f"Loaded SAE: plm_model={plm_model}, plm_layer={plm_layer}, "
        f"dict_size={sae.dict_size}, activation_dim={sae.activation_dim}"
    )
    print(f"Device: {device}")
    return embedder, sae


@torch.no_grad()
def encode_and_evaluate_by_sequence_streaming(
    ctx: StreamingContext,
    df: pd.DataFrame,
    seq_lengths: list[int],
    feature: str,
    combo_cfg: dict,
    single_cfg: dict | None,
) -> tuple[list[SequenceMacroF1], list[SequenceMacroF1] | None]:
    """
    Streaming ESM2 + SAE encode by row_batches, then per-sequence prediction and macro F1.
    """
    combo_neuron_ids = [int(x) for x in combo_cfg["neuron_ids"]]
    combo_coefficients = [float(x) for x in combo_cfg["coefficients"]]
    combo_intercept = float(combo_cfg["intercept_b"])
    combo_threshold = float(combo_cfg["best_threshold"])

    single_neuron_id = None
    single_threshold = None
    if single_cfg is not None:
        single_neuron_id = int(single_cfg["neuron_id"])
        single_threshold = float(single_cfg["best_threshold"])

    print(
        f"Batched {len(seq_lengths)} sequences into "
        f"{len(ctx.row_batches)} streaming encode pass(es)"
    )

    combo_results: list[SequenceMacroF1] = []
    single_results: list[SequenceMacroF1] | None = [] if single_cfg is not None else None

    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc="Streaming encode + evaluate",
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
                    f"sequence length={seq_len}. "
                    "Check ESM max_length (1024) or sequence/label alignment."
                )

            y_seq = parse_binary_label(df.iloc[local_row_idx][feature])
            if y_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: label len={y_seq.shape[0]} != seq_len={seq_len}"
                )

            uniprot_id = str(df.iloc[local_row_idx].get("uniprot_id", local_row_idx))

            combo_continuous = build_combination_prediction_for_sequence(
                f_seq=f_seq,
                neuron_ids=combo_neuron_ids,
                coefficients=combo_coefficients,
                intercept_b=combo_intercept,
            )
            combo_binary = combo_continuous >= combo_threshold
            combo_results.append(
                sequence_macro_f1_from_predictions(
                    y_seq,
                    combo_binary,
                    row_index=local_row_idx,
                    uniprot_id=uniprot_id,
                    seq_length=seq_len,
                )
            )

            if single_cfg is not None:
                single_binary = f_seq[:, single_neuron_id] >= single_threshold
                single_results.append(
                    sequence_macro_f1_from_predictions(
                        y_seq,
                        single_binary,
                        row_index=local_row_idx,
                        uniprot_id=uniprot_id,
                        seq_length=seq_len,
                    )
                )

    if len(combo_results) != len(seq_lengths):
        raise ValueError(
            f"Expected {len(seq_lengths)} per-sequence results, got {len(combo_results)}"
        )

    return combo_results, single_results


def run_evaluation(args: argparse.Namespace) -> dict:
    payload = load_results_json(args.results_json, args.feature)
    config = payload["config"]
    feature_result = payload["features"][args.feature]

    df = load_evaluation_dataframe(args.csv_path)
    num_sequences_total = len(df)
    df, num_skipped_long = filter_dataframe_by_max_length(
        df,
        max_len=args.max_seq_len,
    )
    if df.empty:
        raise ValueError(
            f"No sequences left after filtering len(sequence) <= {args.max_seq_len}"
        )
    validate_row_labels(df, args.feature)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Evaluating sequences: {len(seq_lengths)} "
        f"(skipped {num_skipped_long} with len>{args.max_seq_len}), "
        f"total tokens: {sum(seq_lengths)}"
    )

    plm_model = args.plm_model or config.get("plm_model", "esm2-650m")
    plm_layer = int(args.plm_layer or config.get("plm_layer", 33))
    esm_model_dir = args.esm_model_dir or config.get("esm_model_dir", str(DEFAULT_ESM2_MODEL_DIR))
    max_batch_tokens = int(
        args.max_batch_tokens or config.get("max_batch_tokens", 128000)
    )
    embed_batch_size = int(
        args.embed_batch_size or config.get("embed_batch_size", 24)
    )

    device = torch.device(args.device)
    embedder, sae = load_models(
        esm_model_dir=esm_model_dir,
        plm_model=plm_model,
        plm_layer=plm_layer,
        device=device,
    )
    print(f"Max batch tokens: {max_batch_tokens}, embed batch size: {embed_batch_size}")

    labels = compute_all_labels_from_df(df, [args.feature])
    ctx = build_streaming_context(
        df=df,
        y=labels[args.feature],
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=plm_layer,
        max_batch_tokens=max_batch_tokens,
        embed_batch_size=embed_batch_size,
    )

    combo_cfg = feature_result["combination"]
    single_cfg = resolve_single_neuron_config(feature_result, config)
    has_single_neuron = single_cfg is not None
    if has_single_neuron:
        print(
            f"Single-neuron config found: id={single_cfg['neuron_id']}, "
            f"threshold={float(single_cfg['best_threshold']):.1f}"
        )
    else:
        print(
            "No single-neuron config in results JSON; "
            "skipping single-neuron evaluation and scatter plot."
        )

    combo_threshold = float(combo_cfg["best_threshold"])

    combo_per_sequence, single_per_sequence = encode_and_evaluate_by_sequence_streaming(
        ctx=ctx,
        df=df,
        seq_lengths=seq_lengths,
        feature=args.feature,
        combo_cfg=combo_cfg,
        single_cfg=single_cfg,
    )
    del embedder, sae, ctx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combo_eval = summarize_evaluation(
        method="combination",
        threshold=combo_threshold,
        neuron_ids=[int(x) for x in combo_cfg["neuron_ids"]],
        coefficients=[float(x) for x in combo_cfg["coefficients"]],
        intercept_b=float(combo_cfg["intercept_b"]),
        per_sequence=combo_per_sequence,
    )

    single_eval = None
    single_f1 = None
    combo_f1 = None
    aligned_row_indices = None
    uniprot_ids = None
    single_neuron_id = None
    single_threshold = None
    if single_per_sequence is not None:
        single_neuron_id = int(single_cfg["neuron_id"])
        single_threshold = float(single_cfg["best_threshold"])
        single_eval = summarize_evaluation(
            method="single_neuron",
            threshold=single_threshold,
            neuron_ids=[single_neuron_id],
            coefficients=None,
            intercept_b=None,
            per_sequence=single_per_sequence,
        )
        single_f1, combo_f1, aligned_row_indices, uniprot_ids = align_per_sequence_results(
            combo_per_sequence,
            single_per_sequence,
        )

    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Sequences evaluated: {len(seq_lengths)}")
    print(
        f"Combination (neurons={combo_eval.neuron_ids}, threshold={combo_threshold:.1f}): "
        f"mean macro F1={combo_eval.mean_macro_f1:.6f}"
    )
    if single_eval is not None:
        print(
            f"Single neuron (id={single_neuron_id}, threshold={single_threshold:.1f}): "
            f"mean macro F1={single_eval.mean_macro_f1:.6f}"
        )
    else:
        print("Single-neuron evaluation skipped (not present in train results JSON).")

    plot_path = None
    if not args.no_plot and has_single_neuron and single_eval is not None:
        plot_path = plot_single_vs_combination_macro_f1(
            single_f1,
            combo_f1,
            output_path=args.output_plot,
            feature=args.feature,
            single_neuron_id=single_neuron_id,
            single_threshold=single_threshold,
            combo_neuron_ids=combo_eval.neuron_ids,
            combo_threshold=combo_threshold,
        )
        print(f"Saved scatter plot: {plot_path}")
    elif not args.no_plot and not has_single_neuron:
        print("Skipped scatter plot (requires both combination and single-neuron results).")

    train_json_reference = {
        "combination_best_threshold": combo_threshold,
        "combination_train_macro_f1": float(combo_cfg["best_macro_f1"]),
        "single_best_threshold": (
            float(single_cfg["best_threshold"]) if single_cfg is not None else None
        ),
        "single_train_macro_f1": (
            float(single_cfg["best_macro_f1"]) if single_cfg is not None else None
        ),
    }

    output = {
        "results_json": str(args.results_json),
        "csv_path": str(args.csv_path),
        "streaming": True,
        "esm_model_dir": esm_model_dir,
        "plm_model": plm_model,
        "plm_layer": plm_layer,
        "max_batch_tokens": max_batch_tokens,
        "embed_batch_size": embed_batch_size,
        "max_seq_len": args.max_seq_len,
        "num_sequences_total": num_sequences_total,
        "num_sequences_skipped_long": num_skipped_long,
        "num_sequences": len(df),
        "feature": args.feature,
        "has_single_neuron": has_single_neuron,
        "scatter_plot": plot_path if has_single_neuron else None,
        "scatter_summary": (
            {
                "num_sequences": len(single_f1),
                "mean_single_macro_f1": float(single_f1.mean()),
                "mean_combination_macro_f1": float(combo_f1.mean()),
                "mean_delta_combo_minus_single": float(np.mean(combo_f1 - single_f1)),
                "num_combination_better": int(np.sum(combo_f1 > single_f1)),
                "num_single_better": int(np.sum(single_f1 > combo_f1)),
                "num_tie": int(np.sum(np.isclose(single_f1, combo_f1))),
            }
            if single_eval is not None
            else None
        ),
        "scatter_points": (
            [
                {
                    "row_index": int(aligned_row_indices[i]),
                    "uniprot_id": uniprot_ids[i],
                    "single_macro_f1": float(single_f1[i]),
                    "combination_macro_f1": float(combo_f1[i]),
                }
                for i in range(len(single_f1))
            ]
            if single_eval is not None
            else None
        ),
        "train_json_reference": train_json_reference,
        "combination": evaluation_to_dict(combo_eval),
        "single_neuron": (
            evaluation_to_dict(single_eval) if single_eval is not None else None
        ),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation JSON: {out_path}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate boosting JSON predictions on held-out CSV "
            "(streaming ESM2 + SAE, no embedding.npy)."
        )
    )
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--esm_model_dir", type=str, default=str(DEFAULT_ESM2_MODEL_DIR))
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_protein_domains",
        help="Label column / JSON feature key.",
    )
    parser.add_argument("--plm_model", type=str, default=None)
    parser.add_argument("--plm_layer", type=int, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=128000)
    parser.add_argument("--embed_batch_size", type=int, default=24)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Skip sequences longer than this (default: 1022, ESM2 limit).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save per-sequence evaluation JSON.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=str(DEFAULT_OUTPUT_PLOT),
        help="Path to save single-vs-combination macro F1 scatter plot.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip scatter plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_evaluation(args)


if __name__ == "__main__":
    main()
