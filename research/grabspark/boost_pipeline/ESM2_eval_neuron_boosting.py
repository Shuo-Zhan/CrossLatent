#!/usr/bin/env python3
"""
Evaluate boosting neuron predictions on held-out test CSV and embeddings.

Loads best single-neuron and combination settings from esm2_boosting JSON,
batch-encodes ESM2 embeddings through InterPLM SAE by max_batch_tokens (same as
ESM2_boosting_neuron_feature_r2.py), splits each batch back into sequences, and
computes macro F1 per sequence.

Example:
    python ESM2_eval_neuron_boosting.py

    python ESM2_eval_neuron_boosting.py \\
        --results_json data/uniprot_sprot_2500_2500/esm2_boosting_neuron_feature_results.json \\
        --csv_path data/uniprot_sprot_2500_2500/alpha_helix_labels_test_2500.csv \\
        --embedding_npy data/uniprot_sprot_2500_2500/ESM2_embedding_test_2500.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from neuron_boosting import (
    build_sequence_batches, encode_features,
    load_evaluation_dataframe,
    validate_csv_embedding_pairing,
    validate_row_labels,
    )
from interplm.sae.inference import load_sae_from_hf  # pyright: ignore[reportMissingImports]

DATA_DIR = SCRIPT_DIR.parents[1] / "data"
DEFAULT_RESULTS_JSON = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "results" / "ESM2_650M_boosting_neuron_feature_results_1000.json"
DEFAULT_CSV = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "alpha_helix_labels_test_2500.csv"
DEFAULT_EMBEDDING_NPY = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "ESM2_650M_embedding_test_2500.npy"
DEFAULT_OUTPUT_JSON = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "results" / "ESM2_650M_boosting_test_evaluation_1000.json"
DEFAULT_OUTPUT_PLOT = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "results" / "ESM2_650M_boosting_test_evaluation_scatter_1000.png"


@dataclass
class SequenceMacroF1:
    row_index: int
    uniprot_id: str
    seq_length: int
    macro_f1: float
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class PredictionEvaluation:
    method: str
    threshold: float
    neuron_ids: list[int]
    coefficients: list[float] | None
    intercept_b: float | None
    mean_macro_f1: float
    per_sequence: list[SequenceMacroF1]


def parse_binary_label(label_str: str) -> np.ndarray:
    values = []
    for ch in str(label_str):
        if ch == "1":
            values.append(True)
        elif ch == "0":
            values.append(False)
        else:
            raise ValueError(f"Invalid label character '{ch}', expected '0' or '1'.")
    return np.asarray(values, dtype=bool)


def macro_f1_for_sequence(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int, int, int]:
    """Compute macro F1 for one sequence from binary predictions."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    f1_pos_den = 2 * tp + fp + fn
    f1_pos = (2 * tp / f1_pos_den) if f1_pos_den > 0 else 0.0
    f1_neg_den = 2 * tn + fp + fn
    f1_neg = (2 * tn / f1_neg_den) if f1_neg_den > 0 else 0.0
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return float(macro_f1), tp, tn, fp, fn


def sequence_macro_f1_from_predictions(
    y_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    *,
    row_index: int,
    uniprot_id: str,
    seq_length: int,
) -> SequenceMacroF1:
    macro_f1, tp, tn, fp, fn = macro_f1_for_sequence(y_seq, y_pred_seq)
    return SequenceMacroF1(
        row_index=row_index,
        uniprot_id=uniprot_id,
        seq_length=seq_length,
        macro_f1=macro_f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def load_results_json(path: str, feature: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if feature not in payload.get("features", {}):
        available = list(payload.get("features", {}).keys())
        raise ValueError(f"Feature {feature!r} not in JSON. Available: {available}")
    return payload


def resolve_single_neuron_config(feature_result: dict, config: dict) -> dict | None:
    """
    Detect whether train results JSON includes a usable best-single-neuron config.

    Returns None when search was disabled or best_single_neuron is absent/null.
    """
    if not config.get("search_best_single_neuron", True):
        return None

    single_cfg = feature_result.get("best_single_neuron")
    if single_cfg is None:
        return None

    required_keys = ("neuron_id", "best_threshold", "best_macro_f1")
    missing = [key for key in required_keys if key not in single_cfg]
    if missing:
        raise ValueError(
            f"best_single_neuron in results JSON is incomplete, missing: {missing}"
        )
    return single_cfg


def build_combination_prediction_for_sequence(
    f_seq: np.ndarray,
    neuron_ids: list[int],
    coefficients: list[float],
    intercept_b: float,
) -> np.ndarray:
    if not neuron_ids:
        return np.full(len(f_seq), intercept_b, dtype=np.float64)
    activations = f_seq[:, neuron_ids].astype(np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    return intercept_b + activations @ coef


@torch.no_grad()
def encode_and_evaluate_by_sequence(
    model: torch.nn.Module,
    embedding_npy: str,
    df: pd.DataFrame,
    seq_lengths: list[int],
    feature: str,
    combo_cfg: dict,
    single_cfg: dict | None,
    device: str,
    max_batch_tokens: int,
) -> tuple[list[SequenceMacroF1], list[SequenceMacroF1] | None]:
    """
    Batch-encode embeddings by max_batch_tokens (same as ESM2_boosting), then split
    each batch back into sequences for per-sequence prediction and macro F1.
    """
    row_batches = build_sequence_batches(seq_lengths, max_batch_tokens)
    token_offsets = np.cumsum([0] + seq_lengths)
    all_embeddings = np.load(embedding_npy, mmap_mode="r")

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
        f"{len(row_batches)} model forward pass(es)"
    )
    print(f"Max batch tokens: {max_batch_tokens}")

    combo_results: list[SequenceMacroF1] = []
    single_results: list[SequenceMacroF1] | None = [] if single_cfg is not None else None

    for batch_row_indices in tqdm(
        row_batches,
        desc="Encoding sequence batches",
    ):
        batch_start = int(token_offsets[batch_row_indices[0]])
        batch_end = int(token_offsets[batch_row_indices[-1] + 1])
        batch_embedding = np.asarray(all_embeddings[batch_start:batch_end], dtype=np.float32)
        batch_features = encode_features(model, batch_embedding, device=device)

        offset = 0
        for local_row_idx in batch_row_indices:
            seq_len = seq_lengths[local_row_idx]
            f_seq = batch_features[offset : offset + seq_len]
            offset += seq_len

            y_seq = parse_binary_label(df.iloc[local_row_idx][feature])
            if y_seq.shape[0] != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: label len={y_seq.shape[0]} != seq_len={seq_len}"
                )

            row_idx = local_row_idx
            uniprot_id = str(df.iloc[local_row_idx].get("uniprot_id", row_idx))

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
                    row_index=row_idx,
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
                        row_index=row_idx,
                        uniprot_id=uniprot_id,
                        seq_length=seq_len,
                    )
                )

    if len(combo_results) != len(seq_lengths):
        raise ValueError(
            f"Expected {len(seq_lengths)} per-sequence results, got {len(combo_results)}"
        )

    return combo_results, single_results


def summarize_evaluation(
    method: str,
    threshold: float,
    neuron_ids: list[int],
    coefficients: list[float] | None,
    intercept_b: float | None,
    per_sequence: list[SequenceMacroF1],
) -> PredictionEvaluation:
    mean_macro_f1 = float(np.mean([item.macro_f1 for item in per_sequence]))
    return PredictionEvaluation(
        method=method,
        threshold=float(threshold),
        neuron_ids=neuron_ids,
        coefficients=coefficients,
        intercept_b=intercept_b,
        mean_macro_f1=mean_macro_f1,
        per_sequence=per_sequence,
    )


def evaluation_to_dict(result: PredictionEvaluation) -> dict:
    return {
        "method": result.method,
        "threshold": result.threshold,
        "neuron_ids": result.neuron_ids,
        "coefficients": result.coefficients,
        "intercept_b": result.intercept_b,
        "mean_macro_f1": result.mean_macro_f1,
        "per_sequence": [asdict(item) for item in result.per_sequence],
    }


def align_per_sequence_results(
    combo_per_sequence: list[SequenceMacroF1],
    single_per_sequence: list[SequenceMacroF1],
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Align combination and single-neuron results by row_index."""
    combo_by_row = {item.row_index: item for item in combo_per_sequence}
    single_by_row = {item.row_index: item for item in single_per_sequence}
    if combo_by_row.keys() != single_by_row.keys():
        raise ValueError("Combination and single-neuron results cover different sequences.")

    row_indices = sorted(combo_by_row.keys())
    combo_f1 = np.asarray([combo_by_row[idx].macro_f1 for idx in row_indices], dtype=np.float64)
    single_f1 = np.asarray([single_by_row[idx].macro_f1 for idx in row_indices], dtype=np.float64)
    uniprot_ids = [combo_by_row[idx].uniprot_id for idx in row_indices]
    return single_f1, combo_f1, row_indices, uniprot_ids


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
            f"mean combo={combo_f1.mean():.3f}"
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


def run_evaluation(args: argparse.Namespace) -> dict:
    payload = load_results_json(args.results_json, args.feature)
    config = payload["config"]
    feature_result = payload["features"][args.feature]

    df = load_evaluation_dataframe(args.csv_path)
    validate_row_labels(df, args.feature)
    validate_csv_embedding_pairing(df, args.embedding_npy)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(f"Total sequences: {len(seq_lengths)}, total tokens: {sum(seq_lengths)}")

    device = torch.device(args.device)
    model = load_sae_from_hf(
        plm_model=args.plm_model or config.get("plm_model", "esm2-8m"),
        plm_layer=int(args.plm_layer or config.get("plm_layer", 6)),
    ).to(device)
    model.eval()
    print(
        f"Loaded SAE: plm_model={args.plm_model or config['plm_model']}, "
        f"plm_layer={args.plm_layer or config['plm_layer']}, dict_size={model.dict_size}"
    )
    print(f"SAE device: {device}")

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

    combo_per_sequence, single_per_sequence = encode_and_evaluate_by_sequence(
        model=model,
        embedding_npy=args.embedding_npy,
        df=df,
        seq_lengths=seq_lengths,
        feature=args.feature,
        combo_cfg=combo_cfg,
        single_cfg=single_cfg,
        device=str(device),
        max_batch_tokens=args.max_batch_tokens,
    )
    del model
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
        "embedding_npy": str(args.embedding_npy),
        "num_sequences": len(df),
        "feature": args.feature,
        "has_single_neuron": has_single_neuron,
        "scatter_plot": plot_path if has_single_neuron else None,
        "scatter_summary": (
            {
                "num_sequences": len(single_f1),
                "mean_single_macro_f1": float(single_f1.mean()),
                "mean_combination_macro_f1": float(combo_f1.mean()),
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
        description="Evaluate boosting JSON predictions on held-out CSV/embeddings."
    )
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--embedding_npy", type=str, default=str(DEFAULT_EMBEDDING_NPY))
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_alpha_helix",
        help="Label column / JSON feature key.",
    )
    parser.add_argument("--plm_model", type=str, default=None)
    parser.add_argument("--plm_layer", type=int, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=128000)
    parser.add_argument("--device", type=str, default="cuda:0")
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
    run_evaluation(args)


if __name__ == "__main__":
    main()
