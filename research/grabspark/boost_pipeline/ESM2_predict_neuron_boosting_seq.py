#!/usr/bin/env python3
"""
Generate per-residue 0/1 combo (+ optional single) predictions on a CSV
(streaming ESM2 + SAE).

Loads combination / best-single neuron settings from train results JSON, runs
streaming encode, applies trained thresholds, and writes uniprot_id / sequence /
pred_combo_* / pred_single_* masks to CSV. If best_single_neuron is absent,
pred_single_* is filled with NaN. No macro F1 evaluation.

Example:
    python ESM2_predict_neuron_boosting_seq.py \\
        --results_json fig/disulfide_bonds/ESM2_650M_33_mask_disulfide_bonds_neuron_parameters_pfam04055_10.json \\
        --csv_path data/Protein/clan_CL0036/CL0036_test.csv \\
        --feature mask_disulfide_bonds \\
        --output_csv fig/disulfide_bonds/CL0036_test_combo_pred.csv
"""

from __future__ import annotations

import argparse
import csv
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
    build_combination_prediction_for_sequence,
    load_results_json,
    resolve_single_neuron_config,
)
from neuron_boosting_seq import (  # pyright: ignore[reportMissingImports]
    StreamingContext,
    _encode_batch,
    build_streaming_context,
)

DATA_DIR = SCRIPT_DIR.parents[1] / "data"
SAVE_DIR = SCRIPT_DIR.parents[1] / "fig"

DEFAULT_RESULTS_JSON = (
    DATA_DIR
    / "Protein" 
    / "top20_pfam_families/results/ESM2_650M_boosting_ABC_transporter_feature_results_3.json"
)
DEFAULT_CSV = DATA_DIR / "Protein" / "clan_CL0023" / "CL0023_test_drop_all0_mask_protein_domains.csv"
DEFAULT_ESM2_MODEL_DIR = INTERPLM_ROOT / "esm2_model" / "esm2_650M_UR50D"
DEFAULT_OUTPUT_CSV = (
    SAVE_DIR
    / "ESM2_650M_boosting_pfam0005_transporter_feature_results_3_pred.csv"
)

# ESM2_eval_multi_ESM2_650M_33_mask_active_sites_CL0036_0.0097_neuron_parameters_idx0-100_on_CL0023_PF00709_mask_active_sites

MAX_SEQ_LEN = 1022


def prediction_column_name(feature: str, kind: str) -> str:
    """kind: 'combo' | 'single' → pred_{kind}_mask_..."""
    if feature.startswith("mask_"):
        return f"pred_{kind}_{feature}"
    return f"pred_{kind}_mask_{feature}"


def binary_array_to_mask_string(binary: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in binary.astype(bool))


def load_input_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns:
        raise ValueError("CSV must contain a 'sequence' column.")
    if "uniprot_id" not in df.columns:
        df = df.copy()
        df["uniprot_id"] = df.index.astype(str)
    return df


def filter_dataframe_by_max_length(
    df: pd.DataFrame,
    max_len: int = MAX_SEQ_LEN,
) -> tuple[pd.DataFrame, int]:
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
def encode_and_predict_streaming(
    ctx: StreamingContext,
    df: pd.DataFrame,
    seq_lengths: list[int],
    combo_cfg: dict,
    single_cfg: dict | None,
) -> list[dict[str, Any]]:
    combo_neuron_ids = [int(x) for x in combo_cfg["neuron_ids"]]
    combo_coefficients = [float(x) for x in combo_cfg["coefficients"]]
    combo_intercept = float(combo_cfg["intercept_b"])
    combo_threshold = float(combo_cfg["best_threshold"])

    single_neuron_id: int | None = None
    single_threshold: float | None = None
    if single_cfg is not None:
        single_neuron_id = int(single_cfg["neuron_id"])
        single_threshold = float(single_cfg["best_threshold"])

    print(
        f"Batched {len(seq_lengths)} sequences into "
        f"{len(ctx.row_batches)} streaming encode pass(es)"
    )
    print(
        f"Combo neurons={combo_neuron_ids}, threshold={combo_threshold:.1f}, "
        f"intercept={combo_intercept:.6f}"
    )
    if single_cfg is not None:
        print(
            f"Single neuron={single_neuron_id}, threshold={single_threshold:.1f}"
        )
    else:
        print("Single neuron: absent in JSON → pred_single will be NaN")

    rows: list[dict[str, Any]] = []

    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc="Streaming encode + predict",
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

            uniprot_id = str(df.iloc[local_row_idx]["uniprot_id"])

            combo_continuous = build_combination_prediction_for_sequence(
                f_seq=f_seq,
                neuron_ids=combo_neuron_ids,
                coefficients=combo_coefficients,
                intercept_b=combo_intercept,
            )
            combo_binary = combo_continuous >= combo_threshold

            if single_cfg is not None:
                single_binary = f_seq[:, single_neuron_id] >= single_threshold
                single_mask: str | float = binary_array_to_mask_string(single_binary)
            else:
                single_mask = float("nan")

            rows.append(
                {
                    "row_index": local_row_idx,
                    "uniprot_id": uniprot_id,
                    "sequence": sequence,
                    "length": seq_len,
                    "combo_mask": binary_array_to_mask_string(combo_binary),
                    "single_mask": single_mask,
                }
            )

    if len(rows) != len(seq_lengths):
        raise ValueError(
            f"Expected {len(seq_lengths)} prediction rows, got {len(rows)}"
        )

    return rows


def run_prediction(args: argparse.Namespace) -> pd.DataFrame:
    payload = load_results_json(args.results_json, args.feature)
    config = payload["config"]
    feature_result = payload["features"][args.feature]
    combo_cfg = feature_result["combination"]
    single_cfg = resolve_single_neuron_config(feature_result, config)

    df = load_input_dataframe(args.csv_path)
    num_sequences_total = len(df)
    df, num_skipped_long = filter_dataframe_by_max_length(
        df,
        max_len=args.max_seq_len,
    )
    if df.empty:
        raise ValueError(
            f"No sequences left after filtering len(sequence) <= {args.max_seq_len}"
        )

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Predicting sequences: {len(seq_lengths)} "
        f"(skipped {num_skipped_long} with len>{args.max_seq_len}), "
        f"total tokens: {sum(seq_lengths)}"
    )

    plm_model = args.plm_model or config.get("plm_model", "esm2-650m")
    plm_layer = int(args.plm_layer or config.get("plm_layer", 33))
    esm_model_dir = args.esm_model_dir or config.get(
        "esm_model_dir", str(DEFAULT_ESM2_MODEL_DIR)
    )
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

    dummy_y = np.zeros(sum(seq_lengths), dtype=np.float64)
    ctx = build_streaming_context(
        df=df,
        y=dummy_y,
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=plm_layer,
        max_batch_tokens=max_batch_tokens,
        embed_batch_size=embed_batch_size,
    )

    pred_rows = encode_and_predict_streaming(
        ctx=ctx,
        df=df,
        seq_lengths=seq_lengths,
        combo_cfg=combo_cfg,
        single_cfg=single_cfg,
    )
    del embedder, sae, ctx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combo_col = prediction_column_name(args.feature, "combo")
    single_col = prediction_column_name(args.feature, "single")
    out_df = pd.DataFrame(pred_rows)[
        ["uniprot_id", "sequence", "length", "combo_mask", "single_mask"]
    ].rename(
        columns={
            "combo_mask": combo_col,
            "single_mask": single_col,
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"\n{'=' * 60}")
    print(f"Feature: {args.feature}")
    print(f"Combo column: {combo_col}")
    print(
        f"Single column: {single_col}"
        + (
            f" (neuron={int(single_cfg['neuron_id'])})"
            if single_cfg is not None
            else " (NaN; no best_single_neuron in JSON)"
        )
    )
    print(f"Total input sequences: {num_sequences_total}")
    print(f"Skipped (len>{args.max_seq_len}): {num_skipped_long}")
    print(f"Predicted sequences: {len(out_df)}")
    print(f"Saved CSV: {output_csv}")

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate combo + single 0/1 prediction masks from train results JSON "
            "(streaming ESM2 + SAE). Missing best_single_neuron → NaN."
        )
    )
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--esm_model_dir", type=str, default=str(DEFAULT_ESM2_MODEL_DIR))
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_protein_domains",
        help=(
            "Feature key in train JSON; output columns become "
            "pred_combo_<feature> and pred_single_<feature>."
        ),
    )
    parser.add_argument("--plm_model", type=str, default=None)
    parser.add_argument("--plm_layer", type=int, default=None)
    parser.add_argument("--max_batch_tokens", type=int, default=128000)
    parser.add_argument("--embed_batch_size", type=int, default=24)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Skip sequences longer than this (default: 1022).",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_prediction(args)


if __name__ == "__main__":
    main()

# sequence,length,
# mask_active_sites,
# pred_combo_k1_mask_active_sites,
# pred_combo_k2_mask_active_sites,
# pred_combo_k3_mask_active_sites,
# pred_combo_k4_mask_active_sites,
# pred_single_mask_active_sites,
# combo_max_macro_f1,single_macro_f1,delta_macro_f1
# Q1J1X6
# MPGIAIIGAQWGDEGKGKITDFLAPQADYVVRYQGGANAGHTVTAKGQTFKLNLLPSGVLHPGTVSILGDGMVIDPEKFLAERQNLLDGGLQPELRISDRAHLVLPHHKFVDGRKDFVGTTGRGIGPAYADRARRVGIRFGDLADESVLRERVERLLEAKPNSTRDAGWTSVTDALGYLLPIRDALLPFVGDTGAQLRQAIREGRNVLFEGAQATLLDLNYGTYPFVTSSHPTVGGILVGAGVNHKAINKVYGVAKAFNTRVGHGPFPTEVFGEMETRLRGDGSNPWDEFGTTTGRARRVGWLDLALLKYAVDVNGLDGLVINKMDILAGLDTVKVGVGYNAAGQPVYRELPGWATTAGAESRATLPKEAQAYLDLIEETVNCPVVIFSCGPAREQTYGAVSWD,
# 404,
# 00000000000010000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 00000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 00000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 00000000000010010000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
# 0.8993773349937734,0.4987593052109181,0.40061802978285527
