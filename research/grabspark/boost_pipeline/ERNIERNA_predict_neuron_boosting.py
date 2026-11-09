#!/usr/bin/env python3
"""
Generate per-token 0/1 combo (+ optional single) predictions on Rfam CSV
using precomputed ERNIE-RNA embeddings + local SAE checkpoint.

Loads combination / best-single neuron settings from train results JSON,
batch-encodes embedding.npy through SAE, applies trained thresholds, and
writes rfam_name / sequence / pred_combo_* / pred_single_* masks to CSV.
If best_single_neuron is absent, pred_single_* is filled with NaN.
No macro F1 evaluation.

Example:
    python ERNIERNA_predict_neuron_boosting.py \\
        --results_json data/RNA/Rfam/results/mask_bracket/ERNIERNA_mask_bracket_results_1.json \\
        --csv_path data/RNA/Rfam/ERNIERNA_embeddings/Rfam_test.csv \\
        --embedding_npy data/RNA/Rfam/ERNIERNA_embeddings/layer_12/Rfam_test.npy \\
        --feature mask_bracket \\
        --output_csv data/RNA/Rfam/results/mask_bracket/ERNIERNA_mask_bracket_results_1_pred.csv
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
SAE_TRAIN_DIR = SCRIPT_DIR.parent / "sae_train"
DATA_DIR = SCRIPT_DIR.parents[1] / "data"

sys.path.insert(0, str(SAE_TRAIN_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from sae_models import load_sae_from_checkpoint  # pyright: ignore[reportMissingImports]
from ERNIERNA_eval_neuron_boosting import (
    build_combination_prediction_for_sequence,
    load_evaluation_dataframe,
    load_results_json,
    resolve_single_neuron_config,
    row_sequence_id,
)
from neuron_boosting import (
    build_sequence_batches,
    encode_features,
    validate_csv_embedding_pairing,
)

RFAM_DIR = DATA_DIR / "RNA" / "RNASSTR" / "ERNIERNA_embeddings"
DEFAULT_RESULTS_JSON = (
    DATA_DIR
    / "RNA"
    / "RNASSTR"
    / "results"
    / "ERNIERNA_mask_bracket_results_1.json"
)
DEFAULT_CSV = RFAM_DIR / "RNASSTR_len401-1022_test.csv"
DEFAULT_EMBEDDING_NPY = RFAM_DIR / "layer_12" / "RNASSTR_len401-1022_test_token_embedding.npy"
DEFAULT_SAE_CHECKPOINT = (
    SCRIPT_DIR.parents[1]
    / "ckpt"
    / "ERNIE-RNA"
    / "layer_12_token"
    / "relu_12x_dim768_20260602_013912"
    / "best.pt"
)
DEFAULT_OUTPUT_CSV = (
    DATA_DIR
    / "RNA"
    / "RNASSTR"
    / "results"
    / "ERNIERNA_mask_bracket_results_1_pred.csv"
)


def prediction_column_name(feature: str, kind: str) -> str:
    """kind: 'combo' | 'single' → pred_{kind}_mask_..."""
    if feature.startswith("mask_"):
        return f"pred_{kind}_{feature}"
    return f"pred_{kind}_mask_{feature}"


def binary_array_to_mask_string(binary: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in binary.astype(bool))


def id_column_name(df: pd.DataFrame) -> str:
    if "rfam_name" in df.columns:
        return "rfam_name"
    if "uniprot_id" in df.columns:
        return "uniprot_id"
    return "sequence_id"


@torch.no_grad()
def encode_and_predict_by_sequence(
    model: torch.nn.Module,
    embedding_npy: str,
    df: pd.DataFrame,
    seq_lengths: list[int],
    combo_cfg: dict,
    single_cfg: dict | None,
    device: str,
    max_batch_tokens: int,
) -> list[dict[str, Any]]:
    """
    Batch-encode embeddings by max_batch_tokens, then emit per-sequence
    combo / single 0-1 masks (no F1).
    """
    row_batches = build_sequence_batches(seq_lengths, max_batch_tokens)
    token_offsets = np.cumsum([0] + seq_lengths)
    all_embeddings = np.load(embedding_npy, mmap_mode="r")

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
        f"{len(row_batches)} SAE forward pass(es)"
    )
    print(f"Max batch tokens: {max_batch_tokens}")
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
    id_col = id_column_name(df)

    for batch_row_indices in tqdm(
        row_batches,
        desc="Encode embedding + predict",
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
                    f"sequence length={seq_len}."
                )

            sequence = str(df.iloc[local_row_idx]["sequence"])
            if len(sequence) != seq_len:
                raise ValueError(
                    f"Row {local_row_idx}: sequence len={len(sequence)} != {seq_len}"
                )

            sequence_id = row_sequence_id(df, local_row_idx)

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
                    id_col: sequence_id,
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

    df = load_evaluation_dataframe(args.csv_path)
    validate_csv_embedding_pairing(df, args.embedding_npy)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Predicting sequences: {len(seq_lengths)}, "
        f"total tokens: {sum(seq_lengths)}"
    )

    device = torch.device(args.device)
    sae_checkpoint = args.sae_checkpoint or config.get(
        "sae_checkpoint", str(DEFAULT_SAE_CHECKPOINT)
    )
    sae_type = args.sae_type or config.get("sae_type", "relu")
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

    max_batch_tokens = int(
        args.max_batch_tokens or config.get("max_batch_tokens", 128000)
    )

    pred_rows = encode_and_predict_by_sequence(
        model=model,
        embedding_npy=args.embedding_npy,
        df=df,
        seq_lengths=seq_lengths,
        combo_cfg=combo_cfg,
        single_cfg=single_cfg,
        device=str(device),
        max_batch_tokens=max_batch_tokens,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    id_col = id_column_name(df)
    combo_col = prediction_column_name(args.feature, "combo")
    single_col = prediction_column_name(args.feature, "single")
    out_df = pd.DataFrame(pred_rows)[
        [id_col, "sequence", "length", "combo_mask", "single_mask"]
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
    print(f"Predicted sequences: {len(out_df)}")
    print(f"Saved CSV: {output_csv}")

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate combo + single 0/1 prediction masks from ERNIERNA train "
            "results JSON (precomputed embedding.npy + SAE). "
            "Missing best_single_neuron → NaN."
        )
    )
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
    )
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--feature",
        type=str,
        default="mask_bracket",
        help=(
            "Feature key in train JSON; output columns become "
            "pred_combo_<feature> and pred_single_<feature>."
        ),
    )
    parser.add_argument(
        "--sae_checkpoint",
        type=str,
        default=None,
        help="Override SAE checkpoint; defaults to value in results JSON config.",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        default=None,
        help="Override SAE type; defaults to value in results JSON config.",
    )
    parser.add_argument("--max_batch_tokens", type=int, default=128000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_prediction(args)


if __name__ == "__main__":
    main()

# rfam_name,sequence,length,
# mask_angle,
# pred_combo_k1_mask_angle,
# pred_combo_k2_mask_angle,
# pred_combo_k3_mask_angle,
# pred_combo_k4_mask_angle,
# pred_single_mask_angle,
# combo_max_macro_f1,single_macro_f1,delta_macro_f1
# RF04337,
# AAGAAcUcAacGGGacCuucCcccucgAAUCAGACAAAgagggGgaaGguCCCguUgAgUUCUU,64,
# 1111111111111111111111111100000000000011111111111111111111111111,
# 1101111111101111000011110000000000000001011101101101101101011100,
# 1111010101111111000011100000000000000011111111111100111011111100,
# 1111110111111111000000100000000000000011111111111100111011111000,
# 1111110111111111100111110100000000000011111111111111111011100100,
# 0000010100100001100111110100000000000100000000000111100000000100,
# 0.816267942583732,0.41832473593711617,0.39794320664661587

# RF00671,
# uGcugCAGCUGGUGUuGUGAAUCAgGCCGauGagaagcgccuccuauaacaCGGCuAuUUCACaACACCAGGGcugCa,78,
# 111011001111111111111000111110000000000000000000000111110011111111111110110111,
# 001001001001101001000000011001000000000000000000000001100000001101010010010010,
# 011011001011111011100000011010000000000000000000000011100000001001010010110110,
# 011011001011110011111001011010000000000000000000000011100000011111110110110111,
# 011001101011110011111001011110000000000000000000000111100000111111111110110111,
# 001001111000000000000000001100000000000000000111000100100000101001011001010010,
# 0.8578293289146645,0.501010101010101,0.35681922790456344

# RF00702,
# cgccguuUUUGGCAAUGGUAGAACUCACaCuGGUGAGgUAauugGAUCCgGuGGUUCUAGACUUGCCAACUacggcu,77,
# 01111100111111100011111100011111100000000000000111111111111000111111100111110,
# 00110100100110000000010000010100000000000000000000110000000000000110000010110,
# 01111101101110010000110000000101100000000000000000111100000000001110000011111,
# 01111101101111100001111000001101100000000000000000101100001000001111100111111,
# 01111101001111100001111100011101100000000000000110101100101100001111100111111,
# 00110000000010000100000101010110000000000000000110000000100110000110000010010,
# 0.8143939393939394,0.4962121212121212,0.31818181818181823

# RNASSTR
# id,sequence,length,mask_bracket,pred_combo_k1_mask_bracket,pred_combo_k2_mask_bracket,pred_combo_k3_mask_bracket,pred_combo_k4_mask_bracket,pred_single_mask_bracket,combo_max_macro_f1,single_macro_f1,delta_macro_f1
# 1989|CALYPG010000003.1-161843-162269_RF01766,
# GACAUUUAACGCUGUAUAUUUUUAUGAAAACGGUUAACUUUAUAAUCCAGUUUUUACUCAUUAUAGUUGUUUAUUUGUUUGCAUAUUAAUAUCCGUGUUAAUCGUCUAAGCGAGCCUAUAAAGGCAAAUAUUAAUUAUUGUUAGGAUGUCUAAUAAAUGAAUAAGAAAACUGGCCAAGUAAAAUGGUUUAACGAAAGUAAAGGUUUUGGGUUUAUUACUCCAACUGAUGGAAGCAAAGAUGUAUUUGUUCACUUUUCAGCAAUUUCUGGGGAUGGAUUCAAAACCCUAGCUGAAGGGCAACAAGUCGAAUUUGCUAUCCAAGAUAGCCCUCGUGGACCAGCAGCGGCUGAAGUUGUCGUUGUUUAAUUAUCUAAUUAAAUAAUUCAUUUUAAGCAAAACGGUUCGUUUAUAUAAAUGAACCGUUUUU,427,
# 0000000001101111000000000000110011111111000001010111110111000111111000000010000000111100000000000111100000000000011100000011100000000000100000111111011111000000000000001111100000000000111111111000010111100001111110011110110011111100000001100000110011111101111110111111001110000000011111111100001101111111111000000111111000011101110001111111111111100000001111111100111000001111000011110111000000001111111111110000111111111111000,
# 0000000000000000000000000010010000000000000000000000000000000000000000000000000000000000000000000000000000000000011100000011100000000000000000000100000000101111111100001111111111111111111111111111111111111001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000000001111111111000000011111111111000,
# 0000000000000000001000000010110100000000000000000000000000000000000000000000000000000000000000000000000000000000011100000011100000000000000000000100000000000001000101001101111111111111111111111111111111111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000000000000000000000000001111111111000000011111111111000,
# 0000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000011100000011100000000000000000000000000000000001000100001001111111111111111111111111111111111001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000000001111111111100000011111111111000,
# 0000000000000000001000000010110100000000000000000000000000000000010000000000000001010000000000000000000000000000011100000011100000000000000000000000000000000001000000000001110001100111111111111111110111111101111111111111111111111111110110111111111111111111111111111111110101110100001111110111111111101111111111111111111101101111111111111110111111110101111111111111111000000000000000000000000000001111111111100000011111111111000,
# 1000000000100100000100100100000110000000100000000100000000000000011110010111111010000000000000101000000100000101010000000011000000000000000100011001000000000010000010000001101001000000110000001000100001100001110000000000000001001100100001001000001000000000001000000001111001100000000000001001001110000001001000001000000001000100000101100001001011001001001001001000000000000000000000000000000010000001100010000000000010000100000,
# 0.6674308907415534,0.47723132969034604,0.19019956105120733