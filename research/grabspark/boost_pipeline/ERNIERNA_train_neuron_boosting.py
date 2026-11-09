#!/usr/bin/env python3
"""
Evaluate ERNIE-RNA SAE neuron activation against Rfam structural labels via boosting R².

This entry script loads Rfam.csv labels, ERNIE-RNA layer-12 embeddings (Rfam.npy),
and a locally trained SAE checkpoint (best.pt), then delegates batch encoding,
single-neuron search, boosting, and threshold evaluation to neuron_boosting.
Each iteration checkpoint JSON includes both methods.

Example:
    python ERNIERNA_train_neuron_boosting.py \\
        --num_iterations 10 \\
        --label_column mask_angle \\
        --output_json data/RNA/Rfam/results/ERNIERNA_boosting_neuron_feature_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_TRAIN_DIR = SCRIPT_DIR.parent / "sae_train"
DATA_DIR = SCRIPT_DIR.parents[1] / "data"

sys.path.insert(0, str(SAE_TRAIN_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from sae_models import load_sae_from_checkpoint  # pyright: ignore[reportMissingImports]
from neuron_boosting import (
    THRESHOLD_GRID,
    ActivationData,
    FeatureEvalParams,
    FeatureEvaluationResult,
    build_results_json as _build_results_json,
    compute_activation_data,
    compute_all_labels,
    evaluate_feature as _evaluate_feature,
    run_feature_evaluation,
    normalize_mask_columns,
    validate_csv_embedding_pairing,
    validate_row_labels,
)


def load_evaluation_dataframe(csv_path: str):
    """
    Rfam-safe CSV loader: force string dtypes so all-zero masks (e.g. mask_dot)
    are not inferred as int 0. Shared neuron_boosting.load_evaluation_dataframe
    is left unchanged so other pipelines / old result JSONs are unaffected.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "sequence" not in df.columns:
        raise ValueError("CSV must contain a 'sequence' column.")
    return normalize_mask_columns(df)


RFAM_DIR = DATA_DIR / "RNA" / "Rfam" / "ERNIERNA_embeddings"
DEFAULT_CSV = RFAM_DIR / "Rfam_train.csv"
DEFAULT_EMBEDDING_NPY = RFAM_DIR / "layer_12" / "Rfam_train.npy"
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
DEFAULT_OUTPUT_JSON = (
    DATA_DIR / "RNA" / "Rfam" / "results" / "mask_angle_8" / "ERNIERNA_mask_angle_results.json"
)


@dataclass
class EvaluationConfig:
    csv_path: str = str(DEFAULT_CSV)
    embedding_npy: str = str(DEFAULT_EMBEDDING_NPY)
    sae_checkpoint: str = str(DEFAULT_SAE_CHECKPOINT)
    sae_type: Optional[str] = "relu"
    label_column: str = "mask_angle"
    max_batch_tokens: int = 128000
    device: str = "cuda:7"
    num_iterations: int = 5
    fit_constant: bool = True
    search_best_single_neuron: bool = True
    output_json: str = str(DEFAULT_OUTPUT_JSON)


def load_sae_model(config: EvaluationConfig) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device(config.device)
    model = load_sae_from_checkpoint(
        config.sae_checkpoint,
        sae_type=config.sae_type,
        device=device,
    )
    model.eval()
    print(
        f"Loaded SAE: checkpoint={config.sae_checkpoint}, "
        f"sae_type={config.sae_type}, "
        f"dict_size={model.dict_size}, activation_dim={model.activation_dim}"
    )
    print(f"SAE device: {device}")
    print(f"Max batch tokens: {config.max_batch_tokens}")
    return model, device


def load_activation_data(config: EvaluationConfig) -> ActivationData:
    """Load CSV/labels, validate pairing, load SAE, and encode activations."""
    df = load_evaluation_dataframe(config.csv_path)
    embedding_num_tokens = validate_csv_embedding_pairing(df, config.embedding_npy)
    validate_row_labels(df, config.label_column)

    model, device = load_sae_model(config)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(f"Total sequences: {len(seq_lengths)}, total tokens: {sum(seq_lengths)}")

    labels = compute_all_labels(df, [config.label_column])
    data = compute_activation_data(
        model=model,
        embedding_npy=config.embedding_npy,
        seq_lengths=seq_lengths,
        labels=labels,
        device=str(device),
        max_batch_tokens=config.max_batch_tokens,
        expected_num_tokens=embedding_num_tokens,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return data


def config_to_feature_eval_params(config: EvaluationConfig) -> FeatureEvalParams:
    return FeatureEvalParams(
        num_iterations=config.num_iterations,
        fit_constant=config.fit_constant,
        search_best_single_neuron=config.search_best_single_neuron,
        output_json=config.output_json,
        save_iteration_checkpoints=True,
        config_dict=asdict(config),
        threshold_grid=THRESHOLD_GRID,
    )


def evaluate_feature(
    f,
    y,
    feature: str,
    seq_lengths: list[int],
    config: EvaluationConfig,
) -> FeatureEvaluationResult:
    """Backward-compatible wrapper around neuron_boosting.evaluate_feature."""
    return _evaluate_feature(
        f=f,
        y=y,
        feature=feature,
        seq_lengths=seq_lengths,
        params=config_to_feature_eval_params(config),
    )


def build_results_json(
    config: EvaluationConfig,
    results: list[FeatureEvaluationResult],
    f_shape: tuple[int, ...],
) -> dict:
    """Backward-compatible wrapper around neuron_boosting.build_results_json."""
    return _build_results_json(
        config_dict=asdict(config),
        results=results,
        f_shape=f_shape,
        threshold_grid=THRESHOLD_GRID,
    )


def evaluate_neuron_feature_r2(config: EvaluationConfig) -> list[FeatureEvaluationResult]:
    data = load_activation_data(config)
    params = config_to_feature_eval_params(config)
    return run_feature_evaluation(
        data=data,
        params=params,
        config_dict=asdict(config),
    )


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Boosting-style SAE neuron selection (R²) on Rfam, "
            "then threshold sweep for macro F1."
        )
    )
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--embedding_npy", type=str, default=str(DEFAULT_EMBEDDING_NPY))
    parser.add_argument(
        "--sae_checkpoint",
        type=str,
        default=str(DEFAULT_SAE_CHECKPOINT),
        help="Path to trained SAE checkpoint (e.g. best.pt).",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        default="relu",
        help="SAE architecture type passed to load_sae_from_checkpoint.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="mask_angle",
        help=(
            "Label column name for a single feature "
            "(e.g. mask_angle,mask_bracket,mask_dash,mask_comma,mask_colon,mask_dot)."
        ),
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=128000,
        help="Maximum total tokens per SAE forward pass.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of SAE neurons to select after constant step.",
    )
    parser.add_argument(
        "--no_fit_constant",
        action="store_true",
        help="Skip the preliminary constant feature step (y = a*1).",
    )
    parser.add_argument(
        "--no_search_best_single_neuron",
        action="store_true",
        help="Skip best single-neuron threshold search before boosting.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save JSON results.",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    df_preview = load_evaluation_dataframe(args.csv_path)
    if args.label_column not in df_preview.columns:
        raise ValueError(f"Label column not found in CSV: {args.label_column}")

    return EvaluationConfig(
        csv_path=args.csv_path,
        embedding_npy=args.embedding_npy,
        sae_checkpoint=args.sae_checkpoint,
        sae_type=args.sae_type,
        label_column=args.label_column,
        max_batch_tokens=args.max_batch_tokens,
        device=args.device,
        num_iterations=args.num_iterations,
        fit_constant=not args.no_fit_constant,
        search_best_single_neuron=not args.no_search_best_single_neuron,
        output_json=args.output_json,
    )


def main() -> None:
    config = parse_args()
    evaluate_neuron_feature_r2(config)


if __name__ == "__main__":
    main()

# nohup python ERNIERNA_train_neuron_boosting.py > run.log 2>&1 &
# 1089145  1
