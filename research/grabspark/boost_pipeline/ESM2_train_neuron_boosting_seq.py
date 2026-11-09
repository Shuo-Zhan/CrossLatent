#!/usr/bin/env python3
"""
Streaming SAE neuron boosting: on-the-fly ESM2 embedding + SAE encode per batch.

No precomputed embedding.npy or full activation matrix in memory. Delegates to
neuron_boosting_seq for boosting, closed-form refit, single-neuron search, and
threshold evaluation.

Example:
    python ESM2_train_neuron_boosting_seq.py \\
        --num_iterations 10 \\
        --output_json data/Protein/uniprot_sprot_2500_2500/results/ESM2_650M_boosting_neuron_feature_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from interplm.embedders import get_embedder  # pyright: ignore[reportMissingImports]
from interplm.sae.inference import load_sae_from_hf  # pyright: ignore[reportMissingImports]
from neuron_boosting import THRESHOLD_GRID, validate_row_labels, load_evaluation_dataframe
from neuron_boosting_seq import (  # pyright: ignore[reportMissingImports]
    StreamingFeatureEvalParams,
    build_streaming_context,
    compute_all_labels_from_df,
    run_streaming_feature_evaluation,
)

DATA_DIR = SCRIPT_DIR.parents[1] / "data"
# DEFAULT_CSV = DATA_DIR / "Protein" / "clan_CL0036" / "CL0036_train.csv"
DEFAULT_CSV = DATA_DIR / "Protein" / "clan_CL0063" / "CL0063_train.csv"
DEFAULT_ESM2_MODEL_DIR = INTERPLM_ROOT / "esm2_model" / "esm2_650M_UR50D"
DEFAULT_OUTPUT_JSON = (
    DATA_DIR
    / "Protein"
    / "clan_CL0063"
    / "results"
    / "ESM2_650M_33_mask_regions_of_interest_neuron_parameters_0.9clan0063.json"
)


@dataclass
class EvaluationConfig:
    csv_path: str = str(DEFAULT_CSV)
    esm_model_dir: str = str(DEFAULT_ESM2_MODEL_DIR)
    plm_model: str = "esm2-650m"
    plm_layer: int = 33
    label_column: str = "mask_alpha_helix"
    max_batch_tokens: int = 128000
    embed_batch_size: int = 24
    device: str = "cuda:0"
    num_iterations: int = 10
    fit_constant: bool = True
    search_best_single_neuron: bool = True
    output_json: str = str(DEFAULT_OUTPUT_JSON)
    closed_form_full_pass_token_threshold: int = 100_000
    resume_from: str | None = None


def load_models(config: EvaluationConfig) -> tuple[Any, torch.nn.Module, torch.device]:
    device = torch.device(config.device)
    embedder = get_embedder(
        "esm",
        model_name=config.esm_model_dir,
        device=str(device),
    )
    sae = load_sae_from_hf(
        plm_model=config.plm_model,
        plm_layer=config.plm_layer,
    ).to(device)
    sae.eval()
    print(f"Loaded ESM2 embedder from: {config.esm_model_dir}")
    print(
        f"Loaded SAE: plm_model={config.plm_model}, plm_layer={config.plm_layer}, "
        f"dict_size={sae.dict_size}, activation_dim={sae.activation_dim}"
    )
    print(f"Device: {device}")
    print(f"Max batch tokens: {config.max_batch_tokens}, embed batch size: {config.embed_batch_size}")
    return embedder, sae, device


def build_context(config: EvaluationConfig):
    df = load_evaluation_dataframe(config.csv_path)
    validate_row_labels(df, config.label_column)

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(f"Total sequences: {len(seq_lengths)}, total tokens: {sum(seq_lengths)}")

    labels = compute_all_labels_from_df(df, [config.label_column])
    y = labels[config.label_column]

    embedder, sae, device = load_models(config)
    ctx = build_streaming_context(
        df=df,
        y=y,
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=config.plm_layer,
        max_batch_tokens=config.max_batch_tokens,
        embed_batch_size=config.embed_batch_size,
    )
    return ctx, config.label_column


def config_to_feature_eval_params(config: EvaluationConfig) -> StreamingFeatureEvalParams:
    return StreamingFeatureEvalParams(
        num_iterations=config.num_iterations,
        fit_constant=config.fit_constant,
        search_best_single_neuron=config.search_best_single_neuron,
        output_json=config.output_json,
        save_iteration_checkpoints=True,
        config_dict=asdict(config),
        threshold_grid=THRESHOLD_GRID,
        closed_form_full_pass_token_threshold=config.closed_form_full_pass_token_threshold,
        resume_from=config.resume_from,
    )


def evaluate_neuron_feature_r2(config: EvaluationConfig):
    ctx, feature = build_context(config)
    params = config_to_feature_eval_params(config)
    return run_streaming_feature_evaluation(
        ctx=ctx,
        feature=feature,
        params=params,
        config_dict=asdict(config),
    )


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming boosting-style SAE neuron selection (R²), "
            "then threshold sweep for macro F1 (no embedding.npy)."
        )
    )
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--esm_model_dir", type=str, default=str(DEFAULT_ESM2_MODEL_DIR))
    parser.add_argument("--plm_model", type=str, default="esm2-650m")
    parser.add_argument("--plm_layer", type=int, default=33)
    parser.add_argument(
        "--label_column",
        type=str,
        default="mask_regions_of_interest",
        help="Label column name for a single feature.",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=64000,
        help=(
            "Maximum total tokens per streaming encode batch. "
            "Lower this if ESM2 OOMs (attention scales with sequence length)."
        ),
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=16,
        help=(
            "ESM2 mini-batch size inside extract_embeddings. "
            "Reduce for long proteins (default 16)."
        ),
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
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
        help="Skip best single-neuron threshold search.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--closed_form_full_pass_token_threshold",
        type=int,
        default=500_000,
        help=(
            "When total tokens <= this value, closed-form refit accumulates over all "
            "streaming batches; otherwise uses a single batch (fits y_boost, not labels)."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument(
        "--resume_from",
        type=str,
        default='/data1/zshuo23/GraBSpArk/data/Protein/clan_CL0063/results/ESM2_650M_33_mask_regions_of_interest_neuron_parameters_0.9clan0063_9.json',
        help=(
            "Path to an iteration checkpoint JSON (e.g. output_stem_9.json) to resume "
            "greedy boosting from the last saved step."
        ),
    )

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    df_preview = load_evaluation_dataframe(args.csv_path)
    if args.label_column not in df_preview.columns:
        raise ValueError(f"Label column not found in CSV: {args.label_column}")

    return EvaluationConfig(
        csv_path=args.csv_path,
        esm_model_dir=args.esm_model_dir,
        plm_model=args.plm_model,
        plm_layer=args.plm_layer,
        label_column=args.label_column,
        max_batch_tokens=args.max_batch_tokens,
        embed_batch_size=args.embed_batch_size,
        device=args.device,
        num_iterations=args.num_iterations,
        fit_constant=not args.no_fit_constant,
        search_best_single_neuron=not args.no_search_best_single_neuron,
        output_json=args.output_json,
        closed_form_full_pass_token_threshold=args.closed_form_full_pass_token_threshold,
        resume_from=args.resume_from,
    )


def main() -> None:
    config = parse_args()
    evaluate_neuron_feature_r2(config)


if __name__ == "__main__":
    main()

# nohup python ESM2_train_neuron_boosting_seq.py > output.log 2>&1 &
# 1512781
# 2:34  2:51  3:08  3:27
# mask_regions_of_interest CL0063_train  2044320  4.1933%  0.3477%
# mask_disulfide_bonds     CL0036_train  2046650  3.1567%  18.1369%