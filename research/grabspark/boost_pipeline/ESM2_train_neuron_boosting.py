#!/usr/bin/env python3
"""
Evaluate InterPLM SAE neuron activation against protein token labels via boosting R².

This entry script loads CSV labels, ESM2 embeddings, and the SAE model, then
delegates batch encoding, single-neuron search, boosting, and threshold evaluation
to neuron_boosting. Each iteration checkpoint JSON includes both methods.

Example:
    python ESM2_train_neuron_boosting.py \\
        --num_iterations 10 \\
        --output_json data/Protein/uniprot_sprot_2500_2500/esm2_boosting_neuron_feature_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
INTERPLM_ROOT = SCRIPT_DIR.parents[1] / "InterPLM"

sys.path.insert(0, str(INTERPLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from interplm.sae.inference import load_sae_from_hf  # pyright: ignore[reportMissingImports]
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
    load_evaluation_dataframe,
    validate_csv_embedding_pairing,
    validate_row_labels,
)

DATA_DIR = SCRIPT_DIR.parents[1] / "data"
# DEFAULT_CSV = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "alpha_helix_labels_train_2500.csv"
# DEFAULT_EMBEDDING_NPY = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "ESM2_650M_embedding_train_2500.npy"
# DEFAULT_OUTPUT_JSON = DATA_DIR / "Protein" / "uniprot_sprot_2500_2500" / "results" / "ESM2_650M_boosting_neuron_feature_results.json"
# DEFAULT_CSV = DATA_DIR / "Protein" / "clan_CL0023" / "CL0023_group5.csv"
# DEFAULT_EMBEDDING_NPY = DATA_DIR / "Protein" / "clan_CL0023" / "CL0023_group5_ESM2_650M_embedding.npy"
# DEFAULT_OUTPUT_JSON = DATA_DIR / "Protein" / "clan_CL0023" / "results" / "ESM2_650M_33_mask_protein_domains_neuron_parameters_group5.json"
DEFAULT_CSV = DATA_DIR / "Protein" / "High-quality-dataset" / "mask_turns_CL0063_0.0259.csv"
DEFAULT_EMBEDDING_NPY = DATA_DIR / "Protein" / "High-quality-dataset" / "mask_turns_CL0063_0.0259_ESM2_650M_embedding.npy"
DEFAULT_OUTPUT_JSON = DATA_DIR / "Protein" / "High-quality-dataset" / "results" / "mask_turns" / "ESM2_650M_33_mask_turns_CL0063_0.0259_neuron_parameters.json"

# mask_turns_CL0063_0.0259

@dataclass
class EvaluationConfig:
    csv_path: str = str(DEFAULT_CSV)
    embedding_npy: str = str(DEFAULT_EMBEDDING_NPY)
    plm_model: str = "esm2-650m"
    plm_layer: int = 33
    label_column: str = "mask_alpha_helix"
    max_batch_tokens: int = 128000
    device: str = "cuda:0"
    num_iterations: int = 5
    fit_constant: bool = True
    search_best_single_neuron: bool = True
    output_json: str = str(DEFAULT_OUTPUT_JSON)


def load_sae_model(config: EvaluationConfig) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device(config.device)
    model = load_sae_from_hf(
        plm_model=config.plm_model,
        plm_layer=config.plm_layer,
    ).to(device)
    model.eval()
    print(
        f"Loaded SAE: plm_model={config.plm_model}, plm_layer={config.plm_layer}, "
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
            "Boosting-style SAE neuron selection (R²), then threshold sweep for macro F1."
        )
    )
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--embedding_npy", type=str, default=str(DEFAULT_EMBEDDING_NPY))
    parser.add_argument("--plm_model", type=str, default="esm2-650m")
    parser.add_argument("--plm_layer", type=int, default=33)
    parser.add_argument(
        "--label_column",
        type=str,
        default="mask_turns",
        help="Label column name for a single feature.",
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
        help="Skip best single-neuron threshold search after combination evaluation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save JSON results.",
    )
    parser.add_argument("--device", type=str, default="cuda:7")
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
        plm_model=args.plm_model,
        plm_layer=args.plm_layer,
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

# nohup python ESM2_train_neuron_boosting.py > run.log 2>&1 &
# 325702  1
# 326598  2
# 330923  3
# 331734  4
# 333838  5
# 338362  6
# 339037  7
# 14:21

# 2506900
# 2508278
# mkdir -p /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting && nohup env CUDA_VISIBLE_DEVICES=0 /data1/zshuo23/anaconda3/envs/hfds/bin/python -u /data1/zshuo23/GraBSpArk/grabspark/boost_pipeline/ESM2_train_neuron_boosting.py --csv_path /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33/PF00005_train.csv --embedding_npy /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33/PF00005_train_ESM2_650M_layer33.npy --label_column mask_target_domain --plm_model esm2-650m --plm_layer 33 --num_iterations 20 --max_batch_tokens 64000 --device cuda:0 --output_json /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting/PF00005_mask_target_domain_latent20.json > /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting/PF00005_mask_target_domain_latent20.log 2>&1 & echo "PF00005 PID=$!"
# mkdir -p /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting && nohup env CUDA_VISIBLE_DEVICES=1 /data1/zshuo23/anaconda3/envs/hfds/bin/python -u /data1/zshuo23/GraBSpArk/grabspark/boost_pipeline/ESM2_train_neuron_boosting.py --csv_path /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33/PF00004_train.csv --embedding_npy /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/embeddings/ESM2_650M_layer33/PF00004_train_ESM2_650M_layer33.npy --label_column mask_target_domain --plm_model esm2-650m --plm_layer 33 --num_iterations 20 --max_batch_tokens 64000 --device cuda:0 --output_json /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting/PF00004_mask_target_domain_latent20.json > /data1/zshuo23/GraBSpArk/data/Protein/fig5_pfam00005_pfam00004_labels/full_alignment_sample/results/domain_boosting/PF00004_mask_target_domain_latent20.log 2>&1 & echo "PF00004 PID=$!"

