#!/usr/bin/env python3
"""
Train neuron boosting from precomputed Evo2 layer-26 embeddings (.npy).

Mirrors ESM2_train_neuron_boosting.py: load embeddings + SAE encode into f, then
delegate boosting to neuron_boosting (in-memory, not streaming).

Continuous labels (fasta + atac npy): skips single-neuron F1 search; use
composite_r2. JSON schema matches the ESM2 boosting output.

Embedding npy shape: [num_tokens, 4096] (token order = fasta order).

Example:
    # 1) export embeddings (optional helper)
    CUDA_VISIBLE_DEVICES=2 python Evo2_export_embeddings.py

    # 2) train from npy
    CUDA_VISIBLE_DEVICES=2 python Evo2_train_neuron_boosting.py \\
        --embedding_npy data/test_sample_3000_a/evo2_7b_262k_layer26_embedding.npy \\
        --num_iterations 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from neuron_boosting import (  # noqa: E402
    THRESHOLD_GRID,
    ActivationData,
    FeatureEvalParams,
    FeatureEvaluationResult,
    compute_activation_data,
    run_feature_evaluation,
    validate_csv_embedding_pairing,
)
from evo2_sae_utils import (  # noqa: E402
    DEFAULT_SAE_CKPT,
    load_fasta_continuous_labels,
    load_topk_sae,
)

DEFAULT_FASTA = SCRIPT_DIR / "data" / "test_sample_3000_a" / "sequences.fasta"
DEFAULT_LABEL_NPY = SCRIPT_DIR / "data" / "test_sample_3000_a" / "atac_K562.npy"
DEFAULT_EMBEDDING_NPY = (
    SCRIPT_DIR / "data" / "test_sample_3000_a" / "evo2_7b_262k_layer26_embedding.npy"
)
DEFAULT_OUTPUT_JSON = (
    SCRIPT_DIR
    / "data"
    / "test_sample_3000_a"
    / "results"
    / "Evo2_7b_262k_layer26_atac_K562_neuron_parameters_from_embedding.json"
)


@dataclass
class EvaluationConfig:
    fasta_path: str = str(DEFAULT_FASTA)
    label_npy: str = str(DEFAULT_LABEL_NPY)
    embedding_npy: str = str(DEFAULT_EMBEDDING_NPY)
    sae_ckpt: str = DEFAULT_SAE_CKPT
    feature_name: str = "atac_K562"
    max_batch_tokens: int = 128000
    device: str = "cuda:0"
    num_iterations: int = 10
    fit_constant: bool = True
    output_json: str = str(DEFAULT_OUTPUT_JSON)
    max_sequences: int | None = None


def load_activation_data(config: EvaluationConfig) -> ActivationData:
    """Load fasta/labels, validate embedding pairing, SAE-encode to ActivationData."""
    import torch

    df, y = load_fasta_continuous_labels(
        config.fasta_path,
        config.label_npy,
        feature_name=config.feature_name,
    )
    if config.max_sequences is not None:
        n = int(config.max_sequences)
        df = df.iloc[:n].reset_index(drop=True)
        total = int(df["sequence"].astype(str).str.len().sum())
        y = y[:total]
        print(f"Using first {n} sequences ({total} tokens)")

    embedding_num_tokens = validate_csv_embedding_pairing(df, config.embedding_npy)
    if y.shape[0] != embedding_num_tokens:
        raise ValueError(
            f"Label tokens {y.shape[0]} != embedding tokens {embedding_num_tokens}"
        )

    device = torch.device(config.device)
    sae = load_topk_sae(config.sae_ckpt, device=device, dtype=torch.bfloat16)
    print(
        f"Loaded SAE from {config.sae_ckpt}: "
        f"dict_size={sae.dict_size}, activation_dim={sae.activation_dim}"
    )
    print(f"SAE device: {device}, max_batch_tokens: {config.max_batch_tokens}")

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Total sequences: {len(seq_lengths)}, total tokens: {sum(seq_lengths)}, "
        f"label mean={float(y.mean()):.6g}, std={float(y.std()):.6g}"
    )

    data = compute_activation_data(
        model=sae,
        embedding_npy=config.embedding_npy,
        seq_lengths=seq_lengths,
        labels={config.feature_name: y},
        device=str(device),
        max_batch_tokens=config.max_batch_tokens,
        expected_num_tokens=embedding_num_tokens,
    )

    del sae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return data


def config_to_feature_eval_params(config: EvaluationConfig) -> FeatureEvalParams:
    return FeatureEvalParams(
        num_iterations=config.num_iterations,
        fit_constant=config.fit_constant,
        search_best_single_neuron=False,  # continuous labels
        output_json=config.output_json,
        save_iteration_checkpoints=True,
        config_dict=asdict(config),
        threshold_grid=THRESHOLD_GRID,
    )


def evaluate_neuron_feature_r2(
    config: EvaluationConfig,
) -> list[FeatureEvaluationResult]:
    data = load_activation_data(config)
    params = config_to_feature_eval_params(config)
    print(
        "Note: continuous-label mode. Prefer combination.composite_r2 / "
        "boosting_final_composite_r2; best_macro_f1 is schema-compatible only."
    )
    return run_feature_evaluation(
        data=data,
        params=params,
        config_dict=asdict(config),
    )


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evo2 Layer-26 SAE neuron boosting from precomputed embedding.npy "
            "(continuous labels, R²)."
        )
    )
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument("--label_npy", type=str, default=str(DEFAULT_LABEL_NPY))
    parser.add_argument("--embedding_npy", type=str, default=str(DEFAULT_EMBEDDING_NPY))
    parser.add_argument("--sae_ckpt", type=str, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--feature_name", type=str, default="atac_K562")
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=128000,
        help="Max tokens per SAE encode batch when building f.",
    )
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--no_fit_constant", action="store_true")
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    return EvaluationConfig(
        fasta_path=args.fasta_path,
        label_npy=args.label_npy,
        embedding_npy=args.embedding_npy,
        sae_ckpt=args.sae_ckpt,
        feature_name=args.feature_name,
        max_batch_tokens=args.max_batch_tokens,
        device=args.device,
        num_iterations=args.num_iterations,
        fit_constant=not args.no_fit_constant,
        output_json=args.output_json,
        max_sequences=args.max_sequences,
    )


def main() -> None:
    config = parse_args()
    if not Path(config.embedding_npy).is_file():
        raise FileNotFoundError(
            f"Embedding npy not found: {config.embedding_npy}\n"
            "Export first, e.g.:\n"
            "  CUDA_VISIBLE_DEVICES=2 python Evo2_export_embeddings.py"
        )
    evaluate_neuron_feature_r2(config)


if __name__ == "__main__":
    main()
