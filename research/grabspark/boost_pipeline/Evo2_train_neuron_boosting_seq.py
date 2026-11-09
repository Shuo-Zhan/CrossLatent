#!/usr/bin/env python3
"""
Streaming neuron boosting for Evo2 Layer-26 SAE on continuous labels.

Reads precomputed Evo2 embeddings from a memmap .npy in token chunks (no full
activation matrix f in memory, no online Evo2 forward). SAE-encodes each chunk
and runs the existing neuron_boosting_seq streaming boosting pipeline.

Primary metric: composite R². JSON schema matches ESM2 boosting output.

Example:
    CUDA_VISIBLE_DEVICES=2 python Evo2_train_neuron_boosting_seq.py \\
        --embedding_npy data/test_sample_3000_a/evo2_7b_262k_layer26_embedding.npy \\
        --num_iterations 10 \\
        --max_batch_tokens 128000
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import neuron_boosting_seq as nbs
from neuron_boosting import THRESHOLD_GRID
from neuron_boosting_seq import (
    StreamingFeatureEvalParams,
    build_streaming_context,
    run_streaming_feature_evaluation,
)
from evo2_sae_utils import (
    DEFAULT_SAE_CKPT,
    D_HIDDEN,
    MemmapEmbeddingEmbedder,
    load_fasta_continuous_labels,
    load_topk_sae,
)

DEFAULT_FASTA = SCRIPT_DIR / "data" / "test_sample_3000_a" / "sequences.fasta"
DEFAULT_LABEL_NPY = SCRIPT_DIR / "data" / "test_sample_3000_a" / "phyloP.npy"
DEFAULT_EMBEDDING_NPY = (
    SCRIPT_DIR / "data" / "test_sample_3000_a" / "evo2_7b_262k_layer26_embedding.npy"
)
DEFAULT_OUTPUT_JSON = (
    SCRIPT_DIR
    / "data"
    / "test_sample_3000_a"
    / "results"
    / "Evo2_7b_262k_layer26_phyloP_neuron_parameters.json"
)


@dataclass
class EvaluationConfig:
    fasta_path: str = str(DEFAULT_FASTA)
    label_npy: str = str(DEFAULT_LABEL_NPY)
    embedding_npy: str = str(DEFAULT_EMBEDDING_NPY)
    sae_ckpt: str = DEFAULT_SAE_CKPT
    feature_name: str = "phyloP"
    max_batch_tokens: int = 128000
    embed_batch_size: int = 1  # unused for memmap path; kept for config compatibility
    device: str = "cuda:0"
    num_iterations: int = 10
    fit_constant: bool = True
    output_json: str = str(DEFAULT_OUTPUT_JSON)
    closed_form_full_pass_token_threshold: int = 5_000_000
    resume_from: str | None = None
    max_sequences: int | None = None


def _install_memmap_encode_patches() -> None:
    """
    Patch neuron_boosting_seq encode helpers so MemmapEmbeddingEmbedder gets
    the current row-batch indices before slicing the .npy.
    """

    def iter_f_y_batches(
        ctx,
        *,
        desc: str = "Streaming encode",
        leave: bool = True,
    ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        for batch_row_indices in tqdm(ctx.row_batches, desc=desc, leave=leave):
            batch_start = int(ctx.token_offsets[batch_row_indices[0]])
            batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
            batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
            y_batch = ctx.y[batch_start:batch_end]
            if hasattr(ctx.embedder, "set_batch_rows"):
                ctx.embedder.set_batch_rows(batch_row_indices)
            f_batch = nbs._encode_batch(ctx, batch_sequences)
            yield batch_start, batch_end, f_batch, y_batch

    def _encode_row_batch(ctx, batch_row_indices: list[int]):
        if hasattr(ctx.embedder, "set_batch_rows"):
            ctx.embedder.set_batch_rows(batch_row_indices)
        batch_start = int(ctx.token_offsets[batch_row_indices[0]])
        batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
        batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
        f_batch = nbs._encode_batch(ctx, batch_sequences)
        return batch_start, batch_end, f_batch

    nbs.iter_f_y_batches = iter_f_y_batches
    nbs._encode_row_batch = _encode_row_batch


def build_context(config: EvaluationConfig):
    if not Path(config.embedding_npy).is_file():
        raise FileNotFoundError(
            f"Embedding npy not found: {config.embedding_npy}\n"
            "Export first, e.g.:\n"
            "  CUDA_VISIBLE_DEVICES=2 python Evo2_export_embeddings.py"
        )

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
        print(f"Using first {n} sequences ({total} tokens) for a quick run")

    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    print(
        f"Total sequences: {len(seq_lengths)}, total tokens: {sum(seq_lengths)}, "
        f"label mean={float(y.mean()):.6g}, std={float(y.std()):.6g}"
    )

    device = torch.device(config.device)
    sae = load_topk_sae(config.sae_ckpt, device=device, dtype=torch.bfloat16)
    print(
        f"Loaded SAE from {config.sae_ckpt}: "
        f"dict_size={sae.dict_size}, activation_dim={sae.activation_dim}, device={device}"
    )

    # Temporary context to get token_offsets / row_batches, then attach memmap embedder.
    dummy_embedder = object()
    ctx = build_streaming_context(
        df=df,
        y=y,
        embedder=dummy_embedder,
        sae=sae,
        device=device,
        plm_layer=26,
        max_batch_tokens=config.max_batch_tokens,
        embed_batch_size=config.embed_batch_size,
    )
    embedder = MemmapEmbeddingEmbedder(
        config.embedding_npy,
        token_offsets=ctx.token_offsets,
        seq_lengths=ctx.seq_lengths,
        expected_hidden=D_HIDDEN,
    )
    if int(embedder._mm.shape[0]) < ctx.num_tokens:
        raise ValueError(
            f"Embedding has {embedder._mm.shape[0]} rows but need {ctx.num_tokens} tokens"
        )
    ctx.embedder = embedder

    print(
        f"Streaming from memmap embeddings with max_batch_tokens={config.max_batch_tokens} "
        f"({len(ctx.row_batches)} chunk(s))"
    )
    return ctx, config.feature_name


def config_to_feature_eval_params(config: EvaluationConfig) -> StreamingFeatureEvalParams:
    return StreamingFeatureEvalParams(
        num_iterations=config.num_iterations,
        fit_constant=config.fit_constant,
        search_best_single_neuron=False,
        output_json=config.output_json,
        save_iteration_checkpoints=True,
        config_dict=asdict(config),
        threshold_grid=THRESHOLD_GRID,
        closed_form_full_pass_token_threshold=config.closed_form_full_pass_token_threshold,
        resume_from=config.resume_from,
    )


def run_train(config: EvaluationConfig) -> list[Any]:
    _install_memmap_encode_patches()
    ctx, feature = build_context(config)
    params = config_to_feature_eval_params(config)
    print(
        "Note: continuous-label + memmap-embedding mode. "
        "Use combination.composite_r2 / boosting_final_composite_r2; "
        "best_macro_f1 fields are schema-compatible only."
    )
    return run_streaming_feature_evaluation(
        ctx=ctx,
        feature=feature,
        params=params,
        config_dict=asdict(config),
    )


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evo2 Layer-26 SAE streaming neuron boosting from memmap embedding.npy "
            "(continuous labels, R²)."
        )
    )
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument("--label_npy", type=str, default=str(DEFAULT_LABEL_NPY))
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help="Precomputed Evo2 layer-26 embeddings [N_tokens, 4096].",
    )
    parser.add_argument("--sae_ckpt", type=str, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--feature_name", type=str, default="phyloP")
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=102500,
        help="Tokens per streaming chunk when reading embeddings + SAE encode.",
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=1,
        help="Unused for memmap path (kept for config compatibility).",
    )
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--no_fit_constant", action="store_true")
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument(
        "--closed_form_full_pass_token_threshold",
        type=int,
        default=5_000_000,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
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
        embed_batch_size=args.embed_batch_size,
        device=args.device,
        num_iterations=args.num_iterations,
        fit_constant=not args.no_fit_constant,
        output_json=args.output_json,
        closed_form_full_pass_token_threshold=args.closed_form_full_pass_token_threshold,
        resume_from=args.resume_from,
        max_sequences=args.max_sequences,
    )


def main() -> None:
    config = parse_args()
    run_train(config)


if __name__ == "__main__":
    main()

# nohup python Evo2_train_neuron_boosting_seq.py > output.log 2>&1 &

# 277595