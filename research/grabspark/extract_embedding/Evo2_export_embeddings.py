#!/usr/bin/env python3
"""
Export Evo2-7B-262k layer-26 embeddings to a memmap/npy for offline boosting.

Output shape: [num_tokens, 4096], row order matches fasta sequence order.

Example:
    CUDA_VISIBLE_DEVICES=2 python Evo2_export_embeddings.py \\
        --fasta_path data/test_sample_3000_a/sequences.fasta \\
        --output_npy data/test_sample_3000_a/evo2_7b_262k_layer26_embedding.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_parser_early = argparse.ArgumentParser(add_help=False)
_parser_early.add_argument("--cuda_visible_devices", type=str, default=None)
_early_args, _ = _parser_early.parse_known_args()
if _early_args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.cuda_visible_devices
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evo2 import Evo2

from evo2_sae_utils import (  # noqa: E402
    DEFAULT_EVO2_CKPT,
    DEFAULT_LAYER,
    D_HIDDEN,
    Evo2LayerEmbedder,
    read_fasta,
)

DEFAULT_FASTA = SCRIPT_DIR / "data" / "test_sample_3000_a" / "sequences.fasta"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "data" / "test_sample_3000_a" / "evo2_7b_262k_layer26_embedding.npy"
)


def export_embeddings(args: argparse.Namespace) -> Path:
    ids, sequences = read_fasta(args.fasta_path)
    if args.max_sequences is not None:
        n = int(args.max_sequences)
        ids = ids[:n]
        sequences = sequences[:n]

    seq_lengths = [len(s) for s in sequences]
    total_tokens = int(sum(seq_lengths))
    print(
        f"Sequences: {len(sequences)}, tokens: {total_tokens}, "
        f"embedding dim: {D_HIDDEN}"
    )

    out_path = Path(args.output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Only need Evo2 embedder.
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Visible CUDA devices={torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to a single GPU."
        )
    device = torch.device(args.device)
    evo2_model = Evo2("evo2_7b_262k", local_path=args.evo2_ckpt)
    embedder = Evo2LayerEmbedder(
        evo2_model,
        layer_name=args.layer_name,
        device=device,
    )
    print(f"Loaded Evo2 from: {args.evo2_ckpt}, layer={args.layer_name}, device={device}")

    mm = np.lib.format.open_memmap(
        str(out_path),
        mode="w+",
        dtype=np.float32,
        shape=(total_tokens, D_HIDDEN),
    )

    offset = 0
    for seq_id, seq in tqdm(
        list(zip(ids, sequences)),
        desc="Export Evo2 embeddings",
    ):
        emb = embedder.extract_embeddings(
            sequences=[seq],
            layer=26,
            batch_size=1,
        )
        emb_np = emb.numpy().astype(np.float32, copy=False)
        if emb_np.shape[0] != len(seq):
            raise ValueError(
                f"{seq_id}: emb rows {emb_np.shape[0]} != seq len {len(seq)}"
            )
        mm[offset : offset + len(seq)] = emb_np
        offset += len(seq)
        del emb, emb_np
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mm.flush()
    if offset != total_tokens:
        raise RuntimeError(f"Wrote {offset} rows, expected {total_tokens}")

    print(f"Saved embeddings: {out_path} shape={mm.shape} dtype={mm.dtype}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Evo2 layer-26 embeddings to .npy for offline boosting."
    )
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument("--output_npy", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--evo2_ckpt", type=str, default=DEFAULT_EVO2_CKPT)
    parser.add_argument("--layer_name", type=str, default=DEFAULT_LAYER)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_embeddings(args)


if __name__ == "__main__":
    main()
