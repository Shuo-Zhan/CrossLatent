"""Shared helpers for Evo2-7B-262k + Layer-26 SAE neuron boosting."""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from evo2 import Evo2

D_HIDDEN = 4096
EXPANSION = 8
K = 64
DEFAULT_LAYER = "blocks.26"
DEFAULT_EVO2_CKPT = "/data1/zshuo_zzy/evo2sae/ckpt/evo2_7b_262k.pt"
DEFAULT_SAE_CKPT = (
    "/data1/zshuo_zzy/evo2sae/ckpt/sae-layer26-mixed-expansion_8-k_64.pt"
)


class BatchTopKTiedSAE(nn.Module):
    """TopK SAE matching sparse_autoencoder.ipynb / Goodfire Layer-26 Mixed."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        k: int,
        device: torch.device | str,
        dtype: torch.dtype,
        tiebreaker_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.activation_dim = d_in
        self.dict_size = d_hidden
        self.k = k
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = nn.Parameter(W_mat)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device_ref = device
        self.dtype = dtype
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(device, dtype)

    def encode(self, x: torch.Tensor, tiebreak: bool = False) -> torch.Tensor:
        x = x.to(device=self.W.device, dtype=self.dtype)
        f = torch.nn.functional.relu(x @ self.W + self.b_enc)
        if tiebreak:
            f = f + self.tiebreaker.to(f.device).broadcast_to(f)
        *input_shape, _ = f.shape
        numel = self.k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        out = (
            torch.zeros_like(f.flatten())
            .scatter(-1, f_topk.indices, f_topk.values)
            .reshape(f.shape)
        )
        # neuron_boosting_seq._encode_batch calls .numpy(); bf16 is unsupported.
        return out.float()


def load_topk_sae(
    sae_path: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> BatchTopKTiedSAE:
    state = torch.load(sae_path, weights_only=True, map_location="cpu")
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()
    }
    sae = BatchTopKTiedSAE(D_HIDDEN, D_HIDDEN * EXPANSION, K, device, dtype)
    sae.load_state_dict(state)
    sae.eval()
    return sae


class Evo2LayerEmbedder:
    """
    Drop-in embedder for neuron_boosting_seq._encode_batch.

    Provides extract_embeddings(sequences, layer, batch_size) -> [N_tokens, 4096].
    """

    def __init__(
        self,
        evo2_model: Evo2,
        *,
        layer_name: str = DEFAULT_LAYER,
        device: torch.device | str = "cuda:0",
    ):
        self.evo2_model = evo2_model
        self.layer_name = layer_name
        self.device = torch.device(device)

    @torch.no_grad()
    def extract_embeddings(
        self,
        sequences: list[str],
        layer: int | str | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        # layer arg is kept for API compatibility with ESM embedder; Evo2 uses layer_name.
        _ = layer
        chunks: list[torch.Tensor] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(sequences), bs):
            batch = sequences[start : start + bs]
            # Process one sequence at a time to keep memory predictable for 7B.
            for seq in batch:
                input_ids = torch.tensor(
                    self.evo2_model.tokenizer.tokenize(seq),
                    dtype=torch.int,
                ).unsqueeze(0).to(self.device)
                _, embeddings = self.evo2_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.layer_name],
                )
                emb = embeddings[self.layer_name][0].float().cpu()
                if emb.shape[0] != len(seq):
                    raise ValueError(
                        f"Evo2 tokens {emb.shape[0]} != sequence length {len(seq)}"
                    )
                chunks.append(emb)
                del input_ids, embeddings, emb
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)


class MemmapEmbeddingEmbedder:
    """
    Read precomputed [N_tokens, D] embeddings via numpy memmap for streaming boosting.

    Call set_batch_rows(row_indices) before extract_embeddings so the correct
    token slice is loaded (row batches from neuron_boosting are contiguous).
    """

    def __init__(
        self,
        embedding_npy: str | Path,
        *,
        token_offsets: np.ndarray,
        seq_lengths: list[int],
        expected_hidden: int = D_HIDDEN,
    ):
        self.embedding_npy = str(embedding_npy)
        self.token_offsets = np.asarray(token_offsets, dtype=np.int64)
        self.seq_lengths = list(seq_lengths)
        self._mm = np.load(self.embedding_npy, mmap_mode="r")
        if self._mm.ndim != 2:
            raise ValueError(
                f"Expected embedding shape [num_tokens, hidden], got {self._mm.shape}"
            )
        if int(self._mm.shape[1]) != int(expected_hidden):
            raise ValueError(
                f"Embedding hidden dim {self._mm.shape[1]} != expected {expected_hidden}"
            )
        expected_tokens = int(self.token_offsets[-1]) if len(self.token_offsets) else 0
        if int(self._mm.shape[0]) < expected_tokens:
            raise ValueError(
                f"Embedding rows {self._mm.shape[0]} < required tokens {expected_tokens}"
            )
        self._active_rows: list[int] | None = None
        print(
            f"Memmap embeddings: {self.embedding_npy} "
            f"shape={self._mm.shape} dtype={self._mm.dtype}"
        )

    def set_batch_rows(self, row_indices: list[int]) -> None:
        self._active_rows = list(row_indices)

    def extract_embeddings(
        self,
        sequences: list[str],
        layer: int | str | None = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        _ = layer, batch_size
        if self._active_rows is None:
            raise RuntimeError(
                "MemmapEmbeddingEmbedder.set_batch_rows() must be called "
                "before extract_embeddings()."
            )
        rows = self._active_rows
        if len(rows) != len(sequences):
            raise ValueError(
                f"batch rows {len(rows)} != sequences {len(sequences)}"
            )
        # build_sequence_batches yields contiguous row index ranges
        if rows != list(range(rows[0], rows[0] + len(rows))):
            raise ValueError(
                f"Non-contiguous row batch not supported for memmap slice: {rows[:8]}..."
            )
        for row_idx, seq in zip(rows, sequences):
            if len(seq) != self.seq_lengths[row_idx]:
                raise ValueError(
                    f"Row {row_idx}: sequence len {len(seq)} != "
                    f"expected {self.seq_lengths[row_idx]}"
                )
        start = int(self.token_offsets[rows[0]])
        end = int(self.token_offsets[rows[-1] + 1])
        emb = np.array(self._mm[start:end], dtype=np.float32, copy=True)
        if emb.shape[0] != end - start:
            raise ValueError(
                f"Memmap slice rows {emb.shape[0]} != expected {end - start}"
            )
        return emb


def load_evo2_and_sae(
    *,
    evo2_ckpt: str = DEFAULT_EVO2_CKPT,
    sae_ckpt: str = DEFAULT_SAE_CKPT,
    layer_name: str = DEFAULT_LAYER,
    device: str = "cuda:0",
    sae_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Evo2LayerEmbedder, BatchTopKTiedSAE, torch.device]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Visible CUDA devices={torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to a single GPU before loading Evo2."
        )
    evo2_model = Evo2("evo2_7b_262k", local_path=evo2_ckpt)
    embedder = Evo2LayerEmbedder(
        evo2_model,
        layer_name=layer_name,
        device=torch_device,
    )
    sae = load_topk_sae(sae_ckpt, device=torch_device, dtype=sae_dtype)
    print(f"Loaded Evo2 from: {evo2_ckpt}")
    print(f"Loaded SAE from: {sae_ckpt} (dict_size={sae.dict_size}, k={sae.k})")
    print(f"Embedding layer: {layer_name}, device: {torch_device}")
    return embedder, sae, torch_device


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    seqs: list[str] = []
    name: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    ids.append(name)
                    seqs.append("".join(chunks))
                name = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            ids.append(name)
            seqs.append("".join(chunks))
    return ids, seqs


def load_fasta_continuous_labels(
    fasta_path: str | Path,
    label_npy: str | Path,
    *,
    feature_name: str = "atac_K562",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load DNA fasta + continuous per-base labels.

    label_npy may be [n_seq, seq_len] or length-n_seq object array of vectors.
    Returns (df with sequence/id columns, y flattened to [n_tokens]).
    """
    ids, sequences = read_fasta(fasta_path)
    labels = np.load(label_npy, allow_pickle=True)

    if labels.ndim == 2:
        if labels.shape[0] != len(sequences):
            raise ValueError(
                f"label rows {labels.shape[0]} != n_sequences {len(sequences)}"
            )
        for i, seq in enumerate(sequences):
            if labels.shape[1] != len(seq):
                raise ValueError(
                    f"Row {i}: label width {labels.shape[1]} != seq len {len(seq)}"
                )
        y = labels.astype(np.float64).reshape(-1)
    elif labels.dtype == object:
        if len(labels) != len(sequences):
            raise ValueError(
                f"label objects {len(labels)} != n_sequences {len(sequences)}"
            )
        parts = []
        for i, seq in enumerate(sequences):
            row = np.asarray(labels[i], dtype=np.float64).reshape(-1)
            if row.shape[0] != len(seq):
                raise ValueError(
                    f"Row {i}: label len {row.shape[0]} != seq len {len(seq)}"
                )
            parts.append(row)
        y = np.concatenate(parts, axis=0)
    else:
        raise ValueError(
            f"Unsupported label array shape/dtype: shape={getattr(labels, 'shape', None)}, "
            f"dtype={labels.dtype}"
        )

    # Labels are passed separately as flattened y; keep df lean for streaming context.
    _ = feature_name
    df = pd.DataFrame(
        {
            "sequence_id": ids,
            "uniprot_id": ids,
            "sequence": sequences,
        }
    )
    expected = sum(len(s) for s in sequences)
    if y.shape[0] != expected:
        raise ValueError(f"Flattened labels {y.shape[0]} != total tokens {expected}")
    return df, y


def sequence_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def build_combination_prediction(
    f_seq: np.ndarray,
    neuron_ids: list[int],
    coefficients: list[float],
    intercept_b: float,
) -> np.ndarray:
    pred = np.full(f_seq.shape[0], float(intercept_b), dtype=np.float64)
    for neuron_id, coef in zip(neuron_ids, coefficients):
        pred += float(coef) * f_seq[:, int(neuron_id)]
    return pred
