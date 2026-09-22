"""Load the study's fixed SAE checkpoints; upstream language models remain external."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BatchTopKTiedSAE(nn.Module):
    """Layer-26 Mixed encoding rule from the original Evo2 result workflow."""

    def __init__(self, width: int, dictionary_size: int, k: int = 64) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(width, dictionary_size))
        self.b_enc = nn.Parameter(torch.zeros(dictionary_size))
        self.b_dec = nn.Parameter(torch.zeros(width))
        self.dict_size = dictionary_size
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.relu(x.to(self.W.dtype) @ self.W + self.b_enc)
        values, indices = torch.topk(f.flatten(), self.k * len(x))
        return torch.zeros_like(f.flatten()).scatter(0, indices, values).reshape_as(f).float()


def load_sae(kind: str, checkpoint: str | Path, *, device: str = "cpu",
             dtype: str | None = None) -> nn.Module:
    """Explicit checkpoint paths only; never download models or modify sys.path."""
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    dtype = dtype or ("bfloat16" if kind == "dna" else "float32")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    if kind == "protein":
        from interplm.sae.dictionary import ReLUSAE

        model = ReLUSAE.from_pretrained(str(checkpoint), device=device)
    elif kind == "rna":
        from ._core.sae_models import load_sae_from_checkpoint

        model = load_sae_from_checkpoint(str(checkpoint), sae_type="relu", device=device)
    elif kind == "dna":
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        state = {key.replace("_orig_mod.", "").replace("module.", ""): value for key, value in state.items()}
        model = BatchTopKTiedSAE(*state["W"].shape)
        model.load_state_dict(state)
    else:
        raise ValueError("kind must be protein, rna or dna")
    return model.to(device=device, dtype=getattr(torch, dtype)).eval()


@torch.no_grad()
def encode_embeddings(embeddings: str | Path, output: str | Path, *, kind: str,
                       checkpoint: str | Path, batch_tokens: int, device: str = "cpu",
                       dtype: str | None = None) -> dict:
    """Encode an ordered float matrix, writing mmap features and protocol metadata.

    Fixed contiguous token batches. For BatchTopK this is part of the definition
    of the resulting dictionary_id; it is not a promise of batch invariance.
    Historical sequence-aligned caches must keep their historical protocol.
    """
    if batch_tokens < 1:
        raise ValueError("batch_tokens must be positive")
    source, target = Path(embeddings).resolve(strict=True), Path(output).resolve()
    checkpoint = Path(checkpoint).resolve(strict=True)
    if target in {source, checkpoint}:
        raise ValueError("output must differ from embeddings and checkpoint")
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or not len(values):
        raise ValueError("embeddings must be a nonempty [tokens, hidden_dim] NPY")
    model = load_sae(kind, checkpoint, device=device, dtype=dtype)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(values), model.dict_size))
    parameter = next(model.parameters())
    for start in range(0, len(values), batch_tokens):
        batch = np.array(values[start:start + batch_tokens], copy=True)
        if not np.isfinite(batch).all():
            raise ValueError("embeddings must be finite")
        encoded = model.encode(torch.as_tensor(batch, device=device, dtype=parameter.dtype))
        result[start:start + len(batch)] = encoded.float().cpu().numpy()
    result.flush()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    protocol = {"kind": kind, "checkpoint_sha256": digest.hexdigest(), "batch_tokens": batch_tokens,
                "batching": "contiguous-token-chunks-v1", "compute_dtype": str(parameter.dtype),
                "output_dtype": "float32", "feature_count": model.dict_size}
    protocol["dictionary_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol
