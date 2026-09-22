"""Shared feature sets and decoder-direction interventions (model readout only)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .api import LinearReadout


def shared_latents(models: Sequence[LinearReadout]) -> list[int]:
    """Intersect independently fitted sets in the first model's selection order."""
    if len(models) < 2:
        raise ValueError("at least two independently fitted models are required")
    if len({(m.dictionary_id, m.n_features) for m in models}) != 1:
        raise ValueError("latent identities can only be compared within one dictionary")
    shared = set(models[0].latent_ids)
    for model in models[1:]:
        shared.intersection_update(model.latent_ids)
    return [i for i in models[0].latent_ids if i in shared]


def position_profile(activations: np.ndarray, offsets: np.ndarray, latent_ids: Sequence[int],
                     n_positions: int = 8) -> np.ndarray:
    """Mean activation per annotated offset; -1 means outside the target region.

    This is a descriptive response profile, not an enrichment or significance test.
    Missing positions are NaN, not zero activation.
    """
    values = np.asarray(activations)
    offsets = np.asarray(offsets)
    if values.ndim != 2 or offsets.shape != (len(values),) or n_positions < 1:
        raise ValueError("expected activations [tokens, features] and offsets [tokens]")
    if not np.issubdtype(offsets.dtype, np.integer) or np.any((offsets < -1) | (offsets >= n_positions)):
        raise ValueError("offsets must be integers from -1 to n_positions-1")
    ids = list(latent_ids)
    if any(i < 0 or i >= values.shape[1] for i in ids):
        raise ValueError("latent ID outside dictionary")
    profile = np.full((len(ids), n_positions), np.nan)
    for offset in range(n_positions):
        mask = offsets == offset
        if mask.any():
            profile[:, offset] = values[mask][:, ids].mean(axis=0)
    return profile


@torch.no_grad()
def ablate_latents(hidden: torch.Tensor, features: torch.Tensor, decoder: torch.Tensor,
                   latent_ids: Sequence[int]) -> torch.Tensor:
    """Subtract selected decoder contributions from ORIGINAL hidden states.

    decoder has shape [hidden_dim, dictionary_size]. Caller supplies activations
    in the corresponding raw hidden-state coordinate system. SAE reconstruction
    residual and decoder bias are retained. No region mask is used for ablation.
    """
    ids = list(latent_ids)
    if hidden.ndim != 2 or features.ndim != 2 or decoder.ndim != 2:
        raise ValueError("hidden, features and decoder must be matrices")
    if hidden.shape[0] != features.shape[0] or decoder.shape != (hidden.shape[1], features.shape[1]):
        raise ValueError("incompatible hidden/features/decoder shapes")
    if len(ids) != len(set(ids)) or any(i < 0 or i >= features.shape[1] for i in ids):
        raise ValueError("latent IDs must be unique and in range")
    if hidden.device != features.device or hidden.device != decoder.device:
        raise ValueError("all intervention tensors must be on the same device")
    return hidden - features[:, ids].to(hidden.dtype) @ decoder[:, ids].to(hidden.dtype).T


@torch.no_grad()
def readout_effects(hidden: torch.Tensor, features: torch.Tensor, decoder: torch.Tensor,
                    latent_ids: Sequence[int], head: Callable[[torch.Tensor], torch.Tensor],
                    token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token effects through a frozen caller-provided language-model head.

    Caller must put the head in eval mode. Margins use log-sum-exp of all competing
    logits,  rather than the maximum competitor.
    """
    changed = ablate_latents(hidden, features, decoder, latent_ids)
    original, intervened = head(hidden), head(changed)
    if original.ndim != 2 or original.shape != intervened.shape or original.shape[0] != len(hidden):
        raise ValueError("head must return [tokens, vocabulary] logits")
    if token_ids.shape != (len(hidden),) or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a LongTensor with one ID per token")
    if torch.any((token_ids < 0) | (token_ids >= original.shape[1])):
        raise ValueError("token ID outside output vocabulary")
    index = token_ids[:, None]

    def metrics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logp = logits.log_softmax(-1)
        other = logits.clone().scatter(1, index, -torch.inf)
        margin = logits.gather(1, index).squeeze(1) - other.logsumexp(-1)
        return logp, logp.gather(1, index).squeeze(1), margin

    logp, correct, margin = metrics(original)
    logq, new_correct, new_margin = metrics(intervened)
    return {"delta_correct_logp": new_correct - correct, "delta_margin": new_margin - margin,
            "kl_original_to_intervened": (logp.exp() * (logp - logq)).sum(-1),
            "original_top1": original.argmax(-1) == token_ids,
            "intervened_top1": intervened.argmax(-1) == token_ids}
