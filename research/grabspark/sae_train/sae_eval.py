"""
Standardized SAE evaluation and training loss utilities.

Metrics:
- Reconstruction loss (L2 + L1 + dead-dimension penalty)
- Fidelity / loss recovered via downstream logits head patching
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Module = nn.Module

__all__ = [
    "SAEReconstructionLossConfig",
    "SAEFidelityConfig",
    "SAEEvalConfig",
    "DEFAULT_SAE_EVAL_CONFIG",
    "calculate_loss_recovered",
    "calculate_cross_entropy",
    "calculate_fidelity",
    "train_reconstruction_loss",
    "ReconstructionLossOutput",
]


@dataclass
class SAEReconstructionLossConfig:
    """Hyperparameters for SAE reconstruction training loss."""

    l1_penalty_scale: float = 1.0
    dead_dim_coefficient: float = 0.2
    dead_dim_epsilon: float = 1e-7
    l1_p: float = 1.0


@dataclass
class SAEFidelityConfig:
    """Hyperparameters for fidelity / loss-recovered evaluation."""

    token_start_index: int = 1
    token_end_offset: int = 1
    loss_recovered_clip_min: float = 0.0
    loss_recovered_clip_max: float = 1.0
    loss_recovered_percent_scale: float = 100.0
    denominator_epsilon: float = 0.0


@dataclass
class SAEEvalConfig:
    """Top-level evaluation configuration."""

    reconstruction: SAEReconstructionLossConfig = field(
        default_factory=SAEReconstructionLossConfig
    )
    fidelity: SAEFidelityConfig = field(default_factory=SAEFidelityConfig)


DEFAULT_SAE_EVAL_CONFIG = SAEEvalConfig()


@dataclass(frozen=True)
class ReconstructionLossOutput:
    loss: Tensor
    l2_loss: Tensor
    l1_loss: Tensor
    dead_dim_loss: Tensor


def _resolve_reconstruction_config(
    config: Optional[Union[SAEReconstructionLossConfig, SAEEvalConfig]] = None,
) -> SAEReconstructionLossConfig:
    if config is None:
        return DEFAULT_SAE_EVAL_CONFIG.reconstruction
    if isinstance(config, SAEEvalConfig):
        return config.reconstruction
    return config


def _resolve_fidelity_config(
    config: Optional[Union[SAEFidelityConfig, SAEEvalConfig]] = None,
) -> SAEFidelityConfig:
    if config is None:
        return DEFAULT_SAE_EVAL_CONFIG.fidelity
    if isinstance(config, SAEEvalConfig):
        return config.fidelity
    return config


def calculate_loss_recovered(
    ce_autoencoder: float,
    ce_identity: float,
    ce_zero_ablation: float,
    config: Optional[Union[SAEFidelityConfig, SAEEvalConfig]] = None,
) -> float:
    """
    Compute loss recovered metric for an SAE.

    100% means SAE reconstruction matches original activations as well as possible.
    0% means SAE is as bad as zero ablation.
    """
    cfg = _resolve_fidelity_config(config)

    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity

    if np.isclose(denominator, cfg.denominator_epsilon):
        return 0.0

    loss_recovered = 1.0 - (numerator / denominator)
    loss_recovered = np.clip(
        loss_recovered,
        cfg.loss_recovered_clip_min,
        cfg.loss_recovered_clip_max,
    )
    return loss_recovered * cfg.loss_recovered_percent_scale


def calculate_cross_entropy(
    model_output: Tensor,
    batch_tokens: Tensor,
    batch_len: Sequence[int],
    config: Optional[Union[SAEFidelityConfig, SAEEvalConfig]] = None,
) -> float:
    """Average token cross-entropy over valid positions in each sequence."""
    cfg = _resolve_fidelity_config(config)
    losses: list[float] = []

    for j, seq_len in enumerate(batch_len):
        start = cfg.token_start_index
        end = seq_len + cfg.token_end_offset
        seq_logits = model_output[j, start:end]
        seq_tokens = batch_tokens[j, start:end]
        losses.append(F.cross_entropy(seq_logits, seq_tokens).item())

    return float(np.mean(losses))


def calculate_fidelity(
    sae_output: Tensor,
    batch_label: Tensor,
    batch_len: Sequence[int],
    batch_input: Tensor,
    output_head: Module,
    config: Optional[Union[SAEFidelityConfig, SAEEvalConfig]] = None,
) -> dict[str, float]:
    """Evaluate SAE fidelity via downstream logits-head patching."""
    cfg = _resolve_fidelity_config(config)

    output_head.eval()
    with torch.no_grad():
        sae_logits = output_head(sae_output)
        sae_loss = calculate_cross_entropy(
            sae_logits, batch_label, batch_len, config=cfg
        )

        orig_logits = output_head(batch_input)
        orig_loss = calculate_cross_entropy(
            orig_logits, batch_label, batch_len, config=cfg
        )

        zero_logits = output_head(torch.zeros_like(batch_input))
        zero_loss = calculate_cross_entropy(
            zero_logits, batch_label, batch_len, config=cfg
        )

    loss_recovered = calculate_loss_recovered(
        ce_autoencoder=sae_loss,
        ce_identity=orig_loss,
        ce_zero_ablation=zero_loss,
        config=cfg,
    )

    return {
        "pct_loss_recovered": loss_recovered,
        "CE_w_sae_patching": sae_loss,
        "CE_identity": orig_loss,
        "CE_zero_ablation": zero_loss,
    }


def train_reconstruction_loss(
    x: Tensor,
    x_hat: Tensor,
    features: Tensor,
    current_l1_penalty_scale: Optional[float] = None,
    config: Optional[Union[SAEReconstructionLossConfig, SAEEvalConfig]] = None,
    return_details: bool = False,
) -> Union[
    Tensor,
    tuple[Tensor, Tensor, Tensor, Tensor],
    ReconstructionLossOutput,
]:
    """
    Compute SAE reconstruction loss:

        loss = l2 + (l1 + dead_dim_coefficient * dead_dim) * l1_scale

    Args:
        x: Input activations.
        x_hat: Reconstructed activations.
        features: Encoded sparse features.
        current_l1_penalty_scale: Runtime L1 scale (e.g. warmup schedule).
            Overrides ``config.l1_penalty_scale`` when provided.
        config: Loss hyperparameter config.
        return_details: If True, return ``ReconstructionLossOutput``; otherwise
            return ``(loss, l2_loss, l1_loss, dead_dim_loss)`` for backward
            compatibility.
    """
    cfg = _resolve_reconstruction_config(config)
    l1_scale = (
        current_l1_penalty_scale
        if current_l1_penalty_scale is not None
        else cfg.l1_penalty_scale
    )

    l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
    l1_loss = features.norm(p=cfg.l1_p, dim=-1).mean()

    dim_mean_activation = torch.mean(features, dim=0)
    dead_dim_loss = torch.mean(
        1.0 / (torch.abs(dim_mean_activation) + cfg.dead_dim_epsilon)
    )

    loss = l2_loss + (l1_loss + dead_dim_loss * cfg.dead_dim_coefficient) * l1_scale

    if return_details:
        return ReconstructionLossOutput(
            loss=loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            dead_dim_loss=dead_dim_loss,
        )

    return loss, l2_loss, l1_loss, dead_dim_loss


def with_reconstruction_overrides(**kwargs: Any) -> SAEEvalConfig:
    """Helper to clone default config with reconstruction overrides."""
    return replace(
        DEFAULT_SAE_EVAL_CONFIG,
        reconstruction=replace(DEFAULT_SAE_EVAL_CONFIG.reconstruction, **kwargs),
    )


def with_fidelity_overrides(**kwargs: Any) -> SAEEvalConfig:
    """Helper to clone default config with fidelity overrides."""
    return replace(
        DEFAULT_SAE_EVAL_CONFIG,
        fidelity=replace(DEFAULT_SAE_EVAL_CONFIG.fidelity, **kwargs),
    )
