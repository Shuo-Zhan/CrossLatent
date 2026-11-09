#!/usr/bin/env python3
"""
Evo2-40B SAE training via NVIDIA hosted /forward API.

Each batch = one DNA sequence:
  1) call API for embedding_layer activations
  2) train SAE with reconstruction loss only (no logits / CE fidelity)

Per epoch, randomly sample a subset (default: 1000 train / 100 test).

Example:
    python train_sae_evo2_api.py \\
        --epochs 5 \\
        --train_samples_per_epoch 1000 \\
        --test_samples_per_epoch 100 \\
        --activation_dim 8192 \\
        --sae_type relu \\
        --expansion_factor 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio import SeqIO
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from sae_eval import (
    DEFAULT_SAE_EVAL_CONFIG,
    SAEEvalConfig,
    SAEReconstructionLossConfig,
    train_reconstruction_loss,
)
from sae_models import (
    SAE_REGISTRY,
    BatchTopKSAE,
    IdentityDict,
    JumpReLUSAE,
    ReLUSAE,
    TopKSAE,
    build_sae,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DNA_WINDOWS = PROJECT_ROOT / "data" / "DNA" / "windows_1024" / "chr1"
EXTRACT_SCRIPT = (
    PROJECT_ROOT / "grabspark" / "extract_embedding" / "extract_EVO2_embedding.py"
)


def _load_evo2_api_module():
    """Import extract_EVO2_embedding.py helpers without requiring package install."""
    spec = importlib.util.spec_from_file_location("extract_EVO2_embedding", EXTRACT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {EXTRACT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_evo2 = _load_evo2_api_module()
forward_npz = _evo2.forward_npz
_pick_array = _evo2._pick_array
_to_token_major = _evo2._to_token_major
get_api_key = _evo2.get_api_key
DEFAULT_FORWARD_URL = _evo2.DEFAULT_FORWARD_URL
DEFAULT_EMBED_LAYER = _evo2.DEFAULT_EMBED_LAYER


@dataclass
class Evo2DataConfig:
    train_fasta: str = str(DNA_WINDOWS / "train" / "sequences.fasta")
    test_fasta: str = str(DNA_WINDOWS / "test" / "sequences.fasta")


@dataclass
class SAEModelConfig:
    sae_type: str = "relu"
    activation_dim: int = 8192  # Evo2-40B embedding width
    expansion_factor: float = 4.0
    dict_size: Optional[int] = None
    normalize_to_sqrt_d: bool = False
    topk: int = 64

    def resolved_dict_size(self) -> int:
        if self.dict_size is not None:
            return self.dict_size
        return int(self.activation_dim * self.expansion_factor)


@dataclass
class SAETrainingConfig:
    epochs: int = 10
    train_samples_per_epoch: int = 1000
    test_samples_per_epoch: int = 100
    save_epoch: int = 1
    save_path: Optional[str] = None
    output_root: str = str(PROJECT_ROOT / "ckpt" / "Evo2-40B" / "embedding_layer_sae")

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None
    seed: int = 0

    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    warmup_step_ratio: float = 0.1
    decay_start_ratio: float = 0.7
    scheduler_power: float = 2.0
    min_lr: float = 1e-7

    l1_coefficient: float = 0.06
    l1_warmup_ratio: float = 0.05
    dead_dim_coefficient: float = 0.2
    dead_dim_epsilon: float = 1e-7

    resample_enabled_until_epoch_ratio: float = 0.8
    resample_every_n_batches: int = 100
    resample_inactive_ratio: float = 0.5
    resample_activation_threshold: float = 1e-4
    resample_noise_scale: float = 0.01

    # API
    api_url: str = DEFAULT_FORWARD_URL
    embed_layer: str = DEFAULT_EMBED_LAYER
    api_timeout_s: float = 600.0
    api_retries: int = 3
    api_retry_sleep_s: float = 5.0
    log_every_n_batches: int = 10

    data: Evo2DataConfig = field(default_factory=Evo2DataConfig)
    model: SAEModelConfig = field(default_factory=SAEModelConfig)

    def total_training_steps(self) -> int:
        return self.train_samples_per_epoch * self.epochs

    def warmup_steps(self) -> int:
        return int(self.warmup_step_ratio * self.total_training_steps())

    def eval_config(self) -> SAEEvalConfig:
        return SAEEvalConfig(
            reconstruction=SAEReconstructionLossConfig(
                dead_dim_coefficient=self.dead_dim_coefficient,
                dead_dim_epsilon=self.dead_dim_epsilon,
            ),
            fidelity=DEFAULT_SAE_EVAL_CONFIG.fidelity,
        )

    def resolve_save_path(self) -> str:
        if self.save_path:
            path = os.path.abspath(self.save_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.model.dict_size is not None:
                size_tag = f"d{self.model.dict_size}"
            else:
                size_tag = f"{self.model.expansion_factor:g}x"
            run_name = (
                f"{self.model.sae_type}_{size_tag}_"
                f"dim{self.model.activation_dim}_{timestamp}"
            )
            path = os.path.join(self.output_root, run_name)
        os.makedirs(path, exist_ok=True)
        self.save_path = path
        return path


class WarmupPolynomialDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_start_ratio: float = 0.7,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_ratio = decay_start_ratio
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        decay_start = int(self.decay_start_ratio * self.total_steps)
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / max(self.warmup_steps, 1)
            elif step <= decay_start:
                lr = base_lr
            elif step <= self.total_steps:
                decay_step = step - decay_start
                total_decay = max(self.total_steps - decay_start, 1)
                decay_ratio = max(0.0, 1 - decay_step / total_decay)
                lr = (base_lr - self.min_lr) * (decay_ratio ** self.power) + self.min_lr
            else:
                lr = self.min_lr
            lrs.append(lr)
        return lrs


def load_fasta_records(fasta_path: str) -> list[tuple[str, str]]:
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        records.append((rec.id, str(rec.seq).upper()))
    if not records:
        raise ValueError(f"Empty fasta: {fasta_path}")
    return records


def fetch_embedding(
    sequence: str,
    *,
    config: SAETrainingConfig,
    api_key: str,
) -> np.ndarray:
    last_err: Exception | None = None
    for attempt in range(1, config.api_retries + 1):
        try:
            npz = forward_npz(
                sequence,
                output_layers=[config.embed_layer],
                url=config.api_url,
                api_key=api_key,
                timeout_s=config.api_timeout_s,
            )
            emb = _to_token_major(_pick_array(npz, config.embed_layer))
            if emb.shape[0] != len(sequence):
                raise ValueError(
                    f"embedding length {emb.shape[0]} != sequence length {len(sequence)}"
                )
            if emb.shape[1] != config.model.activation_dim:
                raise ValueError(
                    f"embedding dim {emb.shape[1]} != activation_dim "
                    f"{config.model.activation_dim}"
                )
            return emb.astype(np.float32, copy=False)
        except Exception as err:  # noqa: BLE001 — retry transient API failures
            last_err = err
            print(
                f"[API retry {attempt}/{config.api_retries}] {type(err).__name__}: {err}"
            )
            if attempt < config.api_retries:
                time.sleep(config.api_retry_sleep_s * attempt)
    assert last_err is not None
    raise last_err


def build_model(config: SAETrainingConfig) -> torch.nn.Module:
    model_cfg = config.model
    if model_cfg.sae_type not in SAE_REGISTRY:
        valid = ", ".join(sorted(SAE_REGISTRY))
        raise ValueError(f"Unknown sae_type={model_cfg.sae_type!r}. Valid: {valid}")

    dict_size = model_cfg.resolved_dict_size()
    model_kwargs: dict = {"normalize_to_sqrt_d": model_cfg.normalize_to_sqrt_d}
    if model_cfg.sae_type in {"topk", "batch_topk"}:
        model_kwargs["k"] = model_cfg.topk

    model = build_sae(
        model_cfg.sae_type,
        model_cfg.activation_dim,
        dict_size,
        **model_kwargs,
    )
    return model.to(config.device)


def l1_penalty_scale(step: int, total_steps: int, config: SAETrainingConfig) -> float:
    warmup_steps = max(int(config.l1_warmup_ratio * total_steps), 1)
    warmup_factor = min(1.0, step / warmup_steps)
    return config.l1_coefficient * warmup_factor


def supports_neuron_resampling(model: torch.nn.Module) -> bool:
    return not isinstance(model, IdentityDict)


@torch.no_grad()
def prepare_encoder_input(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    normalized, _ = model._normalize_input_and_get_norms(x)
    if isinstance(model, ReLUSAE):
        return normalized - model.bias
    if isinstance(model, (TopKSAE, BatchTopKSAE)):
        return normalized - model.b_dec
    if isinstance(model, JumpReLUSAE) and model.apply_b_dec_to_input:
        return normalized - model.decoder.bias
    return normalized


def batch_low_activation_mask(features: torch.Tensor, threshold: float) -> torch.Tensor:
    return features.abs().mean(dim=0) < threshold


def _sample_encoder_directions(
    encoder_inputs: torch.Tensor,
    n_dead: int,
    noise_scale: float,
) -> torch.Tensor:
    encoder_inputs = encoder_inputs.reshape(-1, encoder_inputs.shape[-1])
    if encoder_inputs.shape[0] >= n_dead:
        sampled_indices = torch.randperm(
            encoder_inputs.shape[0], device=encoder_inputs.device
        )[:n_dead]
        sampled = encoder_inputs[sampled_indices]
    else:
        sampled = encoder_inputs[
            torch.randint(
                0, encoder_inputs.shape[0], (n_dead,), device=encoder_inputs.device
            )
        ]
    new_weights = sampled + noise_scale * torch.randn_like(sampled)
    return new_weights / new_weights.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _apply_resampled_encoder_rows(
    model: torch.nn.Module,
    dead_neurons: torch.Tensor,
    new_encoder_rows: torch.Tensor,
) -> None:
    model.encoder.weight.data[dead_neurons] = new_encoder_rows
    model.encoder.bias.data[dead_neurons] = 0.0
    model.decoder.weight.data[:, dead_neurons] = new_encoder_rows.T
    if isinstance(model, JumpReLUSAE):
        model.threshold.data[dead_neurons] = 0.001


def resample_neurons(
    model: torch.nn.Module,
    dead_neurons: torch.Tensor,
    activations: torch.Tensor,
    noise_scale: float,
) -> None:
    if not supports_neuron_resampling(model) or not dead_neurons.any():
        return
    n_dead = int(dead_neurons.sum().item())
    print(f"Resampling {n_dead} dead neurons")
    with torch.no_grad():
        encoder_inputs = prepare_encoder_input(model, activations)
        new_encoder_rows = _sample_encoder_directions(
            encoder_inputs, n_dead, noise_scale
        )
        _apply_resampled_encoder_rows(model, dead_neurons, new_encoder_rows)


def plot_loss_curve(train_loss_list, test_loss_list, path: str) -> None:
    plt.figure()
    plt.plot(train_loss_list, label="Train recon loss")
    plt.plot(test_loss_list, label="Test recon loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_lr(lr_list, path: str) -> None:
    plt.figure()
    plt.plot(range(len(lr_list)), lr_list)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.savefig(path)
    plt.close()


def sample_indices(n_pool: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    n_sample = min(n_sample, n_pool)
    return rng.choice(n_pool, size=n_sample, replace=False)


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: _LRScheduler,
    records: list[tuple[str, str]],
    indices: np.ndarray,
    config: SAETrainingConfig,
    api_key: str,
    learn_step: int,
    resample_steps: bool,
) -> tuple[float, int]:
    model.train()
    eval_config = config.eval_config()
    epoch_loss = 0.0
    token_count = 0
    batch_num = 0
    steps_since_active = torch.zeros(
        model.dict_size, dtype=torch.int64, device=config.device
    )
    steps_num = 0
    total_steps = config.total_training_steps()

    for idx in tqdm(indices, desc="train"):
        seq_id, sequence = records[int(idx)]
        emb = fetch_embedding(sequence, config=config, api_key=api_key)
        x = torch.from_numpy(emb).to(config.device)

        optimizer.zero_grad(set_to_none=True)
        x_hat, features = model(x, output_features=True)

        if supports_neuron_resampling(model):
            low_activation = batch_low_activation_mask(
                features, config.resample_activation_threshold
            )
            steps_since_active[low_activation] += x.shape[0]
            steps_since_active[~low_activation] = 0
            steps_num += x.shape[0]

        learn_step += 1
        current_l1_scale = l1_penalty_scale(learn_step, total_steps, config)
        loss, l2_loss, l1_loss, dead_dim_loss = train_reconstruction_loss(
            x,
            x_hat,
            features,
            current_l1_penalty_scale=current_l1_scale,
            config=eval_config,
        )

        if batch_num % config.log_every_n_batches == 0:
            print(
                f"[{seq_id}] loss={loss.item():.6f} "
                f"l2={l2_loss.item():.6f} "
                f"l1={l1_loss.item():.6f} "
                f"dead={dead_dim_loss.item():.6f} "
                f"l1_scale={current_l1_scale:.6f}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item() * x.shape[0]
        token_count += x.shape[0]
        batch_num += 1

        if (
            supports_neuron_resampling(model)
            and resample_steps
            and config.resample_every_n_batches > 0
            and batch_num % config.resample_every_n_batches == 0
        ):
            inactive = steps_since_active > int(
                steps_num * config.resample_inactive_ratio
            )
            resample_neurons(model, inactive, x, config.resample_noise_scale)
            steps_since_active.zero_()
            steps_num = 0

    avg_loss = epoch_loss / max(token_count, 1)
    print(f"Epoch train recon loss (per token): {avg_loss}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    return avg_loss, learn_step


@torch.no_grad()
def evaluate_reconstruction(
    model: torch.nn.Module,
    records: list[tuple[str, str]],
    indices: np.ndarray,
    config: SAETrainingConfig,
    api_key: str,
) -> float:
    model.eval()
    eval_config = config.eval_config()
    epoch_loss = 0.0
    token_count = 0

    for idx in tqdm(indices, desc="test"):
        _seq_id, sequence = records[int(idx)]
        emb = fetch_embedding(sequence, config=config, api_key=api_key)
        x = torch.from_numpy(emb).to(config.device)
        x_hat, features = model(x, output_features=True)
        loss, _l2, _l1, _dead = train_reconstruction_loss(
            x,
            x_hat,
            features,
            current_l1_penalty_scale=config.l1_coefficient,
            config=eval_config,
        )
        epoch_loss += loss.item() * x.shape[0]
        token_count += x.shape[0]

    avg_loss = epoch_loss / max(token_count, 1)
    print(f"Epoch test recon loss (per token): {avg_loss}")
    return avg_loss


def run_training(config: SAETrainingConfig) -> None:
    save_path = config.resolve_save_path()
    print(f"Output directory: {save_path}")
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)

    train_records = load_fasta_records(config.data.train_fasta)
    test_records = load_fasta_records(config.data.test_fasta)
    print(
        f"train pool={len(train_records)}, test pool={len(test_records)}, "
        f"per-epoch train={config.train_samples_per_epoch}, "
        f"per-epoch test={config.test_samples_per_epoch}"
    )

    api_key = get_api_key()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model type: {config.model.sae_type}")
    print(f"Activation dim: {config.model.activation_dim}")
    print(f"Dict size: {config.model.resolved_dict_size()}")
    print(f"Total parameters: {total_params:,}")
    print(f"API url={config.api_url} layer={config.embed_layer}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = WarmupPolynomialDecayScheduler(
        optimizer,
        warmup_steps=config.warmup_steps(),
        total_steps=config.total_training_steps(),
        decay_start_ratio=config.decay_start_ratio,
        power=config.scheduler_power,
        min_lr=config.min_lr,
    )

    train_loss_list: list[float] = []
    test_loss_list: list[float] = []
    lr_history = [optimizer.param_groups[0]["lr"]]
    learn_step = 0
    best_test_loss = float("inf")

    for epoch in range(config.epochs):
        print("\n" + "-" * 48)
        print(f"Epoch [{epoch + 1}/{config.epochs}]")

        train_idx = sample_indices(
            len(train_records), config.train_samples_per_epoch, rng
        )
        test_idx = sample_indices(
            len(test_records), config.test_samples_per_epoch, rng
        )

        resample_ok = epoch < int(
            config.resample_enabled_until_epoch_ratio * config.epochs
        )
        train_loss, learn_step = train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            train_records,
            train_idx,
            config,
            api_key,
            learn_step,
            resample_steps=resample_ok,
        )
        train_loss_list.append(train_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        test_loss = evaluate_reconstruction(
            model, test_records, test_idx, config, api_key
        )
        test_loss_list.append(test_loss)

        plot_loss_curve(
            train_loss_list,
            test_loss_list,
            os.path.join(config.save_path, "loss.png"),
        )
        plot_lr(lr_history, os.path.join(config.save_path, "lr.png"))

        if (epoch + 1) % config.save_epoch == 0:
            ckpt_path = os.path.join(config.save_path, f"epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(config.save_path, "best.pt"))
            print("Best model updated.")


def parse_args() -> SAETrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train SAE on Evo2-40B embeddings via hosted API (recon only)."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_samples_per_epoch", type=int, default=2000)
    parser.add_argument("--test_samples_per_epoch", type=int, default=200)
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--output_root", type=str, default=SAETrainingConfig.output_root
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--sae_type",
        type=str,
        default="relu",
        choices=sorted(SAE_REGISTRY.keys()),
    )
    parser.add_argument("--activation_dim", type=int, default=8192)
    parser.add_argument("--expansion_factor", type=float, default=4.0)
    parser.add_argument("--dict_size", type=int, default=None)
    parser.add_argument("--normalize_to_sqrt_d", action="store_true")
    parser.add_argument("--topk", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_step_ratio", type=float, default=0.1)
    parser.add_argument("--decay_start_ratio", type=float, default=0.7)
    parser.add_argument("--scheduler_power", type=float, default=2.0)
    parser.add_argument("--min_lr", type=float, default=1e-7)

    parser.add_argument("--l1_coefficient", type=float, default=0.06)
    parser.add_argument("--l1_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dead_dim_coefficient", type=float, default=0.2)
    parser.add_argument("--dead_dim_epsilon", type=float, default=1e-7)

    parser.add_argument("--resample_enabled_until_epoch_ratio", type=float, default=0.8)
    parser.add_argument("--resample_every_n_batches", type=int, default=100)
    parser.add_argument("--resample_inactive_ratio", type=float, default=0.5)
    parser.add_argument("--resample_activation_threshold", type=float, default=1e-4)
    parser.add_argument("--resample_noise_scale", type=float, default=0.01)

    parser.add_argument("--api_url", type=str, default=DEFAULT_FORWARD_URL)
    parser.add_argument("--embed_layer", type=str, default=DEFAULT_EMBED_LAYER)
    parser.add_argument("--api_timeout_s", type=float, default=600.0)
    parser.add_argument("--api_retries", type=int, default=3)
    parser.add_argument("--api_retry_sleep_s", type=float, default=5.0)
    parser.add_argument("--log_every_n_batches", type=int, default=10)

    parser.add_argument(
        "--train_fasta", type=str, default=Evo2DataConfig.train_fasta
    )
    parser.add_argument("--test_fasta", type=str, default=Evo2DataConfig.test_fasta)

    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    return SAETrainingConfig(
        epochs=args.epochs,
        train_samples_per_epoch=args.train_samples_per_epoch,
        test_samples_per_epoch=args.test_samples_per_epoch,
        save_epoch=args.save_epoch,
        save_path=args.save_path,
        output_root=args.output_root,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_step_ratio=args.warmup_step_ratio,
        decay_start_ratio=args.decay_start_ratio,
        scheduler_power=args.scheduler_power,
        min_lr=args.min_lr,
        l1_coefficient=args.l1_coefficient,
        l1_warmup_ratio=args.l1_warmup_ratio,
        dead_dim_coefficient=args.dead_dim_coefficient,
        dead_dim_epsilon=args.dead_dim_epsilon,
        resample_enabled_until_epoch_ratio=args.resample_enabled_until_epoch_ratio,
        resample_every_n_batches=args.resample_every_n_batches,
        resample_inactive_ratio=args.resample_inactive_ratio,
        resample_activation_threshold=args.resample_activation_threshold,
        resample_noise_scale=args.resample_noise_scale,
        api_url=args.api_url,
        embed_layer=args.embed_layer,
        api_timeout_s=args.api_timeout_s,
        api_retries=args.api_retries,
        api_retry_sleep_s=args.api_retry_sleep_s,
        log_every_n_batches=args.log_every_n_batches,
        data=Evo2DataConfig(
            train_fasta=args.train_fasta,
            test_fasta=args.test_fasta,
        ),
        model=SAEModelConfig(
            sae_type=args.sae_type,
            activation_dim=args.activation_dim,
            expansion_factor=args.expansion_factor,
            dict_size=args.dict_size,
            normalize_to_sqrt_d=args.normalize_to_sqrt_d,
            topk=args.topk,
        ),
    )


def main() -> None:
    config = parse_args()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        config.device = "cpu"
    run_training(config)


if __name__ == "__main__":
    main()

# cd /data1/zshuo23/GraBSpArk/grabspark/sae_train && \
# nohup /home/zshuo23/anaconda3/envs/hfds/bin/python -u train_sae_evo2_api.py > train_sae_evo2_api.log 2>&1 &
# 1845446