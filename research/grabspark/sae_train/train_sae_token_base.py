#!/usr/bin/env python3
"""
Token-level SAE training script.

Uses standardized modules:
- ``sae_models`` for model construction
- ``sae_eval`` for reconstruction loss and fidelity evaluation

Example:
    python train_sae_token_base.py \\
        --sae_type relu \\
        --activation_dim 768 \\
        --expansion_factor 12 \\
        --l1_coefficient 0.06 \\
        --dead_dim_coefficient 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sae_eval import (
    DEFAULT_SAE_EVAL_CONFIG,
    SAEEvalConfig,
    SAEFidelityConfig,
    SAEReconstructionLossConfig,
    calculate_fidelity,
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

from ERNIERNA_LogitsHead.ERNIERNA_Head import ERNIERNALogitsHead  # pyright: ignore[reportMissingImports]

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "RNA"

@dataclass
class SAEDataConfig:
    train_fasta_prefix: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/val_8_1_1"
    )
    val_fasta_prefix: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/val_8_1_1"
    )
    test_fasta_prefix: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/test_8_1_1"
    )
    train_embedding_h5: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/"
        "ERNIERNA_embeddings/layer_12/val_0_205367.h5"
    )
    val_embedding_h5: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/"
        "ERNIERNA_embeddings/layer_12/val_0_205367.h5"
    )     
    test_embedding_h5: str = str(
        DATA_DIR / "rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/"
        "ERNIERNA_embeddings/layer_12/test_0_205380.h5"
    )


@dataclass
class SAEModelConfig:
    sae_type: str = "relu"
    activation_dim: int = 768
    expansion_factor: float = 12.0
    dict_size: Optional[int] = None
    normalize_to_sqrt_d: bool = False
    topk: int = 64

    def resolved_dict_size(self) -> int:
        if self.dict_size is not None:
            return self.dict_size
        return int(self.activation_dim * self.expansion_factor)


@dataclass
class SAETrainingConfig:
    epochs: int = 50
    batch_size: int = 128_000
    num_workers: int = 0
    save_epoch: int = 1
    save_path: Optional[str] = None
    output_root: str = str(SCRIPT_DIR.parents[1] / "ckpt" / "ERNIE-RNA" / "layer_12_token")

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None
    detect_anomaly: bool = False

    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    warmup_step_ratio: float = 0.1
    decay_start_ratio: float = 0.7
    scheduler_power: float = 2.0
    min_lr: float = 1e-7
    steps_per_epoch: Optional[int] = None

    l1_coefficient: float = 0.06
    l1_warmup_ratio: float = 0.05
    dead_dim_coefficient: float = 0.2
    dead_dim_epsilon: float = 1e-7

    resample_enabled_until_epoch_ratio: float = 0.8
    resample_every_n_batches: int = 100
    resample_inactive_ratio: float = 0.5
    resample_activation_threshold: float = 1e-4
    resample_noise_scale: float = 0.01

    logits_head_path: str = str(SCRIPT_DIR / "ERNIERNA_LogitsHead" / "model_weights.pt")

    log_every_n_batches: int = 10

    data: SAEDataConfig = field(default_factory=SAEDataConfig)
    model: SAEModelConfig = field(default_factory=SAEModelConfig)

    def configure_steps_per_epoch(self, train_batches: int) -> None:
        """Set steps_per_epoch from train dataloader length unless explicitly provided."""
        if self.steps_per_epoch is None:
            self.steps_per_epoch = train_batches
        elif self.steps_per_epoch != train_batches:
            print(
                f"Warning: --steps_per_epoch={self.steps_per_epoch} "
                f"!= len(train_loader)={train_batches}; "
                "LR/L1 schedules may not align with actual training steps."
            )

    def total_training_steps(self) -> int:
        if self.steps_per_epoch is None:
            raise RuntimeError(
                "steps_per_epoch is unset. Call configure_steps_per_epoch() first."
            )
        return self.steps_per_epoch * self.epochs

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
        """
        Resolve the output directory.

        Uses ``save_path`` when provided; otherwise auto-generates a unique
        subdirectory under ``output_root``.
        """
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


def read_fasta_biopython(fasta_file: str) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]


def seq_to_rna_index(sequences: list[str]) -> tuple[np.ndarray, list[int]]:
    rna_len_lst = [len(seq) for seq in sequences]
    max_len = max(rna_len_lst)
    rna_index = np.ones((len(sequences), max_len + 2))

    for i, sequence in enumerate(sequences):
        for j in range(rna_len_lst[i]):
            char = sequence[j]
            if char in set("Aa"):
                rna_index[i, j + 1] = 5
            elif char in set("Cc"):
                rna_index[i, j + 1] = 7
            elif char in set("Gg"):
                rna_index[i, j + 1] = 4
            elif char in set("TUtu"):
                rna_index[i, j + 1] = 6
            else:
                rna_index[i, j + 1] = 3
        rna_index[i, rna_len_lst[i] + 1] = 2

    rna_index[:, 0] = 0
    return rna_index, rna_len_lst


class TokenEmbeddingDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        fasta_prefix: str,
        embedding_h5: str,
        load_sequence: bool = False,
    ):
        self.batch_size = batch_size
        self.fasta_prefix = fasta_prefix
        self.embedding_h5 = embedding_h5
        self.load_sequence = load_sequence

    def __getitem__(self, index: int):
        with h5py.File(self.embedding_h5, "r") as handle:
            start = index * self.batch_size
            end = (index + 1) * self.batch_size
            x = handle["data"][start:end]

        if self.load_sequence:
            with h5py.File(f"{self.fasta_prefix}.h5", "r") as handle:
                char_code = handle["char_codes"][start:end]
            label = "".join(chr(i) for i in char_code)
        else:
            label = "?"

        return torch.from_numpy(x).float(), label

    def __len__(self) -> int:
        with h5py.File(self.embedding_h5, "r") as handle:
            num_tokens = handle["data"].shape[0]
        return num_tokens // self.batch_size


class TokenDataModule:
    def __init__(self, config: SAETrainingConfig):
        self.config = config
        data = config.data
        self.train_dataset = TokenEmbeddingDataset(
            config.batch_size, data.train_fasta_prefix, data.train_embedding_h5
        )
        self.val_dataset = TokenEmbeddingDataset(
            config.batch_size,
            data.val_fasta_prefix,
            data.val_embedding_h5,
            load_sequence=True,
        )
        self.test_dataset = TokenEmbeddingDataset(
            config.batch_size,
            data.test_fasta_prefix,
            data.test_embedding_h5,
            load_sequence=True,
        )

    def _build_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.test_dataset, shuffle=False)


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
    """Return False for architectures where resampling is not applicable."""
    return not isinstance(model, IdentityDict)


@torch.no_grad()
def prepare_encoder_input(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Map raw activations to the same space the encoder sees.

    Applies input normalization (when enabled) and subtracts pre-encoding bias
    (``bias`` / ``b_dec``) when the architecture uses one.
    """
    normalized, _ = model._normalize_input_and_get_norms(x)
    if isinstance(model, ReLUSAE):
        return normalized - model.bias
    if isinstance(model, (TopKSAE, BatchTopKSAE)):
        return normalized - model.b_dec
    if isinstance(model, JumpReLUSAE) and model.apply_b_dec_to_input:
        return normalized - model.decoder.bias
    return normalized


def batch_low_activation_mask(
    features: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Neurons whose mean |activation| in this batch is below ``threshold``."""
    return features.abs().mean(dim=0) < threshold


def _sample_encoder_directions(
    encoder_inputs: torch.Tensor,
    n_dead: int,
    noise_scale: float,
) -> torch.Tensor:
    encoder_inputs = encoder_inputs.reshape(-1, encoder_inputs.shape[-1])
    if encoder_inputs.shape[0] >= n_dead:
        sampled_indices = torch.randperm(encoder_inputs.shape[0], device=encoder_inputs.device)[
            :n_dead
        ]
        sampled = encoder_inputs[sampled_indices]
    else:
        sampled = encoder_inputs[
            torch.randint(0, encoder_inputs.shape[0], (n_dead,), device=encoder_inputs.device)
        ]

    new_weights = sampled + noise_scale * torch.randn_like(sampled)
    return new_weights / new_weights.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _apply_resampled_encoder_rows(
    model: torch.nn.Module,
    dead_neurons: torch.Tensor,
    new_encoder_rows: torch.Tensor,
) -> None:
    """Write resampled encoder rows and keep decoder columns consistent per architecture."""
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
    if not supports_neuron_resampling(model):
        return

    if not dead_neurons.any():
        print("No dead neurons to resample.")
        return

    n_dead = int(dead_neurons.sum().item())
    print(f"Resampling {n_dead} dead neurons")

    with torch.no_grad():
        encoder_inputs = prepare_encoder_input(model, activations)
        new_encoder_rows = _sample_encoder_directions(encoder_inputs, n_dead, noise_scale)
        _apply_resampled_encoder_rows(model, dead_neurons, new_encoder_rows)


def plot_loss_curve(train_loss_list, test_loss_list, path: str) -> None:
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curve")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_accu_curve(train_accu_list, test_accu_list, path: str) -> None:
    plt.plot(train_accu_list, label="Training Accuracy")
    plt.plot(test_accu_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test AA seqs Accuracy")
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


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: _LRScheduler,
    data_loader: DataLoader,
    config: SAETrainingConfig,
    learn_step: int,
    resample_steps: bool,
) -> tuple[float, int]:
    model.train()
    eval_config = config.eval_config()

    epoch_loss = 0.0
    batch_num = 0
    token_count = 0
    steps_since_active = torch.zeros(model.dict_size, dtype=torch.int64, device=config.device)
    steps_num = 0
    total_steps = config.total_training_steps()

    for x, _label in tqdm(data_loader):
        optimizer.zero_grad(set_to_none=True)

        x = x.to(config.device).squeeze(0)
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
                f"loss={loss.item():.6f} "
                f"l2={l2_loss.item():.6f} "
                f"l1={l1_loss.item():.6f} "
                f"dead={dead_dim_loss.item():.6f} "
                f"l1_scale={current_l1_scale:.6f}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        token_count += x.shape[0]
        batch_num += 1

        if (
            supports_neuron_resampling(model)
            and resample_steps
            and config.resample_every_n_batches > 0
            and batch_num % config.resample_every_n_batches == 0
        ):
            inactive = steps_since_active > int(steps_num * config.resample_inactive_ratio)
            resample_neurons(model, inactive, x, config.resample_noise_scale)
            steps_since_active.zero_()
            steps_num = 0

    avg_loss = epoch_loss / max(token_count, 1)
    print(f"Epoch training loss (per token): {avg_loss}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    return avg_loss, learn_step


@torch.no_grad()
def evaluate_fidelity(
    model: torch.nn.Module,
    data_loader: DataLoader,
    config: SAETrainingConfig,
) -> tuple[float, float]:
    model.eval()
    output_head = ERNIERNALogitsHead()
    output_head.load_state_dict(
        torch.load(config.logits_head_path, map_location=config.device, weights_only=True)
    )
    output_head.to(config.device)
    eval_config = config.eval_config()

    epoch_loss = 0.0
    epoch_recovered = 0.0
    batch_num = 0

    for x, label in tqdm(data_loader):
        x = x.to(config.device).squeeze(0)
        label = "".join(label)
        rna_index, rna_len_lst = seq_to_rna_index([label])
        rna_index = torch.from_numpy(rna_index).long().to(config.device)

        x_hat, _features = model(x, output_features=True)
        x_hat = F.pad(x_hat.unsqueeze(0), (0, 0, 1, 1, 0, 0), mode="constant", value=0)
        x_padded = F.pad(x.unsqueeze(0), (0, 0, 1, 1, 0, 0), mode="constant", value=0)

        metrics = calculate_fidelity(
            x_hat,
            rna_index,
            rna_len_lst,
            x_padded,
            output_head,
            config=eval_config,
        )

        epoch_loss += metrics["CE_w_sae_patching"]
        epoch_recovered += metrics["pct_loss_recovered"]
        batch_num += 1

    avg_loss = epoch_loss / max(batch_num, 1)
    avg_recovered = epoch_recovered / max(batch_num, 1)
    print(f"Epoch eval CE: {avg_loss}")
    print(f"Epoch eval loss recovered (%): {avg_recovered}")
    return avg_loss, avg_recovered


def run_training(config: SAETrainingConfig) -> None:
    save_path = config.resolve_save_path()
    print(f"Output directory: {save_path}")
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)

    data_module = TokenDataModule(config)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    config.configure_steps_per_epoch(len(train_loader))
    print(
        f"train batches={len(train_loader)}, "
        f"val batches={len(val_loader)}, "
        f"test batches={len(test_loader)}, "
        f"steps_per_epoch={config.steps_per_epoch}"
    )

    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model type: {config.model.sae_type}")
    print(f"Activation dim: {config.model.activation_dim}")
    print(f"Dict size: {config.model.resolved_dict_size()}")
    print(f"Total parameters: {total_params:,}")

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
    test_recovered_list: list[float] = []
    lr_history = [optimizer.param_groups[0]["lr"]]
    learn_step = 0
    best_test_loss = float("inf")

    for epoch in range(config.epochs):
        print("\n" + "-" * 48)
        print(f"Epoch [{epoch + 1}/{config.epochs}]")

        resample_steps = epoch < int(config.resample_enabled_until_epoch_ratio * config.epochs)
        train_loss, learn_step = train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            config,
            learn_step,
            resample_steps=resample_steps,
        )
        train_loss_list.append(train_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        test_loss, test_recovered = evaluate_fidelity(model, test_loader, config)
        test_loss_list.append(test_loss)
        test_recovered_list.append(test_recovered)

        plot_loss_curve(train_loss_list, test_loss_list, os.path.join(config.save_path, "loss.png"))
        plot_accu_curve(
            test_recovered_list,
            test_recovered_list,
            os.path.join(config.save_path, "accu.png"),
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
    parser = argparse.ArgumentParser(description="Train token-level SAE")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128_000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output directory for checkpoints and plots. "
        "If omitted, a run folder is auto-created under --output_root.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=SAETrainingConfig.output_root,
        help="Base directory used when --save_path is not set.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--detect_anomaly", action="store_true")

    parser.add_argument(
        "--sae_type",
        type=str,
        default="relu",
        choices=sorted(SAE_REGISTRY.keys()),
        help="SAE architecture registered in sae_models.SAE_REGISTRY",
    )
    parser.add_argument("--activation_dim", type=int, default=768)
    parser.add_argument("--expansion_factor", type=float, default=12.0)
    parser.add_argument("--dict_size", type=int, default=None)
    parser.add_argument("--normalize_to_sqrt_d", action="store_true")
    parser.add_argument("--topk", type=int, default=64, help="Used by topk / batch_topk SAE")

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_step_ratio", type=float, default=0.1)
    parser.add_argument("--decay_start_ratio", type=float, default=0.7)
    parser.add_argument("--scheduler_power", type=float, default=2.0)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Batches per epoch for LR/L1 schedules. "
        "Defaults to len(train_dataloader) when omitted.",
    )

    parser.add_argument("--l1_coefficient", type=float, default=0.06)
    parser.add_argument("--l1_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dead_dim_coefficient", type=float, default=0.2)
    parser.add_argument("--dead_dim_epsilon", type=float, default=1e-7)

    parser.add_argument("--resample_enabled_until_epoch_ratio", type=float, default=0.8)
    parser.add_argument("--resample_every_n_batches", type=int, default=100)
    parser.add_argument("--resample_inactive_ratio", type=float, default=0.5)
    parser.add_argument(
        "--resample_activation_threshold",
        type=float,
        default=1e-4,
        help="Batch mean |activation| below this counts as inactive for resampling.",
    )
    parser.add_argument("--resample_noise_scale", type=float, default=0.01)

    parser.add_argument("--logits_head_path", type=str, default=SAETrainingConfig.logits_head_path)
    parser.add_argument("--log_every_n_batches", type=int, default=10)

    parser.add_argument("--train_fasta_prefix", type=str, default=SAEDataConfig.train_fasta_prefix)
    parser.add_argument("--val_fasta_prefix", type=str, default=SAEDataConfig.val_fasta_prefix)
    parser.add_argument("--test_fasta_prefix", type=str, default=SAEDataConfig.test_fasta_prefix)
    parser.add_argument("--train_embedding_h5", type=str, default=SAEDataConfig.train_embedding_h5)
    parser.add_argument("--val_embedding_h5", type=str, default=SAEDataConfig.val_embedding_h5)
    parser.add_argument("--test_embedding_h5", type=str, default=SAEDataConfig.test_embedding_h5)

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    return SAETrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_epoch=args.save_epoch,
        save_path=args.save_path,
        output_root=args.output_root,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        detect_anomaly=args.detect_anomaly,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_step_ratio=args.warmup_step_ratio,
        decay_start_ratio=args.decay_start_ratio,
        scheduler_power=args.scheduler_power,
        min_lr=args.min_lr,
        steps_per_epoch=args.steps_per_epoch,
        l1_coefficient=args.l1_coefficient,
        l1_warmup_ratio=args.l1_warmup_ratio,
        dead_dim_coefficient=args.dead_dim_coefficient,
        dead_dim_epsilon=args.dead_dim_epsilon,
        resample_enabled_until_epoch_ratio=args.resample_enabled_until_epoch_ratio,
        resample_every_n_batches=args.resample_every_n_batches,
        resample_inactive_ratio=args.resample_inactive_ratio,
        resample_activation_threshold=args.resample_activation_threshold,
        resample_noise_scale=args.resample_noise_scale,
        logits_head_path=args.logits_head_path,
        log_every_n_batches=args.log_every_n_batches,
        data=SAEDataConfig(
            train_fasta_prefix=args.train_fasta_prefix,
            val_fasta_prefix=args.val_fasta_prefix,
            test_fasta_prefix=args.test_fasta_prefix,
            train_embedding_h5=args.train_embedding_h5,
            val_embedding_h5=args.val_embedding_h5,
            test_embedding_h5=args.test_embedding_h5,
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
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    print("Training config:")
    print(asdict(config))
    run_training(config)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed seconds: {time.time() - start:.2f}")
