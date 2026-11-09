#!/usr/bin/env python3
"""
Token-level binary classifier on frozen ESM2 650M embeddings.

Trains an MLP to predict mask_alpha_helix labels from per-residue embeddings,
then evaluates on the held-out test split.

Example:
    python train.py

    python train.py \\
        --epochs 20 \\
        --batch_size 8192 \\
        --device cuda:0 \\
        --output_dir data/Protein/uniprot_sprot_2500_2500/results
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "Protein" / "uniprot_sprot_2500_2500"
BOOST_PIPELINE_DIR = SCRIPT_DIR.parent / "boost_pipeline"

sys.path.insert(0, str(BOOST_PIPELINE_DIR))

from neuron_boosting import (  # pyright: ignore[reportMissingImports]
    compute_all_labels,
    load_evaluation_dataframe,
    validate_csv_embedding_pairing,
    validate_row_labels,
)

DEFAULT_TRAIN_CSV = DATA_DIR / "alpha_helix_labels_train_2500.csv"
DEFAULT_TEST_CSV = DATA_DIR / "alpha_helix_labels_test_2500.csv"
DEFAULT_TRAIN_EMB = DATA_DIR / "ESM2_650M_embedding_train_2500.npy"
DEFAULT_TEST_EMB = DATA_DIR / "ESM2_650M_embedding_test_2500.npy"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[1] / "ckpt" / "ESM2_alpha_helix"
print(DEFAULT_OUTPUT_DIR)

@dataclass
class TrainConfig:
    train_csv: str = str(DEFAULT_TRAIN_CSV)
    test_csv: str = str(DEFAULT_TEST_CSV)
    train_embedding_npy: str = str(DEFAULT_TRAIN_EMB)
    test_embedding_npy: str = str(DEFAULT_TEST_EMB)
    label_column: str = "mask_alpha_helix"
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    epochs: int = 15
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    eval_batch_size: int = 16384


class TokenClassificationDataset(Dataset):
    """Random-access token dataset backed by memory-mapped embeddings."""

    def __init__(self, embedding_npy: str, labels: np.ndarray):
        self.embeddings = np.load(embedding_npy, mmap_mode="r")
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected embedding shape [num_tokens, hidden_dim], got {self.embeddings.shape}"
            )
        if len(labels) != self.embeddings.shape[0]:
            raise ValueError(
                f"Label/embedding length mismatch: labels={len(labels)}, "
                f"embeddings={self.embeddings.shape[0]}"
            )
        self.labels = labels.astype(np.float32)
        self.input_dim = int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.embeddings[index], dtype=np.float32).copy()
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return torch.from_numpy(x), y


class TokenClassifier(nn.Module):
    """MLP head for token-level binary classification on frozen PLM embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(csv_path: str, label_column: str) -> tuple[np.ndarray, list[int]]:
    df = load_evaluation_dataframe(csv_path)
    validate_row_labels(df, label_column)
    labels = compute_all_labels(df, [label_column])[label_column]
    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    return labels, seq_lengths


def macro_f1_for_sequence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    f1_pos_den = 2 * tp + fp + fn
    f1_pos = (2 * tp / f1_pos_den) if f1_pos_den > 0 else 0.0
    f1_neg_den = 2 * tn + fp + fn
    f1_neg = (2 * tn / f1_neg_den) if f1_neg_den > 0 else 0.0
    return 0.5 * (f1_pos + f1_neg)


def token_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_prob >= threshold

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: TokenClassificationDataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    total_tokens = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        batch_size_actual = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size_actual
        total_tokens += batch_size_actual

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataset: TokenClassificationDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    for x_batch, _ in loader:
        logits = model(x_batch.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def evaluate_split(
    model: nn.Module,
    dataset: TokenClassificationDataset,
    labels: np.ndarray,
    seq_lengths: list[int],
    device: torch.device,
    batch_size: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = predict_probabilities(model, dataset, device, batch_size)
    token_level = token_metrics(labels, probs, threshold=threshold)

    offset = 0
    seq_macro_f1s: list[float] = []
    for seq_len in seq_lengths:
        y_seq = labels[offset : offset + seq_len]
        p_seq = probs[offset : offset + seq_len]
        seq_macro_f1s.append(macro_f1_for_sequence(y_seq, p_seq >= threshold))
        offset += seq_len

    return {
        **token_level,
        "mean_sequence_macro_f1": float(np.mean(seq_macro_f1s)),
        "median_sequence_macro_f1": float(np.median(seq_macro_f1s)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x_batch, y_batch in tqdm(loader, desc="train", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_tokens += batch_size

    return total_loss / max(total_tokens, 1)


def build_model(input_dim: int, config: TrainConfig) -> TokenClassifier:
    return TokenClassifier(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )


def plot_loss_curve(
    train_losses: list[float],
    test_losses: list[float],
    path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs, test_losses, marker="o", label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_macro_f1_curve(
    train_macro_f1s: list[float],
    test_macro_f1s: list[float],
    path: Path,
) -> None:
    epochs = range(1, len(train_macro_f1s) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_macro_f1s, marker="o", label="Train Mean Sequence Macro F1")
    ax.plot(epochs, test_macro_f1s, marker="o", label="Test Mean Sequence Macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Sequence Macro F1")
    ax.set_title("Training and Test Mean Sequence Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    train_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(payload, path)


def run_training(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train data...")
    train_df = load_evaluation_dataframe(config.train_csv)
    validate_csv_embedding_pairing(train_df, config.train_embedding_npy)
    train_labels, train_seq_lengths = load_labels(config.train_csv, config.label_column)
    train_dataset = TokenClassificationDataset(config.train_embedding_npy, train_labels)

    print("Loading test data...")
    test_df = load_evaluation_dataframe(config.test_csv)
    validate_csv_embedding_pairing(test_df, config.test_embedding_npy)
    test_labels, test_seq_lengths = load_labels(config.test_csv, config.label_column)
    test_dataset = TokenClassificationDataset(config.test_embedding_npy, test_labels)

    model = build_model(train_dataset.input_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    pos_rate = float(np.mean(train_labels))
    print(
        f"Train tokens: {len(train_dataset):,}, test tokens: {len(test_dataset):,}, "
        f"input_dim: {train_dataset.input_dim}, positive rate: {pos_rate:.4f}"
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    history: list[dict[str, float]] = []
    train_losses: list[float] = []
    test_losses: list[float] = []
    train_macro_f1s: list[float] = []
    test_macro_f1s: list[float] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=config.grad_clip,
        )
        scheduler.step()

        avg_test_loss = evaluate_loss(
            model=model,
            dataset=test_dataset,
            criterion=criterion,
            device=device,
            batch_size=config.eval_batch_size,
        )
        train_metrics_epoch = evaluate_split(
            model=model,
            dataset=train_dataset,
            labels=train_labels,
            seq_lengths=train_seq_lengths,
            device=device,
            batch_size=config.eval_batch_size,
        )
        test_metrics_epoch = evaluate_split(
            model=model,
            dataset=test_dataset,
            labels=test_labels,
            seq_lengths=test_seq_lengths,
            device=device,
            batch_size=config.eval_batch_size,
        )

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_macro_f1s.append(train_metrics_epoch["mean_sequence_macro_f1"])
        test_macro_f1s.append(test_metrics_epoch["mean_sequence_macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "train_mean_sequence_macro_f1": train_metrics_epoch["mean_sequence_macro_f1"],
                "test_mean_sequence_macro_f1": test_metrics_epoch["mean_sequence_macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={avg_train_loss:.6f} | test_loss={avg_test_loss:.6f} | "
            f"train_macro_f1={train_metrics_epoch['mean_sequence_macro_f1']:.6f} | "
            f"test_macro_f1={test_metrics_epoch['mean_sequence_macro_f1']:.6f}"
        )

    train_metrics = train_metrics_epoch
    test_metrics = test_metrics_epoch

    elapsed = time.time() - start_time
    checkpoint_path = output_dir / "ESM2_650M_alpha_helix_classifier.pt"
    results_path = output_dir / "ESM2_650M_alpha_helix_classifier_results.json"
    loss_curve_path = output_dir / "ESM2_650M_alpha_helix_classifier_loss_curve.png"
    macro_f1_curve_path = output_dir / "ESM2_650M_alpha_helix_classifier_macro_f1_curve.png"

    plot_loss_curve(train_losses, test_losses, loss_curve_path)
    plot_macro_f1_curve(train_macro_f1s, test_macro_f1s, macro_f1_curve_path)

    save_checkpoint(
        checkpoint_path,
        model,
        config,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    results = {
        "config": asdict(config),
        "input_dim": train_dataset.input_dim,
        "num_train_tokens": len(train_dataset),
        "num_test_tokens": len(test_dataset),
        "train_positive_rate": pos_rate,
        "elapsed_seconds": elapsed,
        "history": history,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_mean_sequence_macro_f1s": train_macro_f1s,
        "test_mean_sequence_macro_f1s": test_macro_f1s,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "loss_curve_path": str(loss_curve_path),
        "macro_f1_curve_path": str(macro_f1_curve_path),
    }

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nTrain metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Saved results to {results_path}")
    print(f"Saved loss curve to {loss_curve_path}")
    print(f"Saved macro F1 curve to {macro_f1_curve_path}")
    return results


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train token-level alpha helix classifier.")
    parser.add_argument("--train_csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--train_embedding_npy", default=str(DEFAULT_TRAIN_EMB))
    parser.add_argument("--test_embedding_npy", default=str(DEFAULT_TEST_EMB))
    parser.add_argument("--label_column", default="mask_alpha_helix")
    parser.add_argument("--hidden_dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--eval_batch_size", type=int, default=16384)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one integer.")

    return TrainConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_embedding_npy=args.train_embedding_npy,
        test_embedding_npy=args.test_embedding_npy,
        label_column=args.label_column,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
