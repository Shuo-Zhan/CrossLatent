"""
Greedy neuron boosting on precomputed activation matrix f.

Callers load embeddings, labels, and models elsewhere; this module handles
batching sequences, SAE encoding, boosting selection, and threshold evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

THRESHOLD_GRID = np.linspace(0.0, 1.0, 11)
LABEL_PREFIX = "mask_"

# Sentinel neuron_id for the preliminary all-ones feature (x = 1 everywhere).
CONST_NEURON_ID = -1


@dataclass
class BoostingStep:
    step: int
    neuron_id: int
    a: float


@dataclass
class StepMetrics:
    step: int
    neuron_id: int
    a: float
    composite_r2: float
    macro_f1: float


@dataclass
class ClosedFormFit:
    """Closed-form least-squares fit on original activations: y = b + sum_i a_i * x_i."""

    neuron_ids: list[int]
    a: np.ndarray
    b: float
    composite_r2: float
    macro_f1: float


@dataclass
class NeuronBoostingConfig:
    """Parameters for greedy neuron selection (excluding f and y)."""

    num_sae_neurons: int
    seq_lengths: list[int] | None = None
    prediction_threshold: float = 0.5
    fit_constant: bool = True
    solve_closed_form: bool = True
    compute_step_f1: bool = True
    verbose: bool = True


@dataclass
class NeuronBoostingResult:
    steps: list[BoostingStep]
    step_metrics: list[StepMetrics]
    y_pred: np.ndarray
    y_residual: np.ndarray
    closed_form: ClosedFormFit | None = None
    y_pred_history: list[np.ndarray] = field(default_factory=list)
    composite_r2_history: list[float] = field(default_factory=list)


def _fit_all_neurons_a(
    f: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Fit y = a*x (no intercept) for every neuron on CPU."""
    x = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    y_col = y[:, None]
    sum_xy = np.sum(x * y_col, axis=0)
    sum_x2 = np.sum(x * x, axis=0)

    a = np.divide(
        sum_xy,
        sum_x2,
        out=np.zeros_like(sum_xy, dtype=np.float64),
        where=sum_x2 > 0,
    )
    sums = {
        "sum_xy": sum_xy,
        "sum_x2": sum_x2,
        "sum_y2": float(np.sum(y * y)),
    }
    return a, sums


def _ss_res_all_neurons(
    a: np.ndarray,
    sums: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Per-neuron SS_res for y = a*x without building [N, dict_size] predictions."""
    sum_xy = sums["sum_xy"]
    sum_x2 = sums["sum_x2"]
    sum_y2 = float(sums["sum_y2"])

    return sum_y2 - 2.0 * a * sum_xy + a * a * sum_x2


def compute_optimal_r2(
    f: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float]:
    """Find best neuron for y = a*x (no intercept). Returns (neuron_id, a)."""
    y = np.asarray(y, dtype=np.float64)
    a, sums = _fit_all_neurons_a(f, y)
    ss_res = _ss_res_all_neurons(a, sums)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0, 0.0

    r2 = 1.0 - ss_res / ss_tot
    best_neuron_id = int(np.argmax(r2))
    return best_neuron_id, float(a[best_neuron_id])


def remove_parallel_from_f_inplace(
    f: np.ndarray,
    direction: np.ndarray,
    eps: float = 1e-12,
) -> None:
    """
    Remove the component parallel to direction from every column of f (in place).

    For unit direction u = direction / ||direction||, update f[:, j] <- f[:, j] - (f[:, j]·u)u.
    """
    u = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(u))
    if norm <= eps:
        return
    u = u / norm
    coef = f.T @ u
    f -= np.outer(u, coef)


def _fit_ones_a(y: np.ndarray) -> float:
    """Fit y = a*x with x=1 for all tokens. Optimal a = mean(y)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.mean(y))


def compute_mean_macro_f1(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray,
    seq_lengths: list[int],
    threshold: float = 0.5,
) -> float:
    """Threshold predictions, macro F1 per sequence, then average."""
    y_true_binary = y_true >= 0.5
    y_pred_binary = y_pred_continuous >= threshold

    boundaries = np.cumsum([0] + seq_lengths)
    tp = np.add.reduceat(
        (y_true_binary & y_pred_binary).astype(np.int64),
        boundaries[:-1],
    )
    tn = np.add.reduceat(
        (~y_true_binary & ~y_pred_binary).astype(np.int64),
        boundaries[:-1],
    )
    fp = np.add.reduceat(
        (~y_true_binary & y_pred_binary).astype(np.int64),
        boundaries[:-1],
    )
    fn = np.add.reduceat(
        (y_true_binary & ~y_pred_binary).astype(np.int64),
        boundaries[:-1],
    )

    f1_pos_den = 2 * tp + fp + fn
    f1_pos = np.divide(
        2 * tp,
        f1_pos_den,
        out=np.zeros_like(tp, dtype=np.float64),
        where=f1_pos_den > 0,
    )
    f1_neg_den = 2 * tn + fp + fn
    f1_neg = np.divide(
        2 * tn,
        f1_neg_den,
        out=np.zeros_like(tn, dtype=np.float64),
        where=f1_neg_den > 0,
    )
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return float(np.mean(macro_f1)) if macro_f1.size > 0 else 0.0


def compute_composite_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """R² of composite prediction vs original y."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _resolve_seq_lengths(
    y: np.ndarray,
    seq_lengths: list[int] | None,
) -> list[int]:
    if seq_lengths is None:
        return [int(y.shape[0])]
    if sum(seq_lengths) != y.shape[0]:
        raise ValueError(
            f"sum(seq_lengths)={sum(seq_lengths)} != len(y)={y.shape[0]}"
        )
    return seq_lengths


def _run_boosting_step(
    *,
    step: int,
    neuron_id: int,
    a: float,
    delta: np.ndarray,
    y_original: np.ndarray,
    y_residual: np.ndarray,
    y_pred: np.ndarray,
    seq_lengths: list[int],
    threshold: float,
    steps: list[BoostingStep],
    step_metrics: list[StepMetrics],
    verbose: bool,
    compute_step_f1: bool = True,
) -> None:
    """Apply one boosting update and record step metrics."""
    y_residual -= delta
    y_pred += delta

    boost_step = BoostingStep(step=step, neuron_id=neuron_id, a=a)
    steps.append(boost_step)

    composite_r2 = compute_composite_r2(y_original, y_pred)
    if compute_step_f1:
        macro_f1 = compute_mean_macro_f1(
            y_original,
            y_pred,
            seq_lengths,
            threshold=threshold,
        )
    else:
        macro_f1 = float("nan")
    metrics = StepMetrics(
        step=step,
        neuron_id=neuron_id,
        a=a,
        composite_r2=composite_r2,
        macro_f1=macro_f1,
    )
    step_metrics.append(metrics)

    if verbose:
        neuron_label = "const" if neuron_id == CONST_NEURON_ID else f"{neuron_id:>5}"
        if compute_step_f1:
            print(
                f"  Step {step:>2}: neuron_id={neuron_label}, "
                f"a={a:.6f} | "
                f"composite R²={composite_r2:.6f}, macro F1={macro_f1:.6f}"
            )
        else:
            print(
                f"  Step {step:>2}: neuron_id={neuron_label}, "
                f"a={a:.6f} | composite R²={composite_r2:.6f}"
            )


def _selected_sae_neuron_ids(steps: list[BoostingStep]) -> list[int]:
    """Unique SAE neuron ids from boosting steps, preserving first-seen order."""
    seen: set[int] = set()
    neuron_ids: list[int] = []
    for step in steps:
        if step.neuron_id == CONST_NEURON_ID:
            continue
        if step.neuron_id in seen:
            continue
        seen.add(step.neuron_id)
        neuron_ids.append(step.neuron_id)
    return neuron_ids


def solve_closed_form_coefficients(
    f: np.ndarray,
    y: np.ndarray,
    steps: list[BoostingStep],
    seq_lengths: list[int] | None = None,
    threshold: float = 0.5,
) -> ClosedFormFit:
    """
    Refit y = b + a1*x1 + a2*x2 + ... on original activations.

    Boosting step coefficients are computed on deflated activations and are not
    directly usable as a closed-form linear model.
    """
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    seq_lengths = _resolve_seq_lengths(y, seq_lengths)
    neuron_ids = _selected_sae_neuron_ids(steps)

    if neuron_ids:
        x_selected = f[:, neuron_ids]
        design = np.column_stack([np.ones(len(y), dtype=np.float64), x_selected])
    else:
        design = np.ones((len(y), 1), dtype=np.float64)

    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    b = float(beta[0])
    a = np.asarray(beta[1:], dtype=np.float64) if neuron_ids else np.zeros(0, dtype=np.float64)

    y_pred = design @ beta
    composite_r2 = compute_composite_r2(y, y_pred)
    macro_f1 = compute_mean_macro_f1(
        y,
        y_pred,
        seq_lengths,
        threshold=threshold,
    )

    return ClosedFormFit(
        neuron_ids=neuron_ids,
        a=a,
        b=b,
        composite_r2=composite_r2,
        macro_f1=macro_f1,
    )


def print_closed_form_fit(closed_form: ClosedFormFit) -> None:
    """Print closed-form coefficients in a readable format."""
    print("\nClosed-form refit on original activations: y = b + sum(a_i * x_i)")
    print(f"  b = {closed_form.b:.6f}")
    if closed_form.neuron_ids:
        for neuron_id, coef in zip(closed_form.neuron_ids, closed_form.a):
            print(f"  neuron_id={neuron_id:>5}, a = {coef:.6f}")
    else:
        print("  (no SAE neurons selected)")
    print(
        f"  composite R²={closed_form.composite_r2:.6f}, "
        f"macro F1={closed_form.macro_f1:.6f}"
    )


def print_closed_form_refit_summary(closed_form: ClosedFormFit) -> None:
    """Print one-line closed-form summary (matches training log format)."""
    print(
        f"Closed-form refit: b={closed_form.b:.6f}, "
        f"neurons={closed_form.neuron_ids}, "
        f"coefficients={[float(x) for x in closed_form.a]}"
    )


def iteration_output_path(output_json: str, iteration: int) -> Path:
    """Map base JSON path to iteration checkpoint path, e.g. results.json -> results_5.json."""
    path = Path(output_json)
    return path.with_name(f"{path.stem}_{iteration}{path.suffix}")


def _checkpoint_iteration_suffix(steps: list[BoostingStep]) -> int:
    """Suffix index = number of selected SAE neurons so far (0 after const-only)."""
    return len(_selected_sae_neuron_ids(steps))


def _partial_boosting_result(
    steps: list[BoostingStep],
    step_metrics: list[StepMetrics],
    y_pred: np.ndarray,
    y_residual: np.ndarray,
    closed_form: ClosedFormFit,
) -> NeuronBoostingResult:
    return NeuronBoostingResult(
        steps=list(steps),
        step_metrics=list(step_metrics),
        y_pred=y_pred,
        y_residual=y_residual,
        closed_form=closed_form,
    )


def save_iteration_checkpoint(
    checkpoint: IterationCheckpointConfig,
    *,
    steps: list[BoostingStep],
    step_metrics: list[StepMetrics],
    y_pred: np.ndarray,
    y_residual: np.ndarray,
    verbose: bool = True,
) -> ClosedFormFit:
    """Run closed-form refit on current boosting steps and save checkpoint JSON."""
    closed_form = solve_closed_form_coefficients(
        f=checkpoint.f,
        y=checkpoint.y,
        steps=steps,
        seq_lengths=checkpoint.seq_lengths,
        threshold=0.5,
    )
    if verbose:
        print_closed_form_refit_summary(closed_form)

    y_pred_closed = build_closed_form_prediction(checkpoint.f, closed_form)
    combo_sweep = sweep_macro_f1_for_prediction(
        y=checkpoint.y,
        y_pred_continuous=y_pred_closed,
        seq_lengths=checkpoint.seq_lengths,
        thresholds=checkpoint.threshold_grid,
    )
    iteration = _checkpoint_iteration_suffix(steps)
    boosting = _partial_boosting_result(
        steps=steps,
        step_metrics=step_metrics,
        y_pred=y_pred,
        y_residual=y_residual,
        closed_form=closed_form,
    )
    payload = {
        "config": checkpoint.config_dict,
        "iteration": iteration,
        "num_boosting_steps": len(steps),
        "threshold_grid": [float(x) for x in checkpoint.threshold_grid],
        "f_shape": list(checkpoint.f_shape),
        "features": {
            checkpoint.feature: {
                "num_tokens": int(checkpoint.f.shape[0]),
                "boosting_steps": boosting_steps_to_records(boosting),
                "boosting_final_composite_r2": float(step_metrics[-1].composite_r2),
                "combination": {
                    **closed_form_to_dict(closed_form),
                    "best_threshold": combo_sweep.best_threshold,
                    "best_macro_f1": combo_sweep.best_macro_f1,
                    "threshold_sweep": combo_sweep.sweep,
                },
                "best_single_neuron": single_neuron_to_dict(
                    checkpoint.best_single_neuron
                ),
            }
        },
    }
    output_path = iteration_output_path(checkpoint.output_json, iteration)
    save_results_json(str(output_path), payload)
    if verbose:
        print(f"Saved iteration checkpoint: {output_path}")
    return closed_form


def select_neurons_iteratively(
    f: np.ndarray,
    y: np.ndarray,
    num_sae_neurons: int,
    *,
    seq_lengths: list[int] | None = None,
    prediction_threshold: float = 0.5,
    fit_constant: bool = True,
    solve_closed_form: bool = True,
    compute_step_f1: bool = True,
    verbose: bool = True,
    collect_predictions: bool = False,
    iteration_checkpoint: IterationCheckpointConfig | None = None,
) -> NeuronBoostingResult:
    """
    Greedy neuron boosting on a precomputed activation matrix.

    Args:
        f: Activation matrix [num_tokens, dict_size], original (not modified).
        y: Target labels [num_tokens].
        num_sae_neurons: Number of SAE neurons to select after the optional constant step.
        seq_lengths: Per-sequence token counts for macro F1. Defaults to one sequence.
        prediction_threshold: Threshold for macro F1 when compute_step_f1=True.
        fit_constant: If True, Step 1 fits y = a*1 before SAE neuron selection.
        solve_closed_form: If True, refit y = b + sum(a_i x_i) on original f at the end.
        compute_step_f1: If True, compute macro F1 at each boosting step.
        verbose: Print per-step progress.
        collect_predictions: If True, store y_pred after each step in the result.
        iteration_checkpoint: If set, run closed-form refit and save JSON after each step.

    Returns:
        NeuronBoostingResult with steps, metrics, final predictions, optional closed form.
    """

    def _maybe_save_iteration_checkpoint() -> None:
        if iteration_checkpoint is None:
            return
        save_iteration_checkpoint(
            iteration_checkpoint,
            steps=steps,
            step_metrics=step_metrics,
            y_pred=y_pred,
            y_residual=y_residual,
            verbose=verbose,
        )
    if num_sae_neurons < 0:
        raise ValueError(f"num_sae_neurons must be >= 0, got {num_sae_neurons}")

    y_original = np.asarray(y, dtype=np.float64)
    if f.shape[0] != y_original.shape[0]:
        raise ValueError(
            f"f rows {f.shape[0]} != y length {y_original.shape[0]}"
        )

    seq_lengths = _resolve_seq_lengths(y_original, seq_lengths)
    y_residual = y_original.copy()
    y_pred = np.zeros_like(y_original)
    f_boost = np.array(f, dtype=np.float64, copy=True)
    steps: list[BoostingStep] = []
    step_metrics: list[StepMetrics] = []
    y_pred_history: list[np.ndarray] = []
    composite_r2_history: list[float] = []
    step_num = 0

    if fit_constant:
        step_num += 1
        a_const = _fit_ones_a(y_residual)
        _run_boosting_step(
            step=step_num,
            neuron_id=CONST_NEURON_ID,
            a=a_const,
            delta=np.full_like(y_residual, a_const),
            y_original=y_original,
            y_residual=y_residual,
            y_pred=y_pred,
            seq_lengths=seq_lengths,
            threshold=prediction_threshold,
            steps=steps,
            step_metrics=step_metrics,
            verbose=verbose,
            compute_step_f1=compute_step_f1,
        )
        if collect_predictions:
            y_pred_history.append(y_pred.copy())
            composite_r2_history.append(step_metrics[-1].composite_r2)

        ones = np.ones(len(y_original), dtype=np.float64)
        remove_parallel_from_f_inplace(f_boost, ones)
        _maybe_save_iteration_checkpoint()

    for _ in range(num_sae_neurons):
        step_num += 1
        neuron_id, a = compute_optimal_r2(f_boost, y_residual)
        x = f_boost[:, neuron_id].copy()
        delta = a * x
        _run_boosting_step(
            step=step_num,
            neuron_id=neuron_id,
            a=a,
            delta=delta,
            y_original=y_original,
            y_residual=y_residual,
            y_pred=y_pred,
            seq_lengths=seq_lengths,
            threshold=prediction_threshold,
            steps=steps,
            step_metrics=step_metrics,
            verbose=verbose,
            compute_step_f1=compute_step_f1,
        )
        remove_parallel_from_f_inplace(f_boost, x)
        if collect_predictions:
            y_pred_history.append(y_pred.copy())
            composite_r2_history.append(step_metrics[-1].composite_r2)
        _maybe_save_iteration_checkpoint()

    closed_form = None
    if solve_closed_form:
        closed_form = solve_closed_form_coefficients(
            f=f,
            y=y_original,
            steps=steps,
            seq_lengths=seq_lengths,
            threshold=prediction_threshold,
        )
        if verbose:
            print_closed_form_fit(closed_form)

    return NeuronBoostingResult(
        steps=steps,
        step_metrics=step_metrics,
        y_pred=y_pred,
        y_residual=y_residual,
        closed_form=closed_form,
        y_pred_history=y_pred_history,
        composite_r2_history=composite_r2_history,
    )


def select_neurons_iteratively_from_config(
    f: np.ndarray,
    y: np.ndarray,
    config: NeuronBoostingConfig,
    *,
    collect_predictions: bool = False,
    iteration_checkpoint: IterationCheckpointConfig | None = None,
) -> NeuronBoostingResult:
    """Convenience wrapper using NeuronBoostingConfig."""
    return select_neurons_iteratively(
        f=f,
        y=y,
        num_sae_neurons=config.num_sae_neurons,
        seq_lengths=config.seq_lengths,
        prediction_threshold=config.prediction_threshold,
        fit_constant=config.fit_constant,
        solve_closed_form=config.solve_closed_form,
        compute_step_f1=config.compute_step_f1,
        verbose=config.verbose,
        collect_predictions=collect_predictions,
        iteration_checkpoint=iteration_checkpoint,
    )


# Backward-compatible aliases
def iterative_boosting_fit(
    f: np.ndarray,
    y: np.ndarray,
    seq_lengths: list[int],
    num_iterations: int,
    threshold: float = 0.5,
) -> tuple[list[BoostingStep], list[StepMetrics]]:
    result = select_neurons_iteratively(
        f=f,
        y=y,
        num_sae_neurons=num_iterations,
        seq_lengths=seq_lengths,
        prediction_threshold=threshold,
        solve_closed_form=False,
    )
    return result.steps, result.step_metrics


def iterative_boosting_fit_collect(
    f: np.ndarray,
    y: np.ndarray,
    seq_lengths: list[int],
    num_iterations: int,
    verbose: bool = False,
) -> tuple[list[BoostingStep], list[np.ndarray], list[float]]:
    result = select_neurons_iteratively(
        f=f,
        y=y,
        num_sae_neurons=num_iterations,
        seq_lengths=seq_lengths,
        solve_closed_form=False,
        verbose=verbose,
        collect_predictions=True,
    )
    return result.steps, result.y_pred_history, result.composite_r2_history


@dataclass
class ActivationData:
    """Precomputed activations and labels ready for boosting."""

    f: np.ndarray
    labels: dict[str, np.ndarray]
    seq_lengths: list[int]


@dataclass
class ThresholdSweepResult:
    best_threshold: float
    best_macro_f1: float
    sweep: list[dict[str, float]]


@dataclass
class CombinationThresholdResult:
    closed_form: ClosedFormFit
    y_pred: np.ndarray
    sweep: ThresholdSweepResult


@dataclass
class SingleNeuronThresholdResult:
    neuron_id: int
    best_threshold: float
    best_macro_f1: float
    sweep: list[dict[str, float]]


@dataclass
class FeatureEvaluationResult:
    feature: str
    boosting: NeuronBoostingResult
    num_tokens: int
    combination: CombinationThresholdResult
    best_single_neuron: SingleNeuronThresholdResult | None = None


@dataclass
class FeatureEvalParams:
    """Task hyperparameters for boosting + threshold evaluation."""

    num_iterations: int
    fit_constant: bool = True
    search_best_single_neuron: bool = True
    output_json: str = ""
    save_iteration_checkpoints: bool = True
    config_dict: dict[str, Any] = field(default_factory=dict)
    threshold_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 11)
    )


@dataclass
class IterationCheckpointConfig:
    """Save closed-form refit + JSON after each boosting step."""

    output_json: str
    feature: str
    config_dict: dict[str, Any]
    f: np.ndarray
    y: np.ndarray
    seq_lengths: list[int]
    threshold_grid: np.ndarray
    f_shape: tuple[int, ...]
    best_single_neuron: SingleNeuronThresholdResult | None = None

def normalize_mask_columns(df):
    """
    Expand mask columns that pandas coerced to scalar 0/1 when reading CSV.

    All-zero masks like "000...0" are often read as int 0; restore to length=sequence.
    """
    import pandas as pd

    df = df.copy()
    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    if not mask_cols:
        return df

    seq_lens = df["sequence"].astype(str).str.len()
    fixed = 0
    for col in mask_cols:
        normalized: list[str] = []
        for value, seq_len in zip(df[col], seq_lens):
            if pd.isna(value):
                normalized.append("0" * seq_len)
                fixed += 1
                continue
            text = str(value).strip()
            if len(text) == seq_len:
                normalized.append(text)
            elif text in ("0", "0.0") or value == 0:
                normalized.append("0" * seq_len)
                fixed += 1
            elif seq_len == 1 and text in ("1", "1.0"):
                normalized.append("1")
            else:
                normalized.append(text)
        df[col] = normalized

    if fixed:
        print(f"Normalized {fixed} scalar mask value(s) to full sequence length")
    return df


def load_evaluation_dataframe(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns:
        raise ValueError("CSV must contain a 'sequence' column.")
    return normalize_mask_columns(df)


def validate_csv_embedding_pairing(df, embedding_npy: str) -> int:
    """Validate sum(sequence lengths) == embedding rows; raise on mismatch."""
    import numpy as np

    total_seq_len = int(df["sequence"].astype(str).str.len().sum())

    embeddings = np.load(embedding_npy, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embedding shape [num_tokens, hidden_dim], got {embeddings.shape}"
        )
    embedding_num_tokens = int(embeddings.shape[0])

    if total_seq_len != embedding_num_tokens:
        raise ValueError(
            f"Pairing FAILED: sum(sequence lengths)={total_seq_len} "
            f"!= embedding.shape[0]={embedding_num_tokens}"
        )

    print(
        f"Pairing OK: sum(sequence lengths)={total_seq_len} "
        f"== embedding.shape[0]={embedding_num_tokens}"
    )
    return embedding_num_tokens


def validate_row_labels(df, label_column: str) -> None:
    if label_column not in df.columns:
        raise ValueError(f"Label column not found in CSV: {label_column}")

    for row_idx, row in df.iterrows():
        seq_len = len(str(row["sequence"]))
        label_str = str(row[label_column])
        if len(label_str) != seq_len:
            raise ValueError(
                f"Row {row_idx}: length mismatch for '{label_column}' "
                f"(sequence={seq_len}, label={len(label_str)})"
            )

def parse_float_label(label_str: str) -> np.ndarray:
    values = []
    for ch in str(label_str):
        if ch == "1":
            values.append(1.0)
        elif ch == "0":
            values.append(0.0)
        else:
            raise ValueError(f"Invalid label character '{ch}', expected '0' or '1'.")
    return np.asarray(values, dtype=np.float64)


def compute_all_labels(df, label_columns: list[str]) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for feature in label_columns:
        parts = [
            parse_float_label(df.iloc[row_idx][feature])
            for row_idx in range(len(df))
        ]
        labels[feature] = np.concatenate(parts)
    return labels


def build_sequence_batches(
    seq_lengths: list[int],
    max_batch_tokens: int,
) -> list[list[int]]:
    if max_batch_tokens <= 0:
        raise ValueError(f"max_batch_tokens must be positive, got {max_batch_tokens}")

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_tokens = 0

    for row_idx, seq_len in enumerate(seq_lengths):
        if seq_len > max_batch_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([row_idx])
            continue

        if current_batch and current_tokens + seq_len > max_batch_tokens:
            batches.append(current_batch)
            current_batch = [row_idx]
            current_tokens = seq_len
        else:
            current_batch.append(row_idx)
            current_tokens += seq_len

    if current_batch:
        batches.append(current_batch)

    return batches


@torch.no_grad()
def encode_features(
    model: torch.nn.Module,
    embedding: np.ndarray,
    device: str,
) -> np.ndarray:
    x = torch.from_numpy(np.asarray(embedding, dtype=np.float32, copy=True)).to(device)
    return model.encode(x).detach().cpu().numpy()


@torch.no_grad()
def compute_all_activations(
    model: torch.nn.Module,
    embedding_npy: str,
    seq_lengths: list[int],
    device: str,
    max_batch_tokens: int = 128000,
) -> np.ndarray:
    total_tokens = int(sum(seq_lengths))
    dict_size = int(model.dict_size)
    f = np.empty((total_tokens, dict_size), dtype=np.float32)

    row_batches = build_sequence_batches(seq_lengths, max_batch_tokens)
    token_offsets = np.cumsum([0] + seq_lengths)
    all_embeddings = np.load(embedding_npy, mmap_mode="r")

    for batch_row_indices in tqdm(
        row_batches,
        desc="Encoding sequence batches",
    ):
        batch_start = int(token_offsets[batch_row_indices[0]])
        batch_end = int(token_offsets[batch_row_indices[-1] + 1])
        batch_embedding = np.asarray(all_embeddings[batch_start:batch_end], dtype=np.float32)
        batch_features = encode_features(model, batch_embedding, device=device)
        f[batch_start:batch_end] = batch_features

    return f


def compute_activation_data(
    *,
    model: torch.nn.Module,
    embedding_npy: str,
    seq_lengths: list[int],
    labels: dict[str, np.ndarray],
    device: str,
    max_batch_tokens: int,
    expected_num_tokens: int | None = None,
) -> ActivationData:
    """Encode embeddings through SAE in sequence batches."""
    f = compute_all_activations(
        model=model,
        embedding_npy=embedding_npy,
        seq_lengths=seq_lengths,
        device=device,
        max_batch_tokens=max_batch_tokens,
    )

    if expected_num_tokens is not None and f.shape[0] != expected_num_tokens:
        raise ValueError(
            f"Activation rows {f.shape[0]} != expected rows {expected_num_tokens}"
        )
    print(f"Computed activations on CPU: f.shape={f.shape}")

    return ActivationData(f=f, labels=labels, seq_lengths=seq_lengths)


def build_closed_form_prediction(
    f: np.ndarray,
    closed_form: ClosedFormFit,
) -> np.ndarray:
    if closed_form.neuron_ids:
        design = np.column_stack(
            [np.ones(len(f), dtype=np.float64), f[:, closed_form.neuron_ids]]
        )
        beta = np.concatenate(
            [[closed_form.b], np.asarray(closed_form.a, dtype=np.float64)]
        )
        return design @ beta
    return np.full(len(f), closed_form.b, dtype=np.float64)


def sweep_macro_f1_for_prediction(
    y: np.ndarray,
    y_pred_continuous: np.ndarray,
    seq_lengths: list[int],
    thresholds: np.ndarray,
) -> ThresholdSweepResult:
    sweep: list[dict[str, float]] = []
    best_threshold = float(thresholds[0])
    best_macro_f1 = -1.0

    for threshold in thresholds:
        macro_f1 = compute_mean_macro_f1(
            y,
            y_pred_continuous,
            seq_lengths,
            threshold=float(threshold),
        )
        sweep.append({"threshold": float(threshold), "macro_f1": float(macro_f1)})
        if macro_f1 > best_macro_f1:
            best_macro_f1 = float(macro_f1)
            best_threshold = float(threshold)

    return ThresholdSweepResult(
        best_threshold=best_threshold,
        best_macro_f1=best_macro_f1,
        sweep=sweep,
    )


def mean_macro_f1_all_neurons(
    y_true_bool: np.ndarray,
    pred_bool: np.ndarray,
    seq_lengths: list[int],
) -> np.ndarray:
    """Average per-sequence macro F1 across all neurons. pred_bool: [N, D]."""
    boundaries = np.cumsum([0] + seq_lengths)
    dict_size = pred_bool.shape[1]
    f1_sum = np.zeros(dict_size, dtype=np.float64)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        y_s = y_true_bool[start:end, None]
        p_s = pred_bool[start:end]
        tp = np.sum(y_s & p_s, axis=0)
        tn = np.sum(~y_s & ~p_s, axis=0)
        fp = np.sum(~y_s & p_s, axis=0)
        fn = np.sum(y_s & ~p_s, axis=0)

        f1_pos_den = 2 * tp + fp + fn
        f1_pos = np.divide(
            2 * tp,
            f1_pos_den,
            out=np.zeros_like(tp, dtype=np.float64),
            where=f1_pos_den > 0,
        )
        f1_neg_den = 2 * tn + fp + fn
        f1_neg = np.divide(
            2 * tn,
            f1_neg_den,
            out=np.zeros_like(tn, dtype=np.float64),
            where=f1_neg_den > 0,
        )
        f1_sum += 0.5 * (f1_pos + f1_neg)

    return f1_sum / len(seq_lengths)


def sweep_best_single_neuron(
    f: np.ndarray,
    y: np.ndarray,
    seq_lengths: list[int],
    thresholds: np.ndarray,
) -> SingleNeuronThresholdResult:
    y_true_bool = y >= 0.5
    best_neuron_id = 0
    best_threshold = float(thresholds[0])
    best_macro_f1 = -1.0

    for threshold in thresholds:
        pred_bool = f >= threshold
        macro_f1_per_neuron = mean_macro_f1_all_neurons(
            y_true_bool,
            pred_bool,
            seq_lengths,
        )
        neuron_idx = int(np.argmax(macro_f1_per_neuron))
        macro_f1 = float(macro_f1_per_neuron[neuron_idx])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_neuron_id = neuron_idx
            best_threshold = float(threshold)

    winner_sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        macro_f1 = compute_mean_macro_f1(
            y,
            f[:, best_neuron_id],
            seq_lengths,
            threshold=float(threshold),
        )
        winner_sweep.append(
            {"threshold": float(threshold), "macro_f1": float(macro_f1)}
        )

    return SingleNeuronThresholdResult(
        neuron_id=best_neuron_id,
        best_threshold=best_threshold,
        best_macro_f1=best_macro_f1,
        sweep=winner_sweep,
    )


def boosting_steps_to_records(boosting: NeuronBoostingResult) -> list[dict]:
    records: list[dict] = []
    for step, metrics in zip(boosting.steps, boosting.step_metrics):
        records.append(
            {
                "step": step.step,
                "neuron_id": step.neuron_id,
                "neuron_label": "const" if step.neuron_id == CONST_NEURON_ID else step.neuron_id,
                "a": float(step.a),
                "composite_r2": float(metrics.composite_r2),
            }
        )
    return records


def closed_form_to_dict(closed_form: ClosedFormFit) -> dict:
    return {
        "neuron_ids": [int(x) for x in closed_form.neuron_ids],
        "coefficients": [float(x) for x in closed_form.a],
        "intercept_b": float(closed_form.b),
        "composite_r2": float(closed_form.composite_r2),
    }


def single_neuron_to_dict(
    single_result: SingleNeuronThresholdResult | None,
) -> dict | None:
    if single_result is None:
        return None
    return {
        "neuron_id": int(single_result.neuron_id),
        "best_threshold": single_result.best_threshold,
        "best_macro_f1": single_result.best_macro_f1,
        "threshold_sweep": single_result.sweep,
    }


def evaluate_feature(
    f: np.ndarray,
    y: np.ndarray,
    feature: str,
    seq_lengths: list[int],
    params: FeatureEvalParams,
) -> FeatureEvaluationResult:
    threshold_grid = params.threshold_grid
    print(f"\n{'=' * 60}")
    print(f"Feature: {feature}")

    single_result: SingleNeuronThresholdResult | None = None
    if params.search_best_single_neuron:
        print(
            f"Threshold sweep on single neurons: {len(threshold_grid)} values "
            f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
        )
        single_result = sweep_best_single_neuron(
            f=f,
            y=y,
            seq_lengths=seq_lengths,
            thresholds=threshold_grid,
        )
        print(
            f"Best single neuron: id={single_result.neuron_id}, "
            f"threshold={single_result.best_threshold:.1f}, "
            f"macro F1={single_result.best_macro_f1:.6f}"
        )
    else:
        print("Skipped best single-neuron search.")

    print(f"Boosting: select {params.num_iterations} SAE neuron(s) by R²")

    iteration_checkpoint = None
    if params.output_json and params.save_iteration_checkpoints:
        iteration_checkpoint = IterationCheckpointConfig(
            output_json=params.output_json,
            feature=feature,
            config_dict=params.config_dict,
            f=f,
            y=y,
            seq_lengths=seq_lengths,
            threshold_grid=threshold_grid,
            f_shape=f.shape,
            best_single_neuron=single_result,
        )

    boosting_config = NeuronBoostingConfig(
        num_sae_neurons=params.num_iterations,
        seq_lengths=seq_lengths,
        fit_constant=params.fit_constant,
        solve_closed_form=False,
        compute_step_f1=False,
        verbose=True,
    )
    boosting = select_neurons_iteratively_from_config(
        f=f,
        y=y,
        config=boosting_config,
        iteration_checkpoint=iteration_checkpoint,
    )

    final_r2 = float(boosting.step_metrics[-1].composite_r2)
    num_const = 1 if params.fit_constant else 0
    print(
        f"\nBoosting finished ({len(boosting.steps)} steps: "
        f"{num_const} const + {len(boosting.steps) - num_const} SAE neurons), "
        f"composite R²={final_r2:.6f}"
    )

    closed_form = solve_closed_form_coefficients(
        f=f,
        y=y,
        steps=boosting.steps,
        seq_lengths=seq_lengths,
        threshold=0.5,
    )
    boosting.closed_form = closed_form
    y_pred = build_closed_form_prediction(f, closed_form)

    print_closed_form_refit_summary(closed_form)
    print(
        f"Threshold sweep on combination: {len(threshold_grid)} values "
        f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
    )
    combo_sweep = sweep_macro_f1_for_prediction(
        y=y,
        y_pred_continuous=y_pred,
        seq_lengths=seq_lengths,
        thresholds=threshold_grid,
    )
    print(
        f"Best combination: threshold={combo_sweep.best_threshold:.1f}, "
        f"macro F1={combo_sweep.best_macro_f1:.6f}"
    )

    return FeatureEvaluationResult(
        feature=feature,
        boosting=boosting,
        num_tokens=f.shape[0],
        combination=CombinationThresholdResult(
            closed_form=closed_form,
            y_pred=y_pred,
            sweep=combo_sweep,
        ),
        best_single_neuron=single_result,
    )


def build_results_json(
    config_dict: dict[str, Any],
    results: list[FeatureEvaluationResult],
    f_shape: tuple[int, ...],
    threshold_grid: np.ndarray | None = None,
) -> dict:
    grid = threshold_grid if threshold_grid is not None else THRESHOLD_GRID
    payload: dict = {
        "config": config_dict,
        "threshold_grid": [float(x) for x in grid],
        "f_shape": list(f_shape),
        "features": {},
    }

    for result in results:
        closed_form = result.combination.closed_form
        payload["features"][result.feature] = {
            "num_tokens": result.num_tokens,
            "boosting_steps": boosting_steps_to_records(result.boosting),
            "boosting_final_composite_r2": float(
                result.boosting.step_metrics[-1].composite_r2
            ),
            "combination": {
                **closed_form_to_dict(closed_form),
                "best_threshold": result.combination.sweep.best_threshold,
                "best_macro_f1": result.combination.sweep.best_macro_f1,
                "threshold_sweep": result.combination.sweep.sweep,
            },
            "best_single_neuron": single_neuron_to_dict(result.best_single_neuron),
        }

    return payload


def save_results_json(output_path: str, payload: dict) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved results JSON: {path}")


def run_feature_evaluation(
    data: ActivationData,
    params: FeatureEvalParams,
    config_dict: dict[str, Any] | None = None,
) -> list[FeatureEvaluationResult]:
    """Run single-neuron search, boosting, and threshold evaluation on activations."""
    threshold_grid = params.threshold_grid
    print("Boosting backend: CPU")
    print(
        f"Search best single neuron: "
        f"{'yes' if params.search_best_single_neuron else 'no'}"
    )
    print(f"SAE neurons to select: {params.num_iterations}")
    print(
        f"Threshold sweep grid: {len(threshold_grid)} values "
        f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
    )

    results: list[FeatureEvaluationResult] = []
    for feature, y in data.labels.items():
        if y.shape[0] != data.f.shape[0]:
            raise ValueError(
                f"Feature '{feature}': label len={y.shape[0]} != activation len={data.f.shape[0]}"
            )
        results.append(
            evaluate_feature(
                f=data.f,
                y=y,
                feature=feature,
                seq_lengths=data.seq_lengths,
                params=params,
            )
        )

    if params.output_json:
        payload = build_results_json(
            config_dict=config_dict or {},
            results=results,
            f_shape=data.f.shape,
            threshold_grid=threshold_grid,
        )
        save_results_json(params.output_json, payload)

    return results
