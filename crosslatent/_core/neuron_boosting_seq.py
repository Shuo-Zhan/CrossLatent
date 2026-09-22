"""
Streaming neuron boosting: on-the-fly ESM2 embedding + SAE encode per batch.

Avoids storing full embedding.npy or activation matrix f in memory. Boosting,
closed-form refit, single-neuron search, and threshold sweeps accumulate
batch-wise statistics with multiple passes over the data when needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm import tqdm

from .neuron_boosting import (
    CONST_NEURON_ID,
    THRESHOLD_GRID,
    BoostingStep,
    ClosedFormFit,
    CombinationThresholdResult,
    FeatureEvalParams,
    FeatureEvaluationResult,
    NeuronBoostingResult,
    SingleNeuronThresholdResult,
    StepMetrics,
    ThresholdSweepResult,
    boosting_steps_to_records,
    build_results_json,
    build_sequence_batches,
    closed_form_to_dict,
    compute_composite_r2,
    compute_mean_macro_f1,
    load_evaluation_dataframe,
    parse_float_label,
    save_results_json,
    single_neuron_to_dict,
    sweep_macro_f1_for_prediction,
    validate_row_labels,
    _fit_ones_a,
    _resolve_seq_lengths,
    _run_boosting_step,
    _selected_sae_neuron_ids,
    print_closed_form_fit,
    print_closed_form_refit_summary,
    iteration_output_path,
)

# At or below this token count, closed-form refit scans all batches (still fits y_boost).
DEFAULT_CLOSED_FORM_FULL_PASS_TOKEN_THRESHOLD = 500_000


@dataclass
class StreamingContext:
    """Everything needed to embed + encode + boost without full f in memory."""

    df: Any
    sequences: list[str]
    seq_lengths: list[int]
    row_batches: list[list[int]]
    token_offsets: np.ndarray
    y: np.ndarray
    embedder: Any
    sae: torch.nn.Module
    device: torch.device
    plm_layer: int
    dict_size: int
    embed_batch_size: int = 32
    num_sequences: int = 0
    num_tokens: int = 0


@dataclass
class StreamingFeatureEvalParams(FeatureEvalParams):
    """Streaming eval params; extends FeatureEvalParams with closed-form policy."""

    closed_form_full_pass_token_threshold: int = (
        DEFAULT_CLOSED_FORM_FULL_PASS_TOKEN_THRESHOLD
    )
    resume_from: str | None = None


def compute_all_labels_from_df(df, label_columns: list[str]) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for feature in label_columns:
        parts = [
            parse_float_label(df.iloc[row_idx][feature])
            for row_idx in range(len(df))
        ]
        labels[feature] = np.concatenate(parts)
    return labels


def _seq_lengths_for_rows(seq_lengths: list[int], row_indices: list[int]) -> list[int]:
    return [seq_lengths[i] for i in row_indices]


@torch.no_grad()
def _encode_batch(
    ctx: StreamingContext,
    batch_sequences: list[str],
) -> np.ndarray:
    embeddings = ctx.embedder.extract_embeddings(
        sequences=batch_sequences,
        layer=ctx.plm_layer,
        batch_size=min(ctx.embed_batch_size, len(batch_sequences)),
    )
    if isinstance(embeddings, torch.Tensor):
        emb = embeddings.float()
    else:
        emb = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    features = ctx.sae.encode(emb.to(ctx.device))
    result = features.detach().cpu().numpy().astype(np.float64)
    del emb, features
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def iter_f_y_batches(
    ctx: StreamingContext,
    *,
    desc: str = "Streaming encode",
    leave: bool = True,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (batch_start, batch_end, f_batch, y_batch) without storing full f."""
    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc=desc,
        leave=leave,
    ):
        batch_start = int(ctx.token_offsets[batch_row_indices[0]])
        batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
        batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
        y_batch = ctx.y[batch_start:batch_end]
        f_batch = _encode_batch(ctx, batch_sequences)
        yield batch_start, batch_end, f_batch, y_batch


def _apply_deflations(
    f_batch: np.ndarray,
    batch_start: int,
    batch_end: int,
    directions: list[np.ndarray],
    column_dots: list[np.ndarray],
) -> np.ndarray:
    """Apply sequential deflations using precomputed global column dots."""
    if not directions:
        return f_batch

    f_def = f_batch.copy()
    for direction, dots in zip(directions, column_dots):
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        u_norm = direction / norm
        u_batch = u_norm[batch_start:batch_end]
        f_def -= np.outer(u_batch, dots)
    return f_def


def _compute_column_dots_for_direction(
    ctx: StreamingContext,
    direction: np.ndarray,
    prior_directions: list[np.ndarray],
    prior_dots: list[np.ndarray],
) -> np.ndarray:
    """One full pass: dots[j] = f_def[:, j] · u for the next deflation direction."""
    dots = np.zeros(ctx.dict_size, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return dots

    u_norm = direction / norm
    for batch_start, batch_end, f_batch, _ in iter_f_y_batches(
        ctx,
        desc="Deflation dot pass",
        leave=False,
    ):
        f_def = _apply_deflations(
            f_batch,
            batch_start,
            batch_end,
            prior_directions,
            prior_dots,
        )
        u_batch = u_norm[batch_start:batch_end]
        dots += f_def.T @ u_batch
    return dots


def _ensure_column_dots(
    ctx: StreamingContext,
    directions: list[np.ndarray],
    column_dots: list[np.ndarray],
) -> None:
    """Extend in-memory column_dots for any directions not yet cached."""
    if len(column_dots) > len(directions):
        raise ValueError(
            f"column_dots length {len(column_dots)} > directions length {len(directions)}"
        )
    for step_idx in range(len(column_dots), len(directions)):
        dots = _compute_column_dots_for_direction(
            ctx,
            directions[step_idx],
            directions[:step_idx],
            column_dots[:step_idx],
        )
        column_dots.append(dots)


def _accumulate_deflated_stats(
    ctx: StreamingContext,
    y_residual: np.ndarray,
    directions: list[np.ndarray],
    column_dots: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sum_xy, sum_x2 on deflated activations for y = a*x (no intercept).

    Reuses column_dots cached in memory across boosting steps; only scans data for
    directions missing from the cache, then one stats pass.
    """
    _ensure_column_dots(ctx, directions, column_dots)

    sum_xy = np.zeros(ctx.dict_size, dtype=np.float64)
    sum_x2 = np.zeros(ctx.dict_size, dtype=np.float64)
    for batch_start, batch_end, f_batch, _ in iter_f_y_batches(
        ctx,
        desc="Deflated stats pass",
        leave=False,
    ):
        f_def = _apply_deflations(
            f_batch,
            batch_start,
            batch_end,
            directions,
            column_dots,
        )
        y_batch = y_residual[batch_start:batch_end]
        sum_xy += f_def.T @ y_batch
        sum_x2 += np.sum(f_def * f_def, axis=0)

    return sum_xy, sum_x2


def _best_neuron_from_stats(
    sum_xy: np.ndarray,
    sum_x2: np.ndarray,
    y_residual: np.ndarray,
    exclude_neuron_ids: set[int] | None = None,
) -> tuple[int, float, float]:
    """Mirror neuron_boosting.compute_optimal_r2 using accumulated sums."""
    a = np.divide(
        sum_xy,
        sum_x2,
        out=np.zeros_like(sum_xy, dtype=np.float64),
        where=sum_x2 > 0,
    )
    sum_y2 = float(np.sum(y_residual * y_residual))
    ss_res = sum_y2 - 2.0 * a * sum_xy + a * a * sum_x2
    ss_tot = float(np.sum((y_residual - np.mean(y_residual)) ** 2))
    if ss_tot <= 0:
        return 0, 0.0, 0.0

    r2 = 1.0 - ss_res / ss_tot
    if exclude_neuron_ids:
        for neuron_id in exclude_neuron_ids:
            if 0 <= neuron_id < len(r2):
                r2[neuron_id] = -np.inf
    best_neuron_id = int(np.argmax(r2))
    return best_neuron_id, float(a[best_neuron_id]), float(r2[best_neuron_id])


def single_neuron_from_dict(data: dict[str, Any] | None) -> SingleNeuronThresholdResult | None:
    if data is None:
        return None
    return SingleNeuronThresholdResult(
        neuron_id=int(data["neuron_id"]),
        best_threshold=float(data["best_threshold"]),
        best_macro_f1=float(data["best_macro_f1"]),
        sweep=list(data.get("threshold_sweep", [])),
    )


def load_streaming_checkpoint(
    checkpoint_path: str,
    feature: str | None = None,
) -> tuple[list[BoostingStep], SingleNeuronThresholdResult | None, dict[str, Any], str]:
    """Load boosting steps and optional single-neuron sweep from an iteration checkpoint."""
    path = Path(checkpoint_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features")
    if not features:
        raise ValueError(f"No features block in checkpoint: {path}")

    if feature is None:
        if len(features) != 1:
            raise ValueError(
                f"Checkpoint has {len(features)} features; pass feature name explicitly."
            )
        feature = next(iter(features))

    feature = str(feature)
    if feature not in features:
        raise KeyError(f"Feature {feature!r} not found in checkpoint {path}")

    records = features[feature]["boosting_steps"]
    initial_steps = [
        BoostingStep(
            step=int(record["step"]),
            neuron_id=int(record["neuron_id"]),
            a=float(record["a"]),
        )
        for record in records
    ]
    single_result = single_neuron_from_dict(features[feature].get("best_single_neuron"))
    return initial_steps, single_result, payload, feature


def _replay_initial_boosting_steps(
    ctx: StreamingContext,
    y: np.ndarray,
    initial_steps: list[BoostingStep],
    *,
    prediction_threshold: float = 0.5,
    verbose: bool = True,
    compute_step_f1: bool = False,
) -> tuple[
    list[BoostingStep],
    list[StepMetrics],
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    """Rebuild y_pred/y_residual and deflation directions from saved boosting steps."""
    y_original = np.asarray(y, dtype=np.float64)
    seq_lengths = _resolve_seq_lengths(y_original, ctx.seq_lengths)
    y_residual = y_original.copy()
    y_pred = np.zeros_like(y_original)
    directions: list[np.ndarray] = []
    cached_column_dots: list[np.ndarray] = []
    steps: list[BoostingStep] = []
    step_metrics: list[StepMetrics] = []

    if verbose:
        print(
            f"  Replaying {len(initial_steps)} saved boosting step(s) "
            f"({len(_selected_sae_neuron_ids(initial_steps))} SAE neuron(s))..."
        )

    for boost_step in initial_steps:
        if boost_step.neuron_id == CONST_NEURON_ID:
            delta = np.full_like(y_residual, boost_step.a)
            direction = np.ones(ctx.num_tokens, dtype=np.float64)
        else:
            _ensure_column_dots(ctx, directions, cached_column_dots)
            direction = _extract_deflated_neuron_vector(
                ctx,
                boost_step.neuron_id,
                directions,
                cached_column_dots,
            )
            delta = boost_step.a * direction

        _run_boosting_step(
            step=boost_step.step,
            neuron_id=boost_step.neuron_id,
            a=boost_step.a,
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

        directions.append(direction.copy())
        _ensure_column_dots(ctx, directions, cached_column_dots)

    return steps, step_metrics, y_pred, y_residual, directions, cached_column_dots


def _extract_deflated_neuron_vector(
    ctx: StreamingContext,
    neuron_id: int,
    directions: list[np.ndarray],
    column_dots: list[np.ndarray],
) -> np.ndarray:
    """Full-length deflated activation vector for the selected neuron."""
    vector = np.empty(ctx.num_tokens, dtype=np.float64)
    for batch_start, batch_end, f_batch, _ in iter_f_y_batches(
        ctx,
        desc="Extract neuron vector",
        leave=False,
    ):
        f_def = _apply_deflations(
            f_batch,
            batch_start,
            batch_end,
            directions,
            column_dots,
        )
        vector[batch_start:batch_end] = f_def[:, neuron_id]
    return vector


def select_neurons_streaming(
    ctx: StreamingContext,
    y: np.ndarray,
    num_sae_neurons: int,
    *,
    fit_constant: bool = True,
    solve_closed_form: bool = True,
    compute_step_f1: bool = False,
    prediction_threshold: float = 0.5,
    verbose: bool = True,
    iteration_checkpoint_cb=None,
    initial_steps: list[BoostingStep] | None = None,
) -> NeuronBoostingResult:
    """Greedy boosting with streaming SAE activations."""
    y_original = np.asarray(y, dtype=np.float64)
    seq_lengths = _resolve_seq_lengths(y_original, ctx.seq_lengths)
    exclude_neuron_ids: set[int] | None = None

    def _maybe_checkpoint() -> None:
        if iteration_checkpoint_cb is not None:
            iteration_checkpoint_cb(steps, step_metrics, y_pred, y_residual)

    if initial_steps:
        (
            steps,
            step_metrics,
            y_pred,
            y_residual,
            directions,
            cached_column_dots,
        ) = _replay_initial_boosting_steps(
            ctx,
            y_original,
            initial_steps,
            prediction_threshold=prediction_threshold,
            verbose=verbose,
            compute_step_f1=compute_step_f1,
        )
        step_num = initial_steps[-1].step
        num_sae_done = len(_selected_sae_neuron_ids(steps))
        num_sae_remaining = num_sae_neurons - num_sae_done
        exclude_neuron_ids = set(_selected_sae_neuron_ids(steps))
        if verbose and num_sae_remaining <= 0:
            print(
                f"  Checkpoint already has {num_sae_done} SAE neuron(s) "
                f"(target {num_sae_neurons}); skipping greedy selection."
            )
    else:
        y_residual = y_original.copy()
        y_pred = np.zeros_like(y_original)
        directions = []
        cached_column_dots = []
        steps = []
        step_metrics = []
        step_num = 0
        num_sae_remaining = num_sae_neurons

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
            directions.append(np.ones(ctx.num_tokens, dtype=np.float64))
            _ensure_column_dots(ctx, directions, cached_column_dots)
            _maybe_checkpoint()

    for _ in range(num_sae_remaining):
        step_num += 1
        if verbose:
            print(f"  Boosting step {step_num}: scanning activations ({len(directions)} deflation directions)...")

        sum_xy, sum_x2 = _accumulate_deflated_stats(
            ctx,
            y_residual,
            directions,
            cached_column_dots,
        )
        neuron_id, a, _best_r2 = _best_neuron_from_stats(
            sum_xy,
            sum_x2,
            y_residual,
            exclude_neuron_ids=exclude_neuron_ids,
        )
        if exclude_neuron_ids is not None:
            exclude_neuron_ids.add(neuron_id)
        x = _extract_deflated_neuron_vector(
            ctx,
            neuron_id,
            directions,
            cached_column_dots,
        )
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
        directions.append(x.copy())
        _ensure_column_dots(ctx, directions, cached_column_dots)
        _maybe_checkpoint()

    closed_form = None
    if solve_closed_form:
        closed_form = solve_closed_form_streaming(
            ctx,
            y_labels=y_original,
            y_boost=y_pred,
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
    )


def _pick_closed_form_row_batch(
    ctx: StreamingContext,
    min_tokens: int,
) -> list[int]:
    """Pick the streaming batch with the most tokens, requiring at least min_tokens."""
    best_indices = ctx.row_batches[0]
    best_len = 0
    for batch_row_indices in ctx.row_batches:
        batch_start = int(ctx.token_offsets[batch_row_indices[0]])
        batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
        batch_len = batch_end - batch_start
        if batch_len >= min_tokens and batch_len > best_len:
            best_indices = batch_row_indices
            best_len = batch_len
    return best_indices


def _encode_row_batch(
    ctx: StreamingContext,
    batch_row_indices: list[int],
) -> tuple[int, int, np.ndarray]:
    batch_start = int(ctx.token_offsets[batch_row_indices[0]])
    batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
    batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
    f_batch = _encode_batch(ctx, batch_sequences)
    return batch_start, batch_end, f_batch


def _closed_form_metrics(
    y_labels: np.ndarray,
    y_continuous: np.ndarray,
    seq_lengths: list[int],
    threshold: float,
) -> tuple[float, float]:
    composite_r2 = compute_composite_r2(y_labels, y_continuous)
    macro_f1 = compute_mean_macro_f1(
        y_labels,
        y_continuous,
        seq_lengths,
        threshold=threshold,
    )
    return composite_r2, macro_f1


def solve_closed_form_streaming(
    ctx: StreamingContext,
    y_labels: np.ndarray,
    y_boost: np.ndarray,
    steps: list[BoostingStep],
    *,
    seq_lengths: list[int] | None = None,
    threshold: float = 0.5,
    full_pass_token_threshold: int = DEFAULT_CLOSED_FORM_FULL_PASS_TOKEN_THRESHOLD,
    verbose: bool = True,
) -> ClosedFormFit:
    """
    Reparameterize boosting prediction y_boost as b + sum(a_i f_i) on original activations.

    Fits y_boost (not labels). When num_tokens <= full_pass_token_threshold, accumulates
    normal equations over all batches; otherwise uses a single streaming batch.
    """
    y_labels = np.asarray(y_labels, dtype=np.float64)
    y_boost = np.asarray(y_boost, dtype=np.float64)
    if y_labels.shape != y_boost.shape:
        raise ValueError(
            f"y_labels shape {y_labels.shape} != y_boost shape {y_boost.shape}"
        )
    seq_lengths = _resolve_seq_lengths(y_labels, seq_lengths or ctx.seq_lengths)
    neuron_ids = _selected_sae_neuron_ids(steps)
    k = len(neuron_ids)

    if k == 0:
        b = float(np.mean(y_boost))
        composite_r2, macro_f1 = _closed_form_metrics(
            y_labels, y_boost, seq_lengths, threshold
        )
        return ClosedFormFit(
            neuron_ids=[],
            a=np.zeros(0, dtype=np.float64),
            b=b,
            composite_r2=composite_r2,
            macro_f1=macro_f1,
        )

    use_full_pass = ctx.num_tokens <= full_pass_token_threshold
    if use_full_pass:
        if verbose:
            print(
                f"Closed-form refit: full pass ({ctx.num_tokens} tokens <= "
                f"{full_pass_token_threshold}), target=y_boost"
            )
        xtx = np.zeros((k + 1, k + 1), dtype=np.float64)
        xty = np.zeros(k + 1, dtype=np.float64)
        for batch_start, batch_end, f_batch, _ in iter_f_y_batches(
            ctx,
            desc="Closed-form refit (full)",
            leave=False,
        ):
            y_target = y_boost[batch_start:batch_end]
            design = np.column_stack(
                [
                    np.ones(batch_end - batch_start, dtype=np.float64),
                    f_batch[:, neuron_ids],
                ]
            )
            xtx += design.T @ design
            xty += design.T @ y_target
        beta = np.linalg.solve(xtx, xty)
    else:
        min_tokens = k + 1
        batch_row_indices = _pick_closed_form_row_batch(ctx, min_tokens=min_tokens)
        batch_start, batch_end, f_batch = _encode_row_batch(ctx, batch_row_indices)
        y_target = y_boost[batch_start:batch_end]
        design = np.column_stack(
            [
                np.ones(batch_end - batch_start, dtype=np.float64),
                f_batch[:, neuron_ids],
            ]
        )
        if verbose:
            print(
                f"Closed-form refit: single batch "
                f"(tokens {batch_end - batch_start}, rows>={min_tokens}), target=y_boost"
            )
        beta, _, _, _ = np.linalg.lstsq(design, y_target, rcond=None)

    b = float(beta[0])
    a = np.asarray(beta[1:], dtype=np.float64)
    composite_r2, macro_f1 = _closed_form_metrics(
        y_labels, y_boost, seq_lengths, threshold
    )
    return ClosedFormFit(
        neuron_ids=neuron_ids,
        a=a,
        b=b,
        composite_r2=composite_r2,
        macro_f1=macro_f1,
    )


def sweep_macro_f1_on_continuous(
    y_labels: np.ndarray,
    y_continuous: np.ndarray,
    seq_lengths: list[int],
    thresholds: np.ndarray,
) -> ThresholdSweepResult:
    """Threshold sweep on a precomputed continuous prediction (no re-encode)."""
    return sweep_macro_f1_for_prediction(
        y=y_labels,
        y_pred_continuous=y_continuous,
        seq_lengths=seq_lengths,
        thresholds=thresholds,
    )


def _macro_f1_per_neuron_for_batch(
    y_true_bool: np.ndarray,
    pred_bool: np.ndarray,
    seq_lengths_in_batch: list[int],
) -> np.ndarray:
    """Macro F1 per neuron averaged over sequences in this batch."""
    if not seq_lengths_in_batch:
        return np.zeros(pred_bool.shape[1], dtype=np.float64)

    f1_sum = np.zeros(pred_bool.shape[1], dtype=np.float64)
    offset = 0
    for seq_len in seq_lengths_in_batch:
        y_s = y_true_bool[offset : offset + seq_len, None]
        p_s = pred_bool[offset : offset + seq_len]
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
        offset += seq_len

    return f1_sum


def sweep_best_single_neuron_streaming(
    ctx: StreamingContext,
    y: np.ndarray,
    seq_lengths: list[int],
    thresholds: np.ndarray,
) -> SingleNeuronThresholdResult:
    """Streaming single-neuron search + winner threshold sweep."""
    y_true_bool = y >= 0.5
    best_neuron_id = 0
    best_threshold = float(thresholds[0])
    best_macro_f1 = -1.0
    num_sequences = len(seq_lengths)
    num_thresholds = len(thresholds)
    f1_accum = np.zeros((num_thresholds, ctx.dict_size), dtype=np.float64)

    # Encode once per streaming batch; thresholds only affect CPU comparisons.
    for batch_row_indices in tqdm(
        ctx.row_batches,
        desc="Single-neuron encode",
    ):
        batch_start = int(ctx.token_offsets[batch_row_indices[0]])
        batch_end = int(ctx.token_offsets[batch_row_indices[-1] + 1])
        batch_sequences = [ctx.sequences[i] for i in batch_row_indices]
        f_batch = _encode_batch(ctx, batch_sequences)
        batch_seq_lengths = _seq_lengths_for_rows(ctx.seq_lengths, batch_row_indices)
        y_batch_bool = y_true_bool[batch_start:batch_end]
        for t_idx, threshold in enumerate(thresholds):
            pred_bool = f_batch >= threshold
            f1_accum[t_idx] += _macro_f1_per_neuron_for_batch(
                y_batch_bool,
                pred_bool,
                batch_seq_lengths,
            )

    for t_idx, threshold in enumerate(thresholds):
        macro_f1_per_neuron = f1_accum[t_idx] / num_sequences
        neuron_idx = int(np.argmax(macro_f1_per_neuron))
        macro_f1 = float(macro_f1_per_neuron[neuron_idx])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_neuron_id = neuron_idx
            best_threshold = float(threshold)

    winner_sweep: list[dict[str, float]] = []
    activation_vector = np.empty(ctx.num_tokens, dtype=np.float64)
    for batch_start, batch_end, f_batch, _ in iter_f_y_batches(
        ctx,
        desc="Single-neuron winner activations",
        leave=False,
    ):
        activation_vector[batch_start:batch_end] = f_batch[:, best_neuron_id]

    for threshold in thresholds:
        macro_f1 = compute_mean_macro_f1(
            y,
            activation_vector,
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


def _save_streaming_iteration_checkpoint(
    ctx: StreamingContext,
    *,
    output_json: str,
    feature: str,
    config_dict: dict[str, Any],
    threshold_grid: np.ndarray,
    steps: list[BoostingStep],
    step_metrics: list[StepMetrics],
    y_pred: np.ndarray,
    y_residual: np.ndarray,
    y: np.ndarray,
    best_single_neuron: SingleNeuronThresholdResult | None,
    full_pass_token_threshold: int,
    verbose: bool = True,
) -> ClosedFormFit:
    """Streaming iteration checkpoint (closed-form refit + threshold sweep)."""
    closed_form = solve_closed_form_streaming(
        ctx,
        y_labels=y,
        y_boost=y_pred,
        steps=steps,
        seq_lengths=ctx.seq_lengths,
        threshold=0.5,
        full_pass_token_threshold=full_pass_token_threshold,
        verbose=verbose,
    )
    if verbose:
        print_closed_form_refit_summary(closed_form)

    combo_sweep = sweep_macro_f1_on_continuous(
        y_labels=y,
        y_continuous=y_pred,
        seq_lengths=ctx.seq_lengths,
        thresholds=threshold_grid,
    )
    iteration = len(_selected_sae_neuron_ids(steps))
    boosting = NeuronBoostingResult(
        steps=list(steps),
        step_metrics=list(step_metrics),
        y_pred=y_pred,
        y_residual=y_residual,
        closed_form=closed_form,
    )
    payload = {
        "config": config_dict,
        "iteration": iteration,
        "num_boosting_steps": len(steps),
        "threshold_grid": [float(x) for x in threshold_grid],
        "f_shape": [ctx.num_tokens, ctx.dict_size],
        "streaming": True,
        "features": {
            feature: {
                "num_tokens": ctx.num_tokens,
                "boosting_steps": boosting_steps_to_records(boosting),
                "boosting_final_composite_r2": float(step_metrics[-1].composite_r2),
                "combination": {
                    **closed_form_to_dict(closed_form),
                    "best_threshold": combo_sweep.best_threshold,
                    "best_macro_f1": combo_sweep.best_macro_f1,
                    "threshold_sweep": combo_sweep.sweep,
                },
                "best_single_neuron": single_neuron_to_dict(best_single_neuron),
            }
        },
    }
    output_path = iteration_output_path(output_json, iteration)
    save_results_json(str(output_path), payload)
    if verbose:
        print(f"Saved iteration checkpoint: {output_path}")
    return closed_form


def build_streaming_context(
    *,
    df: Any,
    y: np.ndarray,
    embedder: Any,
    sae: torch.nn.Module,
    device: torch.device,
    plm_layer: int,
    max_batch_tokens: int,
    embed_batch_size: int = 32,
) -> StreamingContext:
    seq_lengths = df["sequence"].astype(str).str.len().tolist()
    sequences = df["sequence"].astype(str).tolist()
    row_batches = build_sequence_batches(seq_lengths, max_batch_tokens)
    token_offsets = np.cumsum([0] + seq_lengths)
    dict_size = int(sae.dict_size)

    return StreamingContext(
        df=df,
        sequences=sequences,
        seq_lengths=seq_lengths,
        row_batches=row_batches,
        token_offsets=token_offsets,
        y=np.asarray(y, dtype=np.float64),
        embedder=embedder,
        sae=sae,
        device=device,
        plm_layer=plm_layer,
        dict_size=dict_size,
        embed_batch_size=embed_batch_size,
        num_sequences=len(seq_lengths),
        num_tokens=int(sum(seq_lengths)),
    )


def evaluate_feature_streaming(
    ctx: StreamingContext,
    feature: str,
    params: StreamingFeatureEvalParams,
) -> FeatureEvaluationResult:
    """Streaming analogue of neuron_boosting.evaluate_feature."""
    threshold_grid = params.threshold_grid
    y = ctx.y
    print(f"\n{'=' * 60}")
    print(f"Feature: {feature}")

    initial_steps: list[BoostingStep] | None = None
    single_result: SingleNeuronThresholdResult | None = None
    if params.resume_from:
        initial_steps, single_result, checkpoint_payload, _ = load_streaming_checkpoint(
            params.resume_from,
            feature=feature,
        )
        completed_sae = len(_selected_sae_neuron_ids(initial_steps))
        print(
            f"Resume checkpoint: {params.resume_from} "
            f"({completed_sae}/{params.num_iterations} SAE neurons selected)"
        )
        ckpt_config = checkpoint_payload.get("config", {})
        for key in ("csv_path", "label_column", "num_iterations"):
            if key in ckpt_config and key in params.config_dict:
                if str(ckpt_config[key]) != str(params.config_dict[key]):
                    print(
                        f"Warning: checkpoint config {key}={ckpt_config[key]!r} "
                        f"!= current {params.config_dict[key]!r}"
                    )

    if params.search_best_single_neuron and single_result is None:
        print(
            f"Threshold sweep on single neurons (streaming): {len(threshold_grid)} values "
            f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
        )
        single_result = sweep_best_single_neuron_streaming(
            ctx,
            y=y,
            seq_lengths=ctx.seq_lengths,
            thresholds=threshold_grid,
        )
        print(
            f"Best single neuron: id={single_result.neuron_id}, "
            f"threshold={single_result.best_threshold:.1f}, "
            f"macro F1={single_result.best_macro_f1:.6f}"
        )
    elif single_result is not None:
        print(
            "Skipped best single-neuron search (loaded from resume checkpoint): "
            f"id={single_result.neuron_id}, threshold={single_result.best_threshold:.1f}, "
            f"macro F1={single_result.best_macro_f1:.6f}"
        )
    else:
        print("Skipped best single-neuron search.")

    remaining = params.num_iterations
    if initial_steps is not None:
        remaining = max(0, params.num_iterations - len(_selected_sae_neuron_ids(initial_steps)))
    print(
        f"Boosting (streaming): select {params.num_iterations} SAE neuron(s) by R² "
        f"({remaining} remaining)"
    )

    iteration_checkpoint_cb = None
    if params.output_json and params.save_iteration_checkpoints:
        captured_single = single_result

        def iteration_checkpoint_cb(steps, step_metrics, y_pred, y_residual) -> None:
            _save_streaming_iteration_checkpoint(
                ctx,
                output_json=params.output_json,
                feature=feature,
                config_dict=params.config_dict,
                threshold_grid=threshold_grid,
                steps=steps,
                step_metrics=step_metrics,
                y_pred=y_pred,
                y_residual=y_residual,
                y=y,
                best_single_neuron=captured_single,
                full_pass_token_threshold=params.closed_form_full_pass_token_threshold,
            )

    boosting = select_neurons_streaming(
        ctx,
        y=y,
        num_sae_neurons=params.num_iterations,
        fit_constant=params.fit_constant and initial_steps is None,
        solve_closed_form=False,
        compute_step_f1=False,
        verbose=True,
        iteration_checkpoint_cb=iteration_checkpoint_cb,
        initial_steps=initial_steps,
    )

    final_r2 = float(boosting.step_metrics[-1].composite_r2)
    num_const = 1 if params.fit_constant else 0
    print(
        f"\nBoosting finished ({len(boosting.steps)} steps: "
        f"{num_const} const + {len(boosting.steps) - num_const} SAE neurons), "
        f"composite R²={final_r2:.6f}"
    )

    closed_form = solve_closed_form_streaming(
        ctx,
        y_labels=y,
        y_boost=boosting.y_pred,
        steps=boosting.steps,
        seq_lengths=ctx.seq_lengths,
        threshold=0.5,
        full_pass_token_threshold=params.closed_form_full_pass_token_threshold,
    )
    boosting.closed_form = closed_form
    y_pred_combo = boosting.y_pred.copy()

    print_closed_form_refit_summary(closed_form)
    print(
        f"Threshold sweep on combination (streaming): {len(threshold_grid)} values "
        f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
    )
    combo_sweep = sweep_macro_f1_on_continuous(
        y_labels=y,
        y_continuous=y_pred_combo,
        seq_lengths=ctx.seq_lengths,
        thresholds=threshold_grid,
    )
    print(
        f"Best combination: threshold={combo_sweep.best_threshold:.1f}, "
        f"macro F1={combo_sweep.best_macro_f1:.6f}"
    )

    return FeatureEvaluationResult(
        feature=feature,
        boosting=boosting,
        num_tokens=ctx.num_tokens,
        combination=CombinationThresholdResult(
            closed_form=closed_form,
            y_pred=y_pred_combo,
            sweep=combo_sweep,
        ),
        best_single_neuron=single_result,
    )


def run_streaming_feature_evaluation(
    ctx: StreamingContext,
    feature: str,
    params: StreamingFeatureEvalParams,
    config_dict: dict[str, Any] | None = None,
) -> list[FeatureEvaluationResult]:
    """Run streaming boosting pipeline for one label column."""
    threshold_grid = params.threshold_grid
    print("Boosting backend: streaming (on-the-fly embed + SAE encode)")
    print(
        f"Search best single neuron: "
        f"{'yes' if params.search_best_single_neuron else 'no'}"
    )
    print(f"SAE neurons to select: {params.num_iterations}")
    print(
        f"Total sequences: {ctx.num_sequences}, total tokens: {ctx.num_tokens}, "
        f"dict_size: {ctx.dict_size}"
    )
    print(
        f"Threshold sweep grid: {len(threshold_grid)} values "
        f"from {threshold_grid[0]:.1f} to {threshold_grid[-1]:.1f} (step 0.1)"
    )

    feature = str(feature)
    result = evaluate_feature_streaming(
        ctx,
        feature=feature,
        params=params,
    )
    results = [result]

    if params.output_json:
        payload = build_results_json(
            config_dict=config_dict or params.config_dict,
            results=results,
            f_shape=(ctx.num_tokens, ctx.dict_size),
            threshold_grid=threshold_grid,
        )
        payload["streaming"] = True
        save_results_json(params.output_json, payload)

    return results

