"""Train-only selection, validation-only tuning and frozen evaluation on SAE arrays."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._core.neuron_boosting import (
    compute_composite_r2,
    compute_mean_macro_f1,
    select_neurons_iteratively,
    sweep_best_single_neuron,
)


@dataclass
class ActivationDataset:
    """Original SAE columns in a fixed order, with concatenated token labels.

    dictionary_id must identify the checkpoint AND encoding protocol. lengths
    delimit sequences; each sequence receives equal weight in binary macro F1.
    """

    activations: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    dictionary_id: str
    sequence_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.activations = np.asarray(self.activations)
        self.labels = np.asarray(self.labels, dtype=np.float64)
        lengths = np.asarray(self.lengths)
        if self.activations.ndim != 2 or min(self.activations.shape) < 1:
            raise ValueError("activations must be a nonempty [tokens, features] matrix")
        if self.labels.shape != (len(self.activations),):
            raise ValueError("labels must have one value per activation row")
        if lengths.ndim != 1 or not len(lengths) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError("lengths must be a nonempty integer vector")
        self.lengths = lengths.astype(np.int64)
        if np.any(self.lengths <= 0) or self.lengths.sum() != len(self.labels):
            raise ValueError("positive sequence lengths must sum to the number of tokens")
        if not np.isfinite(self.labels).all() or not np.isfinite(self.activations).all():
            raise ValueError("labels and activations must be finite")
        if not isinstance(self.dictionary_id, str) or not self.dictionary_id.strip():
            raise ValueError("dictionary_id is required (checkpoint + encoding protocol)")
        if self.sequence_ids is not None:
            self.sequence_ids = np.asarray(self.sequence_ids).astype(str)
            if self.sequence_ids.shape != self.lengths.shape:
                raise ValueError("sequence_ids must contain one identifier per sequence")
            if len(set(self.sequence_ids)) != len(self.sequence_ids):
                raise ValueError("sequence_ids must be unique within a split")

    @classmethod
    def load(cls, path: str | Path) -> ActivationDataset:
        """Load NPZ without pickle. Keys: activations, labels, lengths, dictionary_id."""
        with np.load(path, allow_pickle=False) as data:
            return cls(data["activations"], data["labels"], data["lengths"],
                       str(data["dictionary_id"].item()),
                       data["sequence_ids"] if "sequence_ids" in data else None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = dict(activations=self.activations, labels=self.labels,
                      lengths=self.lengths, dictionary_id=np.array(self.dictionary_id))
        if self.sequence_ids is not None:
            values["sequence_ids"] = self.sequence_ids
        with path.open("wb") as handle:
            np.savez(handle, **values)


@dataclass
class LinearReadout:
    """Frozen original-coordinate coefficients; no prediction-time fitting."""

    latent_ids: list[int]
    coefficients: list[float]
    intercept: float
    n_features: int
    dictionary_id: str
    task: str = "binary"
    threshold: float | None = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in {"binary", "continuous"}:
            raise ValueError("task must be binary or continuous")
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if len(self.latent_ids) != len(self.coefficients) or len(set(self.latent_ids)) != len(self.latent_ids):
            raise ValueError("latent IDs must be unique and match the coefficients")
        if any(not isinstance(i, int) or i < 0 or i >= self.n_features for i in self.latent_ids):
            raise ValueError("latent ID outside dictionary")
        if not np.isfinite([self.intercept, *self.coefficients]).all():
            raise ValueError("readout parameters must be finite")
        if self.task == "binary" and (self.threshold is None or not np.isfinite(self.threshold)):
            raise ValueError("binary readout requires a finite frozen threshold")
        if not self.dictionary_id:
            raise ValueError("dictionary_id is required")

    def predict(self, activations: np.ndarray) -> np.ndarray:
        values = np.asarray(activations)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError("activation width does not match the frozen dictionary")
        selected = np.asarray(values[:, self.latent_ids], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError("selected activations must be finite")
        return self.intercept + selected @ np.asarray(self.coefficients, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "latent-readout/v1", "latent_ids": self.latent_ids,
                "coefficients": self.coefficients, "intercept": self.intercept,
                "n_features": self.n_features, "dictionary_id": self.dictionary_id,
                "task": self.task, "threshold": self.threshold, "metadata": self.metadata}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> LinearReadout:
        data = json.loads(Path(path).read_text())
        if data.pop("schema", None) != "latent-readout/v1":
            raise ValueError("unsupported model schema; use import-legacy for historical JSON")
        return cls(**data)


def _compatible(*datasets: ActivationDataset) -> None:
    first = datasets[0]
    seen: set[str] = set()
    for data in datasets:
        if data.dictionary_id != first.dictionary_id or data.activations.shape[1] != first.activations.shape[1]:
            raise ValueError("all splits must use the same dictionary and column order")
        if data.sequence_ids is not None:
            ids = set(data.sequence_ids)
            if ids & seen:
                raise ValueError("sequence IDs overlap between supplied splits")
            seen.update(ids)


def _check_task(data: ActivationDataset, task: str) -> None:
    if task not in {"binary", "continuous"}:
        raise ValueError("task must be binary or continuous")
    if task == "binary" and not np.isin(data.labels, [0, 1]).all():
        raise ValueError("binary labels must be exactly 0 or 1")


def _ols(data: ActivationDataset, ids: list[int], task: str) -> LinearReadout:
    design = np.column_stack([np.ones(len(data.labels)), data.activations[:, ids]]).astype(np.float64)
    beta = np.linalg.lstsq(design, data.labels, rcond=None)[0]
    return LinearReadout(ids, beta[1:].tolist(), float(beta[0]), data.activations.shape[1],
                         data.dictionary_id, task, 0.5 if task == "binary" else None)


def _tune(model: LinearReadout, valid: ActivationDataset, tie_break: str) -> tuple[LinearReadout, float]:
    prediction = model.predict(valid.activations)
    if model.task == "continuous":
        return model, compute_composite_r2(valid.labels, prediction)
    thresholds = np.round(np.linspace(0, 1, 21), 10)
    scored = [(compute_mean_macro_f1(valid.labels, prediction, valid.lengths.tolist(), float(t)), float(t))
              for t in thresholds]
    key = (lambda pair: (-pair[0], pair[1])) if tie_break == "lower" else (
        lambda pair: (-pair[0], abs(pair[1] - 0.5), pair[1]))
    score, threshold = min(scored, key=key)
    return replace(model, threshold=threshold), float(score)


def fit(train: ActivationDataset, valid: ActivationDataset, *, max_latents: int = 10,
        task: str = "binary", tie_break: str = "lower", method: str = "residual",
        fixed_budget: bool = False) -> LinearReadout:
    """Select on train, refit each prefix on train, tune K/threshold on valid.

    No test argument is accepted. `lower` matches Biological100; `closest-half`
    prioritizes thresholds near 0.5. A smaller K wins exact validation-score ties.
    `best-single` reproduces the training F1 search for binary tasks (raw activation,
    not an affine single-feature probe). Continuous best-single uses K=1 OLS.
    fixed_budget retains the requested final prefix, for shared-set comparisons.
    """
    _compatible(train, valid)
    for data in (train, valid):
        _check_task(data, task)
    if tie_break not in {"lower", "closest-half"}:
        raise ValueError("tie_break must be lower or closest-half")
    if method not in {"residual", "topk", "best-single"}:
        raise ValueError("method must be residual, topk or best-single")
    width = train.activations.shape[1]
    if not isinstance(max_latents, int) or not 1 <= max_latents <= width:
        raise ValueError("max_latents must be between 1 and dictionary width")
    if method == "best-single" and task == "binary":
        winner = sweep_best_single_neuron(train.activations, train.labels, train.lengths.tolist(),
                                         np.linspace(0, 1, 11))
        candidate = LinearReadout([winner.neuron_id], [1.0], 0.0, width, train.dictionary_id)
        candidates = [candidate]
    else:
        count = 1 if method == "best-single" else max_latents
        if method == "topk":
            x = np.asarray(train.activations, dtype=np.float64)
            x = x - x.mean(axis=0)
            y = train.labels - train.labels.mean()
            norms = np.sum(x * x, axis=0)
            gain = np.divide((x.T @ y) ** 2, norms, out=np.zeros(width), where=norms > 0)
            ids = np.argsort(-gain, kind="stable")[:count].tolist()
        else:
            result = select_neurons_iteratively(train.activations, train.labels, count,
                                                seq_lengths=train.lengths.tolist(),
                                                solve_closed_form=False, compute_step_f1=False, verbose=False)
            # Historical engine can repeat a column on an exhausted/degenerate
            # dictionary. The portable readout stores each original column once.
            ids = list(dict.fromkeys(int(s.neuron_id) for s in result.steps if s.neuron_id >= 0))
        candidates = [_ols(train, ids[:k], task) for k in range(1, len(ids) + 1)]
    selected, best_score = None, -np.inf
    history = []
    for candidate in candidates:
        candidate, score = _tune(candidate, valid, tie_break)
        history.append({"k": len(candidate.latent_ids), "validation_score": score,
                        "threshold": candidate.threshold, "latent_ids": candidate.latent_ids,
                        "coefficients": candidate.coefficients, "intercept": candidate.intercept})
        if selected is None or score > best_score or fixed_budget:
            selected, best_score = candidate, score
    assert selected is not None
    selected.metadata = {"method": method, "selection_split": "validation", "tie_break": tie_break,
                         "budget_selection": "fixed" if fixed_budget else "validation",
                         "requested_max_latents": max_latents, "validation_path": history,
                         "training_tokens": len(train.labels), "validation_tokens": len(valid.labels),
                         "metric": "sequence_macro_class_f1" if task == "binary" else "pooled_r2"}
    return selected


def evaluate(model: LinearReadout, data: ActivationDataset) -> dict[str, float | int | str]:
    """Evaluate a frozen readout; never search thresholds or change coefficients."""
    if model.dictionary_id != data.dictionary_id:
        raise ValueError("dictionary_id does not match the frozen model")
    _check_task(data, model.task)
    prediction = model.predict(data.activations)
    if model.task == "binary":
        value = compute_mean_macro_f1(data.labels, prediction, data.lengths.tolist(), model.threshold)
        metric = "sequence_macro_class_f1"
    else:
        value = compute_composite_r2(data.labels, prediction)
        metric = "pooled_r2"
    return {"metric": metric, "value": float(value), "sequences": len(data.lengths),
            "tokens": len(data.labels), "k": len(model.latent_ids)}


def refit_leave_one_out(model: LinearReadout, train: ActivationDataset,
                       valid: ActivationDataset, test: ActivationDataset) -> dict[str, Any]:
    """Fixed-set conditional complementarity: refit remaining columns, never reselect."""
    _compatible(train, valid, test)
    if model.dictionary_id != train.dictionary_id or model.n_features != train.activations.shape[1]:
        raise ValueError("model and data dictionary differ")
    for data in (train, valid, test):
        _check_task(data, model.task)
    full, _ = _tune(_ols(train, model.latent_ids, model.task), valid, "closest-half")
    full_score = evaluate(full, test)
    rows = []
    for latent in model.latent_ids:
        ids = [i for i in model.latent_ids if i != latent]
        reduced, _ = _tune(_ols(train, ids, model.task), valid, "closest-half")
        score = evaluate(reduced, test)
        rows.append({"latent_id": latent, "full_score": full_score["value"], "reduced_score": score["value"],
                     "delta_refit": full_score["value"] - score["value"], "model": reduced.to_dict()})
    return {"full_model": full.to_dict(), "metric": full_score["metric"], "leave_one_out": rows,
            "scope": "fixed selected set; train OLS; per-model validation threshold; test point estimates"}
