"""Raw-coordinate OLS and ridge probes, fitted from train-only statistics.

No SAE, activation rectification, dimensionality reduction, or score clipping.
The Gram matrix is accumulated in float64; ridge uses alpha = N * lambda.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import eigh


@dataclass
class Statistics:
    n: int
    mean_x: np.ndarray
    mean_y: float
    gram: np.ndarray
    xy: np.ndarray


def accumulate(batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> Statistics:
    """Accumulate centered moments using a stable batch-combination formula."""
    n = 0
    for features, labels in batches:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("Feature/label token alignment or empty batch")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite input")
        m = len(y)
        bx, by = x.mean(axis=0), float(y.mean())
        xc, yc = x - bx, y - by
        bg, bc = xc.T @ xc, xc.T @ yc
        if n == 0:
            mean_x, mean_y, gram, xy = bx, by, bg, bc
        else:
            dx, dy = bx - mean_x, by - mean_y
            scale = n * m / (n + m)
            gram += bg + scale * np.outer(dx, dx)
            xy += bc + scale * dx * dy
            mean_x += dx * (m / (n + m))
            mean_y += dy * (m / (n + m))
        n += m
    if not n:
        raise ValueError("No training tokens")
    return Statistics(n, mean_x, mean_y, (gram + gram.T) / 2, xy)


def ols_path(stats: Statistics, max_k: int = 10) -> list[dict]:
    """Conditional squared-error gain, stable coordinate-ID tie breaking."""
    g, c = stats.gram, stats.xy
    residual_gram, residual_xy = g.copy(), c.copy()
    selected: list[int] = []
    models = []
    # Relative to each original column norm, plus an absolute machine floor.
    tolerance = 1e-12 * np.maximum(np.diag(g), 1.0)
    for _ in range(max_k):
        norm = np.diag(residual_gram)
        valid = norm > tolerance
        valid[selected] = False
        gains = np.full(len(c), -np.inf)
        gains[valid] = residual_xy[valid] ** 2 / norm[valid]
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]):
            break
        selected.append(j)
        ids = np.array(selected, dtype=int)
        coef = np.linalg.lstsq(g[np.ix_(ids, ids)], c[ids], rcond=1e-12)[0]
        models.append({"method": "Raw-OLS", "K": len(ids), "coordinates": ids.tolist(),
                       "coefficients": coef.tolist(),
                       "intercept": float(stats.mean_y - stats.mean_x[ids] @ coef),
                       "conditional_gain": float(gains[j]), "lambda": None})
        column = residual_gram[:, j].copy()
        residual_xy -= column * (residual_xy[j] / norm[j])
        residual_gram -= np.outer(column, column) / norm[j]
        residual_gram = (residual_gram + residual_gram.T) / 2
    return models


def ridge_path(stats: Statistics) -> tuple[list[dict], dict]:
    """Standardized full-coordinate ridge with unpenalized intercept."""
    variance = np.diag(stats.gram) / stats.n
    keep = variance > 1e-12 * max(1.0, float(variance.max()))
    ids = np.flatnonzero(keep)
    scale = np.sqrt(variance[keep])
    gram = stats.gram[np.ix_(ids, ids)] / np.outer(scale, scale) / stats.n
    rhs = stats.xy[ids] / scale / stats.n
    values, vectors = eigh(gram, check_finite=False)
    if values.size and values.min() < -1e-9 * max(1.0, float(values.max())):
        raise ValueError("Training Gram is not positive semidefinite")
    values = np.maximum(values, 0)
    projected = vectors.T @ rhs
    models = []
    for exponent in range(-8, 5):
        penalty = float(10.0 ** exponent)
        standardized_coef = vectors @ (projected / (values + penalty))
        coef = standardized_coef / scale
        models.append({"method": "Raw-Ridge", "K": None, "lambda": penalty,
                       "alpha": stats.n * penalty, "coordinates": ids.tolist(),
                       "coefficients": coef.tolist(),
                       "intercept": float(stats.mean_y - stats.mean_x[ids] @ coef)})
    normalization = {"n_train_tokens": stats.n, "mean": stats.mean_x.tolist(),
                     "retained_coordinates": ids.tolist(), "scale": scale.tolist(),
                     "removed_coordinates": np.flatnonzero(~keep).tolist(),
                     "near_zero_variance_rule": "variance <= 1e-12 * max(1, max variance)",
                     "formula": "w_raw = w_standardized / scale; b_raw = mean_y - mean_x @ w_raw",
                     "objective": "||y-b-Zw||^2/(2*N) + lambda*||w||^2/2"}
    return models, normalization


def predict(features: np.ndarray, model: dict) -> np.ndarray:
    """Predict unbounded raw linear scores in the original coordinates."""
    return (np.asarray(features[:, model["coordinates"]], dtype=np.float64)
            @ np.asarray(model["coefficients"]) + model["intercept"])


def confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Class-macro F1 for one sequence, with zero denominators set to zero."""
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(scores) >= threshold
    if y.shape != pred.shape or not len(y):
        raise ValueError("Score/label shape mismatch or empty sequence")
    tp, tn = int(np.sum(y & pred)), int(np.sum(~y & ~pred))
    fp, fn = int(np.sum(~y & pred)), int(np.sum(y & ~pred))
    pd, nd = 2 * tp + fp + fn, 2 * tn + fp + fn
    positive, negative = 2 * tp / pd if pd else 0.0, 2 * tn / nd if nd else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "f1_positive": positive, "f1_negative": negative,
            "macro_f1": (positive + negative) / 2}


def threshold_grid(labels: np.ndarray, scores: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sequence-equal class-macro F1 at the frozen 0..1, step .05 grid."""
    if offsets[0] != 0 or offsets[-1] != len(labels) or len(scores) != len(labels):
        raise ValueError("Token boundary mismatch")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Empty or reversed sequence boundary")
    thresholds = np.arange(21) / 20
    result = np.zeros(21)
    for start, end in zip(offsets[:-1], offsets[1:]):
        y = labels[start:end].astype(bool)[:, None]
        pred = scores[start:end, None] >= thresholds
        tp = np.sum(y & pred, axis=0)
        tn = np.sum(~y & ~pred, axis=0)
        fp = np.sum(~y & pred, axis=0)
        fn = np.sum(y & ~pred, axis=0)
        pd, nd = 2 * tp + fp + fn, 2 * tn + fp + fn
        result += (np.divide(2 * tp, pd, out=np.zeros(21), where=pd != 0)
                   + np.divide(2 * tn, nd, out=np.zeros(21), where=nd != 0)) / 2
    return result / (len(offsets) - 1)


def select_validation(candidates: list[dict], fig2: bool = False) -> dict:
    """Pure validation-only selection; the caller cannot supply test metrics."""
    def key(row: dict) -> tuple:
        budget = row["K"] if row["method"] == "Raw-OLS" else -row["lambda"]
        threshold = row.get("threshold", 0.0)
        return (-row["valid_metric"], budget,
                abs(threshold - 0.5) if fig2 else threshold, threshold)
    if any(any(k.startswith("test") for k in row) for row in candidates):
        raise ValueError("Test metrics must not be passed to validation selection")
    return min(candidates, key=key).copy()
