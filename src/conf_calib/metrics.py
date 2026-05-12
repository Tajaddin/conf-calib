"""Calibration metrics: ECE, Brier score, log-loss, reliability curve.

All functions operate on numpy arrays of shape ``(n,)``:

* ``probs``  — predicted probability of the positive class, in [0, 1]
* ``labels`` — ground-truth binary labels in {0, 1}

ECE uses equal-width binning by default. Pass ``binning="quantile"`` for
equal-mass binning, which gives a more robust ECE estimate when probabilities
are clumped near one end of the [0, 1] range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _validate(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape != labels.shape:
        raise ValueError(f"probs and labels shape mismatch: {probs.shape} vs {labels.shape}")
    if probs.ndim != 1:
        raise ValueError("probs and labels must be 1D arrays")
    if probs.size == 0:
        raise ValueError("empty input")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must be in [0, 1]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be in {0, 1}")
    return probs, labels


# ---------------------------------------------------------------------------
# Reliability curve (the data the diagram is drawn from)
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityCurve:
    """Per-bin reliability data, suitable for plotting a reliability diagram.

    ``bin_edges`` always has length ``n_bins + 1``. Empty bins have ``count = 0``
    and ``bin_confidence`` / ``bin_accuracy`` set to ``nan``.
    """

    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    binning: str


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    binning: Literal["uniform", "quantile"] = "uniform",
) -> ReliabilityCurve:
    """Return the reliability curve for plotting / inspection."""
    probs, labels = _validate(probs, labels)
    if binning == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif binning == "quantile":
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        raise ValueError(f"binning must be 'uniform' or 'quantile', got {binning!r}")
    edges = np.asarray(edges)

    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the right endpoint of the last bin so probs==1.0 isn't dropped.
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        bin_count[i] = n
        if n > 0:
            bin_conf[i] = float(probs[mask].mean())
            bin_acc[i] = float(labels[mask].mean())
    return ReliabilityCurve(edges, bin_conf, bin_acc, bin_count, binning)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    binning: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """Compute ECE = sum over bins of (count_b / N) * |conf_b - acc_b|.

    For binary classification this uses the predicted probability of the
    *positive* class directly, not the top-class confidence — which is the
    correct formulation when labels are 0/1.
    """
    curve = reliability_curve(probs, labels, n_bins=n_bins, binning=binning)
    n_total = curve.bin_count.sum()
    ece = 0.0
    for i in range(len(curve.bin_count)):
        n = curve.bin_count[i]
        if n == 0:
            continue
        ece += (n / n_total) * abs(curve.bin_confidence[i] - curve.bin_accuracy[i])
    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    binning: Literal["uniform", "quantile"] = "uniform",
    min_bin_count: int = 5,
) -> float:
    """Max over bins of |conf - acc|, ignoring bins with fewer than ``min_bin_count`` samples."""
    curve = reliability_curve(probs, labels, n_bins=n_bins, binning=binning)
    diffs = []
    for i in range(len(curve.bin_count)):
        if curve.bin_count[i] >= min_bin_count:
            diffs.append(abs(curve.bin_confidence[i] - curve.bin_accuracy[i]))
    return float(max(diffs)) if diffs else 0.0


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary label."""
    probs, labels = _validate(probs, labels)
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def log_loss(probs: np.ndarray, labels: np.ndarray, *, eps: float = 1e-12) -> float:
    """Binary log loss with epsilon clipping to avoid log(0)."""
    probs, labels = _validate(probs, labels)
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
