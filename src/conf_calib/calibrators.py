"""Three classic calibrators: Platt scaling, isotonic regression, temperature scaling.

All implement the :class:`Calibrator` protocol::

    cal.fit(probs, labels)
    cal.transform(probs) -> calibrated_probs

Each calibrator is fit on a held-out validation set and applied to test
predictions. Picklable for save / load.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


_EPS = 1e-12


def _validate(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape != labels.shape:
        raise ValueError("probs and labels shape mismatch")
    if probs.ndim != 1:
        raise ValueError("probs must be 1D")
    return probs, labels


class Calibrator(Protocol):
    """All calibrators expose ``fit`` and ``transform``."""

    name: str
    fitted: bool

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Calibrator": ...
    def transform(self, probs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Platt scaling: fit logistic regression on raw probabilities.

    For binary classification this is a 1-D logistic regression with the raw
    probability (or, equivalently, its logit) as the only feature. We use the
    logit because Platt's original paper does, and it makes the model
    invariant to monotonic transforms of the raw confidence.
    """

    name = "platt"

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        probs, labels = _validate(probs, labels)
        logits = _logit(probs)
        self._lr = LogisticRegression(C=1e6)  # near-unregularized
        self._lr.fit(logits.reshape(-1, 1), labels)
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted or self._lr is None:
            raise RuntimeError("Call fit() before transform()")
        probs = np.asarray(probs, dtype=np.float64)
        logits = _logit(probs)
        return self._lr.predict_proba(logits.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Non-parametric monotonic mapping fit by isotonic regression.

    More flexible than Platt — can capture any monotonic re-shaping of the
    confidence curve. Slightly more prone to overfitting, so prefer it when
    the validation set is large (≥ a few hundred samples).
    """

    name = "isotonic"

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        probs, labels = _validate(probs, labels)
        self._iso.fit(probs, labels.astype(np.float64))
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        probs = np.asarray(probs, dtype=np.float64)
        return self._iso.predict(probs)


# ---------------------------------------------------------------------------
# Temperature scaling (single-parameter)
# ---------------------------------------------------------------------------


class TemperatureCalibrator:
    """Single-parameter temperature scaling on logits.

    Given raw probabilities ``p``, we treat ``logit(p)`` as a 1D logit and
    apply ``sigmoid(logit(p) / T)``. ``T > 1`` softens overconfident outputs;
    ``T < 1`` sharpens underconfident ones. ``T`` is fit by minimizing
    negative log-likelihood (= log loss) on the validation set.

    For verbalized confidence (e.g. Llama saying "85"), this is the most
    interpretable knob — recruiters and dashboards can see "we softened
    Llama's confidence by 1.4×" rather than a learned isotonic curve.
    """

    name = "temperature"

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        probs, labels = _validate(probs, labels)
        logits = _logit(probs)

        def nll(t: float) -> float:
            t = max(t, 1e-3)
            cal = _sigmoid(logits / t)
            cal = np.clip(cal, _EPS, 1 - _EPS)
            return float(-np.mean(labels * np.log(cal) + (1 - labels) * np.log(1 - cal)))

        # Search over a wide range; scipy's bounded scalar minimizer.
        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(result.x)
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        probs = np.asarray(probs, dtype=np.float64)
        logits = _logit(probs)
        return _sigmoid(logits / self.temperature)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
