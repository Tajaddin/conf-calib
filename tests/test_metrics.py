"""Metric tests with known synthetic cases."""

from __future__ import annotations

import numpy as np
import pytest

from conf_calib import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)


def test_ece_perfect_calibration_is_zero() -> None:
    # If every prediction is 0.5 and exactly half the labels are 1, ECE = 0.
    probs = np.full(1000, 0.5)
    labels = np.zeros(1000, dtype=int)
    labels[:500] = 1
    np.random.default_rng(0).shuffle(labels)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_extreme_overconfidence_high() -> None:
    # Every prediction says probability 1.0 but labels are 50/50 → ECE = 0.5.
    probs = np.ones(1000)
    labels = np.array([0, 1] * 500)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.5, abs=1e-6)


def test_ece_extreme_underconfidence_high() -> None:
    probs = np.zeros(1000)
    labels = np.array([0, 1] * 500)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.5, abs=1e-6)


def test_brier_perfect_score_is_zero() -> None:
    probs = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert brier_score(probs, labels) == 0.0


def test_brier_extreme_wrong_is_one() -> None:
    probs = np.array([1.0, 0.0])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == 1.0


def test_log_loss_perfect_is_near_zero() -> None:
    probs = np.array([0.999999, 0.000001])
    labels = np.array([1, 0])
    assert log_loss(probs, labels) < 1e-3


def test_log_loss_uniform_is_ln2() -> None:
    probs = np.full(1000, 0.5)
    labels = np.random.default_rng(0).integers(0, 2, 1000)
    assert log_loss(probs, labels) == pytest.approx(np.log(2), abs=1e-4)


def test_reliability_curve_uniform_binning() -> None:
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    curve = reliability_curve(probs, labels, n_bins=5)
    assert curve.bin_edges.shape == (6,)
    # Empty bins should have nan.
    assert np.isnan(curve.bin_confidence[2])  # bin [0.4, 0.6) has no samples
    assert curve.bin_count[2] == 0


def test_reliability_curve_quantile_binning_balances_counts() -> None:
    probs = np.linspace(0.0, 1.0, 100)
    labels = (probs > 0.5).astype(int)
    curve = reliability_curve(probs, labels, n_bins=10, binning="quantile")
    # Each bin should have ~10 samples.
    assert curve.bin_count.min() >= 5


def test_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.5]), np.array([1]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([2]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([]), np.array([]))


def test_reliability_curve_bad_binning() -> None:
    with pytest.raises(ValueError):
        reliability_curve(np.array([0.5]), np.array([0]), binning="bogus")  # type: ignore[arg-type]
