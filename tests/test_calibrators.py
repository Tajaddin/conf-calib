"""Calibrator tests: each calibrator should reduce ECE on synthetic miscalibrated data."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from conf_calib import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
)


def _synthetic_miscalibrated(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Overconfident outputs: reported probs cluster near 0/1, true prob is moderate."""
    rng = np.random.default_rng(seed)
    probs = np.clip(rng.beta(0.5, 0.5, size=n), 1e-3, 1 - 1e-3)
    true_p = 0.3 + 0.4 * probs
    labels = (rng.random(n) < true_p).astype(int)
    return probs, labels


@pytest.mark.parametrize(
    "cal_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureCalibrator]
)
def test_calibrator_reduces_ece(cal_cls) -> None:
    probs, labels = _synthetic_miscalibrated()
    raw_ece = expected_calibration_error(probs, labels)
    cal = cal_cls()
    cal.fit(probs[:1000], labels[:1000])
    cal_test = cal.transform(probs[1000:])
    new_ece = expected_calibration_error(cal_test, labels[1000:])
    assert new_ece < raw_ece * 0.5  # at least halves ECE


@pytest.mark.parametrize(
    "cal_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureCalibrator]
)
def test_calibrator_reduces_brier(cal_cls) -> None:
    probs, labels = _synthetic_miscalibrated()
    raw_brier = brier_score(probs, labels)
    cal = cal_cls()
    cal.fit(probs[:1000], labels[:1000])
    cal_test = cal.transform(probs[1000:])
    new_brier = brier_score(cal_test, labels[1000:])
    assert new_brier < raw_brier


def test_temperature_finds_T_greater_than_1_for_overconfident() -> None:
    """Overconfident outputs should yield T > 1 (softening)."""
    probs, labels = _synthetic_miscalibrated()
    cal = TemperatureCalibrator().fit(probs[:1000], labels[:1000])
    assert cal.temperature > 1.0


def test_temperature_finds_T_less_than_1_for_underconfident() -> None:
    """Underconfident outputs (probs hug 0.5) should yield T < 1 (sharpening)."""
    rng = np.random.default_rng(1)
    n = 2000
    # Reported probs near 0.5 even when labels are quite separable.
    probs = np.clip(0.5 + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    labels = (rng.random(n) < (0.05 + 0.9 * (probs > 0.5))).astype(int)
    cal = TemperatureCalibrator().fit(probs, labels)
    assert cal.temperature < 1.0


def test_transform_requires_fit() -> None:
    for cls in (PlattCalibrator, IsotonicCalibrator, TemperatureCalibrator):
        cal = cls()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))


@pytest.mark.parametrize(
    "cal_cls", [PlattCalibrator, IsotonicCalibrator, TemperatureCalibrator]
)
def test_calibrator_pickle_roundtrip(cal_cls) -> None:
    probs, labels = _synthetic_miscalibrated()
    cal = cal_cls().fit(probs[:500], labels[:500])
    cal2 = pickle.loads(pickle.dumps(cal))
    np.testing.assert_allclose(cal.transform(probs[500:600]), cal2.transform(probs[500:600]))


def test_isotonic_output_bounded_in_unit() -> None:
    probs, labels = _synthetic_miscalibrated(n=500)
    cal = IsotonicCalibrator().fit(probs, labels)
    out = cal.transform(np.array([0.0, 0.01, 0.5, 0.99, 1.0]))
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_platt_monotonic() -> None:
    """Platt's transform is monotonic in raw probability."""
    probs, labels = _synthetic_miscalibrated(n=1000)
    cal = PlattCalibrator().fit(probs, labels)
    test_x = np.linspace(0.0, 1.0, 50)
    test_y = cal.transform(test_x)
    assert (np.diff(test_y) >= -1e-9).all()
