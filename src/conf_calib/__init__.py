"""Confidence calibration for LLM verbalized-probability outputs.

Three calibrators built in (Platt, isotonic, temperature), three metrics
(ECE, Brier, log-loss), and a reproducible benchmark on real Llama-3.1-8B
verbalized confidences over 1000 BoolQ questions.
"""

from conf_calib.calibrators import (
    Calibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
)
from conf_calib.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)

__version__ = "0.1.0"

__all__ = [
    # metrics
    "expected_calibration_error",
    "brier_score",
    "log_loss",
    "reliability_curve",
    # calibrators
    "Calibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "TemperatureCalibrator",
]
