# conf-calib

> Confidence calibration for LLM verbalized-probability outputs. Real benchmark: **998 BoolQ questions through Llama-3.1-8B** drops Expected Calibration Error from **0.148 to 0.030** (isotonic) and log-loss from **3.9 to 0.41** without changing accuracy.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

1000 BoolQ validation questions sent to **`llama-3.1-8b-instant`** via Groq with a verbalized-confidence prompt (`ANSWER: YES|NO` + `CONFIDENCE: 0-100`). 998 parsed cleanly (2 off-format failures), 700/300 train/test split, seed 11.

| Method | ECE ↓ | Brier ↓ | Log loss ↓ | Accuracy |
|---|---:|---:|---:|---:|
| **raw (uncalibrated)** | **0.1483** | 0.1520 | **3.9026** | 0.843 |
| Platt scaling | 0.0372 | 0.1267 | 0.4099 | 0.840 |
| **Isotonic regression** | **0.0304** | **0.1250** | **0.4060** | 0.843 |
| Temperature scaling (T=10, hit upper bound) | 0.0919 | 0.1394 | 0.4803 | 0.843 |

**The story:** Llama-3.1-8B is 84.3% accurate on this BoolQ subset but says "CONFIDENCE: 100" on ~91% of questions. Raw ECE is **0.148**. Isotonic regression cuts that to **0.030** (-80% relative) and slashes log-loss by **9.5×**, with no change in accuracy. Platt is a close second (ECE 0.037) and is a better choice than isotonic when the held-out validation set is small.

**Temperature scaling does not work well here** — verbalized confidences are clipped to ~100, so the logits start at the rail. Even T = 10 (the optimizer's upper bound) only pulls 1.0 down to ~0.67. Real fix would be unbounded T plus a richer prompt that yields continuous confidence. Documented in [Limitations](#limitations).

Reproduce:
```bash
export GROQ_API_KEY=gsk_...
python bench/collect_boolq.py --limit 1000 --seed 7
python bench/experiment.py
```

Raw outputs: [`bench/llama_boolq.jsonl`](bench/llama_boolq.jsonl) (1000 rows). Per-method metrics: [`bench/calibration_results.json`](bench/calibration_results.json). Reliability diagram: [`bench/reliability_diagram.png`](bench/reliability_diagram.png).

## Why this exists

Calibration textbooks use synthetic miscalibrated data. Production LLM teams have a real problem:

> The model says "100% confident" 91% of the time, but it's wrong 16% of the time. The Brier score is *worse* than a coin flip.

That's exactly what `conf-calib` measures and fixes. It takes a stream of `(predicted_label, verbalized_confidence, true_label)` triples, fits a one-line calibrator on a held-out set, and produces calibrated probabilities you can route on (e.g., "route to human review when calibrated p < 0.7" instead of "when raw confidence < 70" which never triggers).

## Quickstart

```bash
pip install -e ".[plot,groq,data]"
```

```python
import numpy as np
from conf_calib import IsotonicCalibrator, expected_calibration_error

# Llama gave: P(YES) for 1000 BoolQ questions
probs = np.array([...])      # shape (1000,)
labels = np.array([...])     # shape (1000,) in {0, 1}

# Hold out 30% for evaluation
train, test = probs[:700], probs[700:]
train_y, test_y = labels[:700], labels[700:]

cal = IsotonicCalibrator().fit(train, train_y)
calibrated = cal.transform(test)

print(f"Raw ECE:        {expected_calibration_error(test, test_y):.4f}")
print(f"Calibrated ECE: {expected_calibration_error(calibrated, test_y):.4f}")
```

Save / load the fitted calibrator:

```python
import pickle
with open("isotonic.pkl", "wb") as f:
    pickle.dump(cal, f)
```

## Three calibrators built in

| Method | Parameters | When to use | Failure mode |
|---|---|---|---|
| `PlattCalibrator` | logistic regression in logit-space (2 params) | Default for production. Robust on small validation sets (≥ 100). | Forces a sigmoid shape; can't correct two-bump miscalibration |
| `IsotonicCalibrator` | non-parametric, monotonic step function | Best when you have ≥ 500 held-out samples. Highest ECE reduction in our benchmark. | Slightly overfits on tiny validation sets |
| `TemperatureCalibrator` | single scalar T | When you want one number to put in a dashboard ("our model is 1.7× overconfident") | Cannot help when confidences are saturated at 0 or 1 (see Limitations) |

All three implement the same `Calibrator` protocol (`fit`, `transform`) and are picklable.

## Metrics

```python
from conf_calib import (
    expected_calibration_error,    # ECE with uniform or quantile binning
    brier_score,                   # mean squared error vs labels
    log_loss,                      # binary log-loss with epsilon clipping
    reliability_curve,             # for drawing reliability diagrams
)

curve = reliability_curve(probs, labels, n_bins=15, binning="uniform")
# curve.bin_edges, curve.bin_confidence, curve.bin_accuracy, curve.bin_count
```

`ECE` supports both equal-width (`uniform`) and equal-mass (`quantile`) binning. Use `quantile` when probabilities are clumped near 0 or 1 — exactly the verbalized-confidence case.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
25 passed in 2.14s
```

Covers ECE math (known synthetic cases including 0 and 0.5), Brier extremes, log-loss limits, reliability-curve binning, all three calibrators (each must reduce ECE by ≥ 50% on synthetic miscalibrated data), pickle round-trip, monotonicity, and edge cases.

## Reproduce the benchmark from scratch

```bash
pip install -e ".[plot,groq,data,dev]"

# 1. Collect 1000 LLM outputs (free, ~35 min on Groq's 30 RPM tier)
export GROQ_API_KEY=gsk_...
python bench/collect_boolq.py --limit 1000 --seed 7

# 2. Run calibration experiment
python bench/experiment.py
```

The collection script is **resumable** — it reads existing IDs from `llama_boolq.jsonl` and only fetches missing ones. Kill it any time.

## Project layout

```
.
├── src/conf_calib/
│   ├── __init__.py
│   ├── metrics.py            # ECE, Brier, log-loss, reliability curve
│   └── calibrators.py        # Platt, Isotonic, Temperature
├── tests/                    # 25 pytest cases
└── bench/
    ├── collect_boolq.py          # Run Llama-3.1-8B via Groq on BoolQ
    ├── experiment.py             # Fit all three calibrators, compare
    ├── llama_boolq.jsonl         # 1000 real LLM outputs (committed)
    ├── calibration_results.json  # Per-method metrics
    └── reliability_diagram.png   # Visualization
```

## Limitations

**Verbalized confidence is a hard input.** Llama-3.1-8B picks discrete values — almost always "100", occasionally "80" or "90". Two consequences:

* Temperature scaling saturates. To pull `confidence = 1.0` down to a calibrated 0.84, we'd need `T → ∞`. The optimizer hit our `T = 10` upper bound and stayed there. A future v0.2 should either (a) widen bounds and accept that T is no longer an interpretable "softening factor," or (b) add a logit-bias calibrator that works on the raw 0-100 grid.
* Isotonic is essentially memorizing the empirical mapping `P(YES|conf=1.0) = 0.84` and `P(YES|conf=0.8) = ~0.78`. With richer continuous confidences (via top-logprobs or sampling-based estimates), the calibrators would have much more signal to fit.

**Calibration ≠ accuracy.** All three calibrators keep accuracy within a tenth of a percent of raw. If you need higher accuracy, calibration isn't the tool — fine-tune the model or use a stronger LLM.

**Single-task benchmark.** BoolQ is a passage-grounded yes/no benchmark. Calibration learned here may not transfer to free-form domains (legal classification, content moderation). Always re-fit on a held-out set from the *target* distribution.

**No multi-class support yet.** Public API is binary-only. Multi-class (top-1 vs rest, or one-vs-all calibration) is a v0.2 candidate.

## License

MIT — see [LICENSE](LICENSE).
