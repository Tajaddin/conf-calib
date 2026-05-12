"""End-to-end calibration experiment.

Loads ``bench/llama_boolq.jsonl`` (collected by ``collect_boolq.py``), splits
70/30 train/test, fits Platt / Isotonic / Temperature calibrators on the train
half, and reports ECE / Brier / log-loss / accuracy on the test half before
and after each calibration. Saves a reliability diagram to
``bench/reliability_diagram.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conf_calib import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)


BENCH = Path(__file__).resolve().parent
DEFAULT_IN = BENCH / "llama_boolq.jsonl"
DEFAULT_OUT_JSON = BENCH / "calibration_results.json"
DEFAULT_OUT_PLOT = BENCH / "reliability_diagram.png"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (probs, labels, meta). Skips rows with missing predicted/confidence.

    ``confidence`` is treated as P(YES). If ``predicted == "NO"``, we flip:
    P(YES) = 1 - confidence. ``gold`` is converted to 0/1 (YES=1).
    """
    probs: list[float] = []
    labels: list[int] = []
    skipped = 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n += 1
            label = obj.get("predicted")
            conf = obj.get("confidence")
            gold = obj.get("gold")
            if label is None or conf is None or gold is None:
                skipped += 1
                continue
            p_yes = float(conf) if label == "YES" else 1.0 - float(conf)
            probs.append(p_yes)
            labels.append(1 if gold else 0)
    return (
        np.asarray(probs, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        {"n_rows": n, "skipped": skipped, "n_used": len(probs)},
    )


def _report(label: str, probs: np.ndarray, y: np.ndarray) -> dict:
    return {
        "label": label,
        "n": int(probs.size),
        "ece": round(expected_calibration_error(probs, y, n_bins=15), 4),
        "brier": round(brier_score(probs, y), 4),
        "log_loss": round(log_loss(probs, y), 4),
        "accuracy": round(float(((probs >= 0.5).astype(int) == y).mean()), 4),
    }


def _plot_reliability(
    title_to_probs: dict[str, np.ndarray], y_test: np.ndarray, out_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping reliability diagram")
        return

    n_bins = 15
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    colors = {"raw": "#888", "platt": "#1f77b4", "isotonic": "#2ca02c", "temperature": "#d62728"}
    for label, probs in title_to_probs.items():
        curve = reliability_curve(probs, y_test, n_bins=n_bins)
        # Plot bin midpoints where data exists.
        valid = ~np.isnan(curve.bin_confidence)
        ax.plot(
            curve.bin_confidence[valid],
            curve.bin_accuracy[valid],
            marker="o",
            linestyle="-",
            label=label,
            color=colors.get(label, None),
            markersize=6,
        )
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title("Reliability diagram — Llama-3.1-8B verbalized confidence on BoolQ")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(DEFAULT_IN))
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-plot", type=str, default=str(DEFAULT_OUT_PLOT))
    args = p.parse_args()

    probs_all, labels_all, meta = _load(Path(args.input))
    print(f"Loaded {meta['n_used']} usable rows (skipped {meta['skipped']} of {meta['n_rows']})")
    n = len(probs_all)
    if n < 50:
        print("Not enough rows to fit calibration; need more data.")
        return 1

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(args.train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    p_train, y_train = probs_all[train_idx], labels_all[train_idx]
    p_test, y_test = probs_all[test_idx], labels_all[test_idx]

    raw_report = _report("raw", p_test, y_test)
    print("\nRaw (uncalibrated):")
    print(json.dumps(raw_report, indent=2))

    calibrators = {
        "platt": PlattCalibrator(),
        "isotonic": IsotonicCalibrator(),
        "temperature": TemperatureCalibrator(),
    }
    per_method_test_probs = {"raw": p_test}
    reports = [raw_report]
    for name, cal in calibrators.items():
        cal.fit(p_train, y_train)
        cal_test = cal.transform(p_test)
        rep = _report(name, cal_test, y_test)
        if name == "temperature":
            rep["temperature"] = round(cal.temperature, 3)  # type: ignore[attr-defined]
        reports.append(rep)
        per_method_test_probs[name] = cal_test
        print(f"\n{name.capitalize()}:")
        print(json.dumps(rep, indent=2))

    out_json = Path(args.out_json)
    out_json.write_text(
        json.dumps({"meta": meta, "reports": reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {out_json}")

    _plot_reliability(per_method_test_probs, y_test, Path(args.out_plot))

    # Compact markdown table.
    print("\n| Method | ECE | Brier | Log loss | Accuracy |")
    print("|---|---:|---:|---:|---:|")
    for r in reports:
        print(
            f"| {r['label']} | {r['ece']} | {r['brier']} | {r['log_loss']} | {r['accuracy']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
