"""Collect Llama-3.1-8B verbalized confidence on N BoolQ questions via Groq.

Writes one JSON object per line to ``bench/llama_boolq.jsonl`` with::

    {"id", "question", "passage", "gold", "predicted", "confidence",
     "raw_text", "elapsed_ms"}

The script is **resumable**: it reads existing IDs from the output file and
only fetches missing ones. Safe to kill and restart.

Rate-limited to ~28 RPM via :class:`GroqClient`'s ``min_interval`` so it stays
under the 30 RPM free-tier ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BENCH = Path(__file__).resolve().parent
OUT_PATH = BENCH / "llama_boolq.jsonl"

SYSTEM = (
    "You answer factual yes/no questions grounded in a short passage. "
    "Reply with EXACTLY one line in the format:\n"
    "ANSWER: <YES or NO>\n"
    "CONFIDENCE: <integer 0-100, your subjective probability that ANSWER is correct>\n"
    "No other text. No reasoning. No prose."
)

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(YES|NO)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def build_prompt(passage: str, question: str) -> str:
    return f"Passage:\n{passage}\n\nQuestion: {question}"


def parse_response(text: str) -> tuple[str | None, float | None]:
    """Return (label, confidence_in_[0,1]) or (None, None) if unparseable."""
    if not text:
        return None, None
    ans_m = _ANSWER_RE.search(text)
    conf_m = _CONF_RE.search(text)
    label = ans_m.group(1).upper() if ans_m else None
    if conf_m is None:
        # Heuristic fallback: look for a bare 0-100 number on its own line.
        for line in text.splitlines():
            line = line.strip()
            if line.isdigit() and 0 <= int(line) <= 100:
                conf = float(line)
                return label, conf / 100.0
        return label, None
    conf = float(conf_m.group(1))
    return label, max(0.0, min(1.0, conf / 100.0))


def load_boolq(limit: int, seed: int) -> list[dict]:
    """Pull ``limit`` BoolQ validation examples, deterministic shuffle by ``seed``."""
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    indices = list(range(len(ds)))
    import random

    random.Random(seed).shuffle(indices)
    out: list[dict] = []
    for idx in indices[:limit]:
        ex = ds[idx]
        out.append(
            {
                "id": str(idx),
                "passage": ex["passage"],
                "question": ex["question"],
                "gold": bool(ex["answer"]),  # True/False
            }
        )
    return out


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                ids.add(str(obj["id"]))
    return ids


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", type=str, default="llama-3.1-8b-instant")
    p.add_argument("--min-interval-secs", type=float, default=2.1)
    p.add_argument("--out", type=str, default=str(OUT_PATH))
    args = p.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 2

    from groq import Groq, RateLimitError

    client = Groq(api_key=api_key)
    out_path = Path(args.out)
    done = existing_ids(out_path)
    examples = load_boolq(args.limit, args.seed)
    print(f"Total target: {len(examples)}, already done: {len(done)}, remaining: {len(examples) - len(done)}")

    last_call = 0.0
    parse_failures = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, ex in enumerate(examples):
            if ex["id"] in done:
                continue
            # Rate limit.
            now = time.monotonic()
            wait = args.min_interval_secs - (now - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()

            user_prompt = build_prompt(ex["passage"], ex["question"])
            start = time.perf_counter()
            text = ""
            err: str | None = None
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=50,
                    )
                    text = resp.choices[0].message.content or ""
                    break
                except RateLimitError:
                    backoff = 4.0 * (2 ** attempt)
                    print(f"  [{i+1}] rate limited, sleeping {backoff:.0f}s", file=sys.stderr)
                    time.sleep(backoff)
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {str(exc)[:200]}"
                    break

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            label, conf = parse_response(text)
            if label is None or conf is None:
                parse_failures += 1
            row: dict[str, Any] = {
                "id": ex["id"],
                "question": ex["question"],
                "passage": ex["passage"][:400],
                "gold": ex["gold"],
                "predicted": label,
                "confidence": conf,
                "raw_text": text,
                "elapsed_ms": elapsed_ms,
                "error": err,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                pct = 100 * (i + 1) / len(examples)
                print(
                    f"  [{i+1}/{len(examples)}] ({pct:.0f}%) "
                    f"parse_failures={parse_failures}"
                )

    print(f"\nDone. parse_failures={parse_failures}, results in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
