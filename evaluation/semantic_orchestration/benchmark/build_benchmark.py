"""
Deterministically build the frozen JSONL benchmark files from the literal
Python source data in prompts_supported.py / prompts_challenge.py.

Usage:
    python3 evaluation/semantic_orchestration/benchmark/build_benchmark.py

This script performs NO network calls and NO randomness. Running it twice
on unchanged source data must produce byte-identical JSONL output (keys are
emitted in a fixed order via `_ordered`, and json.dumps uses sort_keys=False
with an explicit key order instead, so re-ordering the source dicts does not
change the frozen bytes).
"""

from __future__ import annotations

import json
from pathlib import Path

from prompts_supported import ALL_SUPPORTED_PROMPTS, CATEGORIES
from prompts_challenge import ALL_CHALLENGE_PROMPTS, CHALLENGE_TYPES

HERE = Path(__file__).resolve().parent

SUPPORTED_FIELD_ORDER = [
    "id",
    "capability",
    "category",
    "domain",
    "text",
    "gold_capability_id",
    "gold_outcome_type",
]

CHALLENGE_FIELD_ORDER = [
    "id",
    "challenge_type",
    "domain",
    "text",
    "expected_behavior",
]


def _ordered(d: dict, order: list[str]) -> dict:
    out = {}
    for k in order:
        if k in d:
            out[k] = d[k]
    # Any unexpected extra keys are appended (stable, sorted) rather than
    # silently dropped, so schema drift is visible in the JSONL diff.
    for k in sorted(d.keys()):
        if k not in out:
            out[k] = d[k]
    return out


def build_supported_jsonl() -> Path:
    out_path = HERE / "prompts_supported.jsonl"
    lines = []
    for p in ALL_SUPPORTED_PROMPTS:
        entry = _ordered(p, SUPPORTED_FIELD_ORDER)
        entry.setdefault("gold_outcome_type", "time_to_event" if entry["capability"] == "survival_adjusted_curves" else "unknown")
        lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def build_challenge_jsonl() -> Path:
    out_path = HERE / "prompts_challenge.jsonl"
    lines = []
    for p in ALL_CHALLENGE_PROMPTS:
        entry = _ordered(p, CHALLENGE_FIELD_ORDER)
        entry.setdefault("gold_capability_id", None)
        lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    sup = build_supported_jsonl()
    chal = build_challenge_jsonl()
    print(f"Wrote {sup} ({len(ALL_SUPPORTED_PROMPTS)} entries)")
    print(f"Wrote {chal} ({len(ALL_CHALLENGE_PROMPTS)} entries)")
    print(f"Categories: {CATEGORIES}")
    print(f"Challenge types: {CHALLENGE_TYPES}")


if __name__ == "__main__":
    main()
