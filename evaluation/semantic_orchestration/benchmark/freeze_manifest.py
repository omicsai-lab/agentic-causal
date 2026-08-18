"""
Compute SHA-256 hashes over the frozen benchmark files and write
BENCHMARK_MANIFEST.json at the evaluation/semantic_orchestration root.

Run this AFTER build_benchmark.py and BEFORE any live model call. Once
written, the manifest is the source of truth for "was the benchmark
altered after evaluation started" -- see EVALUATION_PROTOCOL.md,
"Benchmark immutability".

Usage:
    python3 evaluation/semantic_orchestration/benchmark/freeze_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parents[1]

MANIFEST_INPUTS = [
    "benchmark/prompts_supported.py",
    "benchmark/prompts_challenge.py",
    "benchmark/build_benchmark.py",
    "benchmark/prompts_supported.jsonl",
    "benchmark/prompts_challenge.jsonl",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest() -> dict:
    import sys

    sys.path.insert(0, str(HERE))
    from prompts_supported import ALL_SUPPORTED_PROMPTS, CATEGORIES  # type: ignore
    from prompts_challenge import ALL_CHALLENGE_PROMPTS, CHALLENGE_TYPES  # type: ignore

    files = {}
    for rel in MANIFEST_INPUTS:
        p = ROOT / rel
        files[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}

    by_cap_cat = {}
    for p in ALL_SUPPORTED_PROMPTS:
        key = f"{p['capability']}::{p['category']}"
        by_cap_cat[key] = by_cap_cat.get(key, 0) + 1

    by_challenge_type = {}
    for p in ALL_CHALLENGE_PROMPTS:
        by_challenge_type[p["challenge_type"]] = by_challenge_type.get(p["challenge_type"], 0) + 1

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_freeze": _git_commit(),
        "benchmark_version": "1.0.0",
        "counts": {
            "supported_total": len(ALL_SUPPORTED_PROMPTS),
            "supported_causal_ate": sum(1 for p in ALL_SUPPORTED_PROMPTS if p["capability"] == "causal_ate"),
            "supported_survival_adjusted_curves": sum(
                1 for p in ALL_SUPPORTED_PROMPTS if p["capability"] == "survival_adjusted_curves"
            ),
            "supported_by_capability_category": by_cap_cat,
            "challenge_total": len(ALL_CHALLENGE_PROMPTS),
            "challenge_by_type": by_challenge_type,
        },
        "categories": CATEGORIES,
        "challenge_types": CHALLENGE_TYPES,
        "files": files,
    }
    return manifest


def main() -> None:
    manifest = build_manifest()
    out_path = ROOT / "BENCHMARK_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
