"""
Verify BENCHMARK_MANIFEST.json's recorded hashes match the current
benchmark files on disk. A failure here means the benchmark was edited
after freezing (or the manifest is stale and freeze_manifest.py needs to
be re-run) -- see EVALUATION_PROTOCOL.md, "Benchmark immutability".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "BENCHMARK_MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_exists():
    assert MANIFEST_PATH.exists(), "BENCHMARK_MANIFEST.json not found; run benchmark/freeze_manifest.py"


def test_manifest_file_hashes_match_disk():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for rel_path, info in manifest["files"].items():
        p = ROOT / rel_path
        assert p.exists(), f"manifest references missing file: {rel_path}"
        actual = _sha256(p)
        assert actual == info["sha256"], f"{rel_path} hash mismatch: benchmark was modified after freezing"
        assert p.stat().st_size == info["bytes"], f"{rel_path} size mismatch"


def test_manifest_counts_match_declared_totals():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["supported_total"] == 120
    assert counts["supported_causal_ate"] == 60
    assert counts["supported_survival_adjusted_curves"] == 60
    assert counts["challenge_total"] == 30
    assert sum(counts["challenge_by_type"].values()) == 30
    assert all(v == 10 for v in counts["supported_by_capability_category"].values())
    assert len(counts["supported_by_capability_category"]) == 12
