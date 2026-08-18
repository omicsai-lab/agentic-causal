from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

HERE = Path(__file__).resolve().parent
SEMANTIC_ORCH_ROOT = HERE.parent
BENCHMARK_DIR = SEMANTIC_ORCH_ROOT / "benchmark"
RESULTS_RAW_DIR = SEMANTIC_ORCH_ROOT / "results" / "raw"
RESULTS_PROCESSED_DIR = SEMANTIC_ORCH_ROOT / "results" / "processed"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_supported_prompts() -> List[Dict[str, Any]]:
    return read_jsonl(BENCHMARK_DIR / "prompts_supported.jsonl")


def load_challenge_prompts() -> List[Dict[str, Any]]:
    return read_jsonl(BENCHMARK_DIR / "prompts_challenge.jsonl")


class JsonlWriter:
    """Append-only JSONL writer; each write() call is flushed immediately
    so a crashed/interrupted evaluation run keeps whatever completed."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def already_completed_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple]:
    """Read an existing raw-results JSONL (if present) and return the set of
    (key_field...) tuples already recorded, so a resumed run can skip them."""
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl(path):
        try:
            done.add(tuple(row[k] for k in key_fields))
        except KeyError:
            continue
    return done
