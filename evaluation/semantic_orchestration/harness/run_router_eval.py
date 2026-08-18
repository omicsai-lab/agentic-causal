"""
Run the GPT-5.4 router (src/agent/router_llm.py::llm_choose_capability)
against the frozen benchmark.

  - Supported prompts (120): 3 repeats each, temperature 0, model="gpt-5.4"
    fixed (see harness/config.py). No model substitution: if the requested
    model errors, the error is recorded verbatim -- the harness never
    retries with a different model id.
  - Challenge prompts (30): 1 repeat each (see run_challenge_eval.py,
    which calls the same call_router function for the challenge half).

This script performs live API calls when executed with a non-flag
invocation. It is NOT run as part of Step 1 (see task instructions:
"Do not run the full GPT-5.4 supported/challenge evaluation in this
step"). It exists, fully implemented, so Step 2 can invoke it directly.

Usage:
    python3 -m evaluation.semantic_orchestration.harness.run_router_eval [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.config import (  # noqa: E402
    REQUESTED_MODEL,
    SUPPORTED_REPEATS,
)
from evaluation.semantic_orchestration.harness.io_utils import (  # noqa: E402
    RESULTS_RAW_DIR,
    JsonlWriter,
    already_completed_keys,
    load_supported_prompts,
)
from evaluation.semantic_orchestration.harness.model_client import call_router  # noqa: E402


def run(out_path: Path, limit: int | None = None, model: str = REQUESTED_MODEL) -> int:
    prompts = load_supported_prompts()
    if limit is not None:
        prompts = prompts[:limit]

    done = already_completed_keys(out_path, ("prompt_id", "repeat_index"))
    n_written = 0

    with JsonlWriter(out_path) as writer:
        for p in prompts:
            for repeat_index in range(1, SUPPORTED_REPEATS + 1):
                key = (p["id"], repeat_index)
                if key in done:
                    continue
                result = call_router(
                    prompt_id=p["id"],
                    request_text=p["text"],
                    repeat_index=repeat_index,
                    model=model,
                )
                record = {
                    "prompt_id": p["id"],
                    "repeat_index": repeat_index,
                    "gold_capability_id": p["gold_capability_id"],
                    "capability": p["capability"],
                    "category": p["category"],
                    "predicted_capability_id": result.capability_id,
                    "reason": result.reason,
                    "fallback_detected": result.fallback_detected,
                    "fallback_marker": result.fallback_marker,
                    "harness_error": result.harness_error,
                    "requested_model": model,
                    "api_call_succeeded": result.telemetry.api_call_succeeded,
                    "api_exception_type": result.telemetry.api_exception_type,
                    "api_exception": result.telemetry.api_exception,
                    "returned_models": result.telemetry.returned_models,
                    "model_substituted": result.telemetry.model_substituted,
                    "latency_seconds": result.telemetry.latency_seconds,
                }
                writer.write(record)
                n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N supported prompts (for smoke testing).")
    ap.add_argument("--out", type=str, default=str(RESULTS_RAW_DIR / "router_supported.jsonl"))
    ap.add_argument("--model", type=str, default=REQUESTED_MODEL)
    args = ap.parse_args()

    n = run(Path(args.out), limit=args.limit, model=args.model)
    print(f"Wrote {n} new router-eval records to {args.out}")


if __name__ == "__main__":
    main()
