"""
Minimal preflight check: can the harness reach the requested model
("gpt-5.4", see config.REQUESTED_MODEL) at all? Makes AT MOST ONE small
API call (a single router call on a throwaway prompt, csv unused). Does
NOT touch the benchmark files, does NOT write to results/raw, and must
NEVER be followed automatically by the 780-call evaluation.

Usage:
    python3 -m evaluation.semantic_orchestration.harness.preflight
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.config import REQUESTED_MODEL  # noqa: E402
from evaluation.semantic_orchestration.harness.model_client import call_router  # noqa: E402

PREFLIGHT_PROMPT = "Estimate the average treatment effect of drug A vs placebo on outcome in trial.csv."


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("PREFLIGHT: SKIPPED -- OPENAI_API_KEY is not set in this environment.")
        print("No API call was made. The 780-call GPT-5.4 evaluation cannot run until a key is configured.")
        return 2

    result = call_router(prompt_id="preflight", request_text=PREFLIGHT_PROMPT, repeat_index=0, model=REQUESTED_MODEL)

    if result.harness_error:
        print(f"PREFLIGHT: FAILED -- uncaught error calling router with model={REQUESTED_MODEL!r}: {result.harness_error}")
        return 1

    if not result.telemetry.attempted:
        print("PREFLIGHT: FAILED -- no API call was attempted (openai package unavailable at import time?).")
        return 1

    if not result.telemetry.api_call_succeeded:
        print(
            f"PREFLIGHT: FAILED -- API call attempted but did not succeed. "
            f"exception_type={result.telemetry.api_exception_type} exception={result.telemetry.api_exception}"
        )
        return 1

    if result.fallback_detected:
        print(
            f"PREFLIGHT: FAILED -- API call reported success but router internally fell back "
            f"(marker={result.fallback_marker!r}); check that model={REQUESTED_MODEL!r} is a valid, accessible model id."
        )
        return 1

    if result.telemetry.model_substituted:
        print(
            f"PREFLIGHT: FAILED -- model substitution detected. Requested={REQUESTED_MODEL!r}, "
            f"returned={result.telemetry.returned_models!r}."
        )
        return 1

    print(f"PREFLIGHT: OK -- model={REQUESTED_MODEL!r} reachable, no fallback, no substitution.")
    print(f"  capability_id={result.capability_id!r} reason={result.reason!r}")
    print(f"  latency_seconds={result.telemetry.latency_seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
