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

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.config import REQUESTED_MODEL  # noqa: E402
from evaluation.semantic_orchestration.harness.model_client import call_router, is_same_model  # noqa: E402

PREFLIGHT_PROMPT = "Estimate the average treatment effect of drug A vs placebo on outcome in trial.csv."

PROVENANCE_MODEL_RESOLUTION_PATH = Path(__file__).resolve().parents[1] / "provenance" / "model_resolution.json"


def _record_model_resolution_provenance(result) -> Path:
    """
    Persist exactly which model snapshot the API returned (e.g.
    "gpt-5.4-2026-03-05") for `REQUESTED_MODEL`, independent of the
    pass/fail verdict, so the resolved snapshot is on record even for a
    borderline or failing preflight.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": REQUESTED_MODEL,
        "returned_models": result.telemetry.returned_models,
        "model_substituted": result.telemetry.model_substituted,
        "per_returned_model": [
            {"returned": m, "same_model_as_requested": is_same_model(REQUESTED_MODEL, m)}
            for m in result.telemetry.returned_models
        ],
    }
    PROVENANCE_MODEL_RESOLUTION_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PROVENANCE_MODEL_RESOLUTION_PATH


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("PREFLIGHT: SKIPPED -- OPENAI_API_KEY is not set in this environment.")
        print("No API call was made. The 780-call GPT-5.4 evaluation cannot run until a key is configured.")
        return 2

    result = call_router(prompt_id="preflight", request_text=PREFLIGHT_PROMPT, repeat_index=0, model=REQUESTED_MODEL)

    if result.telemetry.attempted and result.telemetry.returned_models:
        prov_path = _record_model_resolution_provenance(result)
        print(f"  model resolution recorded: {prov_path} (returned_models={result.telemetry.returned_models!r})")

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
