"""
Run the GPT-5.4 router and planner against the 30 challenge prompts, 1
repeat each (no consistency/repeat analysis is meaningful for the
challenge set -- see EVALUATION_PROTOCOL.md, "Challenge set scoring").

Not run as part of Step 1. Fully implemented for Step 2.

Usage:
    python3 -m evaluation.semantic_orchestration.harness.run_challenge_eval [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.config import (  # noqa: E402
    CHALLENGE_REPEATS,
    REQUESTED_MODEL,
)
from evaluation.semantic_orchestration.harness.io_utils import (  # noqa: E402
    RESULTS_RAW_DIR,
    JsonlWriter,
    already_completed_keys,
    load_challenge_prompts,
)
from evaluation.semantic_orchestration.harness.model_client import (  # noqa: E402
    call_planner,
    call_router,
)


def run(
    router_out_path: Path,
    planner_out_path: Path,
    limit: int | None = None,
    model: str = REQUESTED_MODEL,
) -> tuple[int, int]:
    prompts = load_challenge_prompts()
    if limit is not None:
        prompts = prompts[:limit]

    router_done = already_completed_keys(router_out_path, ("prompt_id", "repeat_index"))
    planner_done = already_completed_keys(planner_out_path, ("prompt_id", "repeat_index"))

    n_router = 0
    n_planner = 0

    with JsonlWriter(router_out_path) as router_writer, JsonlWriter(planner_out_path) as planner_writer:
        for p in prompts:
            for repeat_index in range(1, CHALLENGE_REPEATS + 1):
                key = (p["id"], repeat_index)

                if key not in router_done:
                    r = call_router(prompt_id=p["id"], request_text=p["text"], repeat_index=repeat_index, model=model)
                    router_writer.write(
                        {
                            "prompt_id": p["id"],
                            "repeat_index": repeat_index,
                            "challenge_type": p["challenge_type"],
                            "expected_behavior": p["expected_behavior"],
                            "predicted_capability_id": r.capability_id,
                            "reason": r.reason,
                            "fallback_detected": r.fallback_detected,
                            "fallback_marker": r.fallback_marker,
                            "harness_error": r.harness_error,
                            "requested_model": model,
                            "api_call_succeeded": r.telemetry.api_call_succeeded,
                            "api_exception_type": r.telemetry.api_exception_type,
                            "api_exception": r.telemetry.api_exception,
                            "returned_models": r.telemetry.returned_models,
                            "model_substituted": r.telemetry.model_substituted,
                            "latency_seconds": r.telemetry.latency_seconds,
                        }
                    )
                    n_router += 1

                if key not in planner_done:
                    pl = call_planner(prompt_id=p["id"], request_text=p["text"], repeat_index=repeat_index, model=model)
                    planner_writer.write(
                        {
                            "prompt_id": p["id"],
                            "repeat_index": repeat_index,
                            "challenge_type": p["challenge_type"],
                            "expected_behavior": p["expected_behavior"],
                            "plan": pl.plan,
                            "likely_fallback_by_content_match": pl.likely_fallback_by_content_match,
                            "harness_error": pl.harness_error,
                            "requested_model": model,
                            "api_call_succeeded": pl.telemetry.api_call_succeeded,
                            "api_exception_type": pl.telemetry.api_exception_type,
                            "api_exception": pl.telemetry.api_exception,
                            "returned_models": pl.telemetry.returned_models,
                            "model_substituted": pl.telemetry.model_substituted,
                            "latency_seconds": pl.telemetry.latency_seconds,
                        }
                    )
                    n_planner += 1

    return n_router, n_planner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--router-out", type=str, default=str(RESULTS_RAW_DIR / "router_challenge.jsonl"))
    ap.add_argument("--planner-out", type=str, default=str(RESULTS_RAW_DIR / "planner_challenge.jsonl"))
    ap.add_argument("--model", type=str, default=REQUESTED_MODEL)
    args = ap.parse_args()

    n_router, n_planner = run(Path(args.router_out), Path(args.planner_out), limit=args.limit, model=args.model)
    print(f"Wrote {n_router} router-challenge records, {n_planner} planner-challenge records.")


if __name__ == "__main__":
    main()
