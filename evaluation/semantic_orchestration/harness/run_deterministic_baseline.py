"""
Deterministic, non-LLM baseline evaluation. No network calls, no API key
required, safe to run as part of Step 1 (benchmark freeze).

What this measures
-------------------
Both `router_llm.llm_choose_capability` and `planner_llm.
llm_generate_analysis_plan` have an internal, fully deterministic fallback
path that fires whenever no usable OpenAI API key is available (see
AUDIT.md, Findings 3-4):

  - router_llm.llm_choose_capability: returns the alphabetically-first
    registered capability_id ("binary_edrip") with a fixed reason string,
    without ever constructing an OpenAI client.
  - planner_llm.llm_generate_analysis_plan: `OpenAI()` raises inside its
    try/except (no api_key), so it falls through to the pure, rule-based
    `_fallback_plan(request)` function.

This script calls the REAL production functions (not a reimplementation),
after temporarily removing OPENAI_API_KEY from a copy of the environment
for the duration of the calls (restored immediately after), so the result
is reproducible regardless of whether the ambient environment happens to
have a real key configured. This is the "deterministic `_fallback_plan()`
baseline evaluation" requested for Step 1, generalized to also cover the
router's own internal deterministic fallback.

Usage:
    python3 -m evaluation.semantic_orchestration.harness.run_deterministic_baseline
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.io_utils import (  # noqa: E402
    RESULTS_PROCESSED_DIR,
    load_challenge_prompts,
    load_supported_prompts,
)


@contextmanager
def _no_api_key_env() -> Iterator[None]:
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        yield
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original


def run_router_baseline(prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from src.agent import router_llm

    rows = []
    with _no_api_key_env():
        for p in prompts:
            out = router_llm.llm_choose_capability(request=p["text"], csv_columns=None)
            rows.append(
                {
                    "prompt_id": p["id"],
                    "gold_capability_id": p.get("gold_capability_id"),
                    "predicted_capability_id": out.get("capability_id"),
                    "reason": out.get("reason"),
                    "correct": out.get("capability_id") == p.get("gold_capability_id"),
                }
            )
    return rows


def run_planner_baseline(prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from src.agent import planner_llm

    rows = []
    with _no_api_key_env():
        for p in prompts:
            plan = planner_llm.llm_generate_analysis_plan(request=p["text"])
            gold = p.get("gold_capability_id")
            rows.append(
                {
                    "prompt_id": p["id"],
                    "gold_capability_id": gold,
                    "recommended_tool": plan.get("recommended_tool"),
                    "task_type": plan.get("task_type"),
                    "outcome_type": plan.get("outcome_type"),
                    "gold_outcome_type": p.get("gold_outcome_type"),
                    "recommended_tool_correct": plan.get("recommended_tool") == gold,
                }
            )
    return rows


def summarize(rows: List[Dict[str, Any]], correct_field: str) -> Dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get(correct_field))
    return {"n": n, "n_correct": n_correct, "accuracy": (n_correct / n) if n else float("nan")}


def main() -> None:
    supported = load_supported_prompts()
    challenge = load_challenge_prompts()

    router_supported_rows = run_router_baseline(supported)
    planner_supported_rows = run_planner_baseline(supported)

    # Challenge prompts have no gold_capability_id by design; we still
    # record what the deterministic fallback produces, for descriptive
    # reporting only (see BENCHMARK_REVIEW.md).
    router_challenge_rows = run_router_baseline(challenge)
    planner_challenge_rows = run_planner_baseline(challenge)

    out = {
        "router_supported": {
            "summary": summarize(router_supported_rows, "correct"),
            "rows": router_supported_rows,
        },
        "planner_supported": {
            "summary": summarize(planner_supported_rows, "recommended_tool_correct"),
            "rows": planner_supported_rows,
        },
        "router_challenge_predicted_capability_distribution": _distribution(
            router_challenge_rows, "predicted_capability_id"
        ),
        "planner_challenge_recommended_tool_distribution": _distribution(
            planner_challenge_rows, "recommended_tool"
        ),
    }

    out_path = RESULTS_PROCESSED_DIR / "deterministic_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Router deterministic-fallback accuracy on 120 supported prompts: "
          f"{out['router_supported']['summary']['n_correct']}/{out['router_supported']['summary']['n']} "
          f"= {out['router_supported']['summary']['accuracy']:.4f}")
    print(f"Planner deterministic-fallback (_fallback_plan) accuracy on 120 supported prompts: "
          f"{out['planner_supported']['summary']['n_correct']}/{out['planner_supported']['summary']['n']} "
          f"= {out['planner_supported']['summary']['accuracy']:.4f}")
    print(f"Wrote {out_path}")


def _distribution(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for r in rows:
        v = str(r.get(field))
        dist[v] = dist.get(v, 0) + 1
    return dist


if __name__ == "__main__":
    main()
