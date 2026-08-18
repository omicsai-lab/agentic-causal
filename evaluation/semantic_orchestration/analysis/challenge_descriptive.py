"""
Descriptive-only analysis of the 30 challenge prompts.

IMPORTANT (see EVALUATION_PROTOCOL.md, "Challenge set scoring", and
AUDIT.md): no challenge prompt has a single correct gold capability_id by
design -- `underspecified` prompts admit more than one defensible
capability, `unsupported_method` prompts have no correct capability at
all, and `out_of_scope_mixed` prompts are unrelated or mixed-intent. This
module therefore computes distributions and per-prompt detail only. It
does NOT compute, and must never be read as implying, an accuracy,
abstention-rate, or safety-benchmark score for the challenge set. The
challenge set is descriptive only, not an abstention/safety benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, List


def router_challenge_detail(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "prompt_id": r["prompt_id"],
                "challenge_type": r["challenge_type"],
                "predicted_capability_id": r.get("predicted_capability_id"),
                "reason": r.get("reason"),
                "fallback_detected": r.get("fallback_detected"),
                "api_call_succeeded": r.get("api_call_succeeded"),
                "returned_models": r.get("returned_models"),
                "expected_behavior": r.get("expected_behavior"),
            }
        )
    return sorted(out, key=lambda x: x["prompt_id"])


def planner_challenge_detail(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        plan = r.get("plan", {}) or {}
        out.append(
            {
                "prompt_id": r["prompt_id"],
                "challenge_type": r["challenge_type"],
                "recommended_tool": plan.get("recommended_tool"),
                "task_type": plan.get("task_type"),
                "outcome_type": plan.get("outcome_type"),
                "reasoning": plan.get("reasoning"),
                "likely_fallback_by_content_match": r.get("likely_fallback_by_content_match"),
                "api_call_succeeded": r.get("api_call_succeeded"),
                "returned_models": r.get("returned_models"),
                "expected_behavior": r.get("expected_behavior"),
            }
        )
    return sorted(out, key=lambda x: x["prompt_id"])


def _distribution(rows: List[Dict[str, Any]], group_field: str, value_fn) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        g = r[group_field]
        v = str(value_fn(r))
        out.setdefault(g, {})
        out[g][v] = out[g].get(v, 0) + 1
    return out


def challenge_report(router_rows: List[Dict[str, Any]], planner_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    router_detail = router_challenge_detail(router_rows)
    planner_detail = planner_challenge_detail(planner_rows)

    router_dist = _distribution(router_rows, "challenge_type", lambda r: r.get("predicted_capability_id"))
    planner_dist = _distribution(planner_rows, "challenge_type", lambda r: (r.get("plan") or {}).get("recommended_tool"))

    n_router_fallback = sum(1 for r in router_rows if r.get("fallback_detected"))
    n_planner_likely_fallback = sum(1 for r in planner_rows if r.get("likely_fallback_by_content_match"))
    n_router_api_fail = sum(1 for r in router_rows if not r.get("api_call_succeeded"))
    n_planner_api_fail = sum(1 for r in planner_rows if not r.get("api_call_succeeded"))

    return {
        "note": (
            "Descriptive only -- no challenge prompt has a single correct gold "
            "capability_id by design, so no accuracy/abstention/safety score is "
            "computed here. See EVALUATION_PROTOCOL.md 'Challenge set scoring'."
        ),
        "n_router_calls": len(router_rows),
        "n_planner_calls": len(planner_rows),
        "n_router_fallback": n_router_fallback,
        "n_router_api_failure": n_router_api_fail,
        "n_planner_likely_fallback_by_content_match": n_planner_likely_fallback,
        "n_planner_api_failure": n_planner_api_fail,
        "router_predicted_capability_distribution_by_challenge_type": router_dist,
        "planner_recommended_tool_distribution_by_challenge_type": planner_dist,
        "router_detail": router_detail,
        "planner_detail": planner_detail,
    }
