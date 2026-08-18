"""
Planner structured-field accuracy: how often the planner's JSON output
matches the gold `recommended_tool` (== gold_capability_id),
`task_type` (causal_ate -> "causal_effect", survival_adjusted_curves ->
"survival"), and `outcome_type` (from prompts_supported.jsonl's
gold_outcome_type field).

Consumes results/raw/planner_supported.jsonl (schema defined in
harness/run_planner_eval.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.semantic_orchestration.analysis.stats import wilson_ci

GOLD_TASK_TYPE = {
    "causal_ate": "causal_effect",
    "survival_adjusted_curves": "survival",
}


def _field_accuracy(rows: List[Dict[str, Any]], correct_fn) -> Dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for r in rows if correct_fn(r))
    p_hat, lo, hi = wilson_ci(n_correct, n) if n else (float("nan"), float("nan"), float("nan"))
    return {"n": n, "n_correct": n_correct, "accuracy": p_hat, "wilson_ci_lower": lo, "wilson_ci_upper": hi}


def _recommended_tool_correct(r: Dict[str, Any]) -> bool:
    return r["plan"].get("recommended_tool") == r["gold_capability_id"]


def _task_type_correct(r: Dict[str, Any]) -> bool:
    return r["plan"].get("task_type") == GOLD_TASK_TYPE.get(r["gold_capability_id"])


def _outcome_type_correct(r: Dict[str, Any]) -> bool:
    gold = r.get("gold_outcome_type")
    if gold is None:
        return False
    return r["plan"].get("outcome_type") == gold


def planner_accuracy_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = {
        "recommended_tool": _recommended_tool_correct,
        "task_type": _task_type_correct,
        "outcome_type": _outcome_type_correct,
    }

    overall_by_field = {name: _field_accuracy(rows, fn) for name, fn in fields.items()}

    def all_fields_correct(r: Dict[str, Any]) -> bool:
        return all(fn(r) for fn in fields.values())

    overall_all_fields = _field_accuracy(rows, all_fields_correct)

    by_capability: Dict[str, Any] = {}
    for cap in sorted({r["capability"] for r in rows}):
        cap_rows = [r for r in rows if r["capability"] == cap]
        by_capability[cap] = {
            name: _field_accuracy(cap_rows, fn) for name, fn in fields.items()
        }

    by_category: Dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        by_category[cat] = {
            name: _field_accuracy(cat_rows, fn) for name, fn in fields.items()
        }

    n_likely_fallback = sum(1 for r in rows if r.get("likely_fallback_by_content_match"))

    return {
        "overall_by_field": overall_by_field,
        "overall_all_fields_correct": overall_all_fields,
        "by_capability": by_capability,
        "by_category": by_category,
        "n_likely_fallback_by_content_match": n_likely_fallback,
        "likely_fallback_rate": n_likely_fallback / len(rows) if rows else float("nan"),
    }
