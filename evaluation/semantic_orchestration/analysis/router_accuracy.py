"""
Router accuracy analysis: overall / per-capability / per-category accuracy
with Wilson CIs, computed per repeat and pooled across the 3 repeats.

Consumes evaluation/semantic_orchestration/results/raw/router_supported.jsonl
(one row per (prompt_id, repeat_index), schema defined in
harness/run_router_eval.py).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from evaluation.semantic_orchestration.analysis.stats import wilson_ci


def _cell_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for r in rows if r["predicted_capability_id"] == r["gold_capability_id"])
    p_hat, lo, hi = wilson_ci(n_correct, n) if n else (float("nan"), float("nan"), float("nan"))
    return {"n": n, "n_correct": n_correct, "accuracy": p_hat, "wilson_ci_lower": lo, "wilson_ci_upper": hi}


def router_accuracy_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    `rows` = raw router_supported.jsonl rows (all repeats pooled).

    Returns a dict with:
      - overall (pooled across all repeats, n=360 if complete)
      - by_capability (causal_ate / survival_adjusted_curves)
      - by_category (6 categories, pooled across capability)
      - by_capability_category (12 cells)
      - by_repeat (repeat 1/2/3 separately, overall accuracy each)
      - fallback_rate (overall and by repeat)
    """
    overall = _cell_accuracy(rows)

    by_capability: Dict[str, Any] = {}
    for cap in sorted({r["capability"] for r in rows}):
        by_capability[cap] = _cell_accuracy([r for r in rows if r["capability"] == cap])

    by_category: Dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        by_category[cat] = _cell_accuracy([r for r in rows if r["category"] == cat])

    by_capability_category: Dict[str, Any] = {}
    for cap in sorted({r["capability"] for r in rows}):
        for cat in sorted({r["category"] for r in rows}):
            cell_rows = [r for r in rows if r["capability"] == cap and r["category"] == cat]
            if cell_rows:
                by_capability_category[f"{cap}::{cat}"] = _cell_accuracy(cell_rows)

    by_repeat: Dict[str, Any] = {}
    for rep in sorted({r["repeat_index"] for r in rows}):
        by_repeat[str(rep)] = _cell_accuracy([r for r in rows if r["repeat_index"] == rep])

    n_fallback = sum(1 for r in rows if r.get("fallback_detected"))
    fallback_rate = n_fallback / len(rows) if rows else float("nan")

    n_api_failed = sum(1 for r in rows if not r.get("api_call_succeeded") and not r.get("fallback_detected"))

    return {
        "overall": overall,
        "by_capability": by_capability,
        "by_category": by_category,
        "by_capability_category": by_capability_category,
        "by_repeat": by_repeat,
        "fallback_rate": fallback_rate,
        "n_fallback": n_fallback,
        "n_unexplained_api_failure": n_api_failed,
    }


def confusion_pairs(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count (gold -> predicted) confusion pairs, most useful restricted to
    incorrect predictions, e.g. how often causal_ate gold prompts get
    routed to binary_edrip."""
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        gold, pred = r["gold_capability_id"], r["predicted_capability_id"]
        if gold != pred:
            counts[f"{gold} -> {pred}"] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
