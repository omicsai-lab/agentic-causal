"""
Confusion matrices for router predictions against gold capability_id,
across the full 7-capability registry (not just the 2 supported ones),
since distractor capabilities (binary_edrip, hello_world,
linear_regression, logistic_regression, summary_stats) remain live router
options and can appear as predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.semantic_orchestration.analysis.stats import confusion_matrix

ALL_CAPABILITY_IDS = [
    "binary_edrip",
    "causal_ate",
    "hello_world",
    "linear_regression",
    "logistic_regression",
    "summary_stats",
    "survival_adjusted_curves",
]


def router_confusion_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold = [r["gold_capability_id"] for r in rows]
    pred = [r["predicted_capability_id"] for r in rows]
    return confusion_matrix(gold, pred, label_order=list(ALL_CAPABILITY_IDS))


def router_confusion_matrix_by_repeat(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rep in sorted({r["repeat_index"] for r in rows}):
        rep_rows = [r for r in rows if r["repeat_index"] == rep]
        out[str(rep)] = router_confusion_matrix(rep_rows)
    return out


def challenge_prediction_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """For challenge prompts (no single gold capability_id), report the
    distribution of predicted capability_id per challenge_type."""
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        ct = r["challenge_type"]
        pred = r["predicted_capability_id"]
        out.setdefault(ct, {})
        out[ct][pred] = out[ct].get(pred, 0) + 1
    return out
