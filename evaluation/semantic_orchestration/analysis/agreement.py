"""
Planner-router agreement, exact McNemar comparison between planner and
router correctness, and paired-bootstrap CI on the accuracy difference.

"Agreement" here means: for the same (prompt_id, repeat_index), does the
planner's `recommended_tool` equal the router's `predicted_capability_id`?
This is orthogonal to gold-label correctness -- two components can agree
with each other while both being wrong, or disagree while one is right.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evaluation.semantic_orchestration.analysis.stats import (
    BOOTSTRAP_SEED,
    exact_mcnemar,
    paired_bootstrap_ci,
    wilson_ci,
)


def _index_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {(r["prompt_id"], r["repeat_index"]): r for r in rows}


def planner_router_agreement(router_rows: List[Dict[str, Any]], planner_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    router_by_key = _index_by_key(router_rows)
    planner_by_key = _index_by_key(planner_rows)

    common_keys = sorted(set(router_by_key) & set(planner_by_key))
    n = len(common_keys)
    n_agree = 0
    for key in common_keys:
        rr = router_by_key[key]
        pr = planner_by_key[key]
        if rr["predicted_capability_id"] == pr["plan"].get("recommended_tool"):
            n_agree += 1

    p_hat, lo, hi = wilson_ci(n_agree, n) if n else (float("nan"),) * 3

    return {
        "n_matched_calls": n,
        "n_agree": n_agree,
        "agreement_rate": p_hat,
        "wilson_ci_lower": lo,
        "wilson_ci_upper": hi,
    }


def planner_vs_router_mcnemar(
    router_rows: List[Dict[str, Any]],
    planner_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Exact McNemar's test comparing router correctness vs planner
    (recommended_tool) correctness on matched (prompt_id, repeat_index)
    items, plus a paired bootstrap CI (seed=20260818) on the accuracy
    difference (router_accuracy - planner_accuracy).
    """
    router_by_key = _index_by_key(router_rows)
    planner_by_key = _index_by_key(planner_rows)
    common_keys = sorted(set(router_by_key) & set(planner_by_key))

    b = 0  # router correct, planner incorrect
    c = 0  # router incorrect, planner correct
    diffs: List[float] = []

    for key in common_keys:
        rr = router_by_key[key]
        pr = planner_by_key[key]
        router_correct = rr["predicted_capability_id"] == rr["gold_capability_id"]
        planner_correct = pr["plan"].get("recommended_tool") == pr["gold_capability_id"]

        if router_correct and not planner_correct:
            b += 1
        elif not router_correct and planner_correct:
            c += 1

        diffs.append(float(router_correct) - float(planner_correct))

    mcnemar = exact_mcnemar(b, c)
    bootstrap = paired_bootstrap_ci(diffs, seed=BOOTSTRAP_SEED)

    return {
        "n_matched_calls": len(common_keys),
        "mcnemar": mcnemar,
        "bootstrap_accuracy_diff": bootstrap,
    }
