"""
Three-run consistency analysis for both router and planner: pairwise
agreement across the 3 repeats, full (3/3) consistency rate, and majority
voting with the resulting majority-vote accuracy compared to per-run
accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.semantic_orchestration.analysis.stats import (
    majority_vote,
    pairwise_agreement,
    wilson_ci,
)


def _label_matrix_by_prompt(rows: List[Dict[str, Any]], label_fn) -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = {}
    for r in rows:
        out.setdefault(r["prompt_id"], {})[r["repeat_index"]] = label_fn(r)
    return out


def router_consistency_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_prompt = _label_matrix_by_prompt(rows, lambda r: r["predicted_capability_id"])
    gold_by_prompt = {r["prompt_id"]: r["gold_capability_id"] for r in rows}

    repeats_present = sorted({rep for reps in by_prompt.values() for rep in reps})
    runs = []
    prompt_order = sorted(by_prompt.keys())
    complete_prompts = [pid for pid in prompt_order if set(by_prompt[pid].keys()) == set(repeats_present)]

    for rep in repeats_present:
        runs.append([by_prompt[pid][rep] for pid in complete_prompts])

    agreement = pairwise_agreement(runs) if len(runs) >= 2 else {}

    majority_correct = 0
    majority_clean = 0
    for pid in complete_prompts:
        labels = [by_prompt[pid][rep] for rep in repeats_present]
        winner, is_clean = majority_vote(labels)
        if winner == gold_by_prompt[pid]:
            majority_correct += 1
        if is_clean:
            majority_clean += 1

    n = len(complete_prompts)
    p_hat, lo, hi = wilson_ci(majority_correct, n) if n else (float("nan"),) * 3

    return {
        "n_prompts_with_all_repeats": n,
        "repeats_present": repeats_present,
        "agreement": agreement,
        "majority_vote_accuracy": {"n": n, "n_correct": majority_correct, "accuracy": p_hat, "wilson_ci_lower": lo, "wilson_ci_upper": hi},
        "majority_vote_clean_rate": majority_clean / n if n else float("nan"),
    }


def planner_consistency_report(rows: List[Dict[str, Any]], field: str = "recommended_tool") -> Dict[str, Any]:
    by_prompt = _label_matrix_by_prompt(rows, lambda r: r["plan"].get(field))
    gold_by_prompt = {r["prompt_id"]: r["gold_capability_id"] for r in rows}

    repeats_present = sorted({rep for reps in by_prompt.values() for rep in reps})
    prompt_order = sorted(by_prompt.keys())
    complete_prompts = [pid for pid in prompt_order if set(by_prompt[pid].keys()) == set(repeats_present)]

    runs = [[by_prompt[pid][rep] for pid in complete_prompts] for rep in repeats_present]
    agreement = pairwise_agreement(runs) if len(runs) >= 2 else {}

    majority_correct = 0
    for pid in complete_prompts:
        labels = [by_prompt[pid][rep] for rep in repeats_present]
        winner, _ = majority_vote(labels)
        if winner == gold_by_prompt[pid]:
            majority_correct += 1

    n = len(complete_prompts)
    p_hat, lo, hi = wilson_ci(majority_correct, n) if n else (float("nan"),) * 3

    return {
        "field": field,
        "n_prompts_with_all_repeats": n,
        "repeats_present": repeats_present,
        "agreement": agreement,
        "majority_vote_accuracy": {"n": n, "n_correct": majority_correct, "accuracy": p_hat, "wilson_ci_lower": lo, "wilson_ci_upper": hi},
    }
