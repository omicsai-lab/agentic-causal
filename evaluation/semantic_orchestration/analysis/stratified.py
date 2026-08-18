"""
Prompt-level (majority-vote) accuracy, 3-run consistency, and confusion
matrices, stratified by gold capability and by linguistic category, for
both the router and the planner. Generalizes the overall-only majority
vote in `analysis/consistency.py` and the run-level-only confusion matrix
in `analysis/confusion.py` without modifying either -- everything here
composes the same unchanged primitives from `analysis/stats.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from evaluation.semantic_orchestration.analysis.confusion import ALL_CAPABILITY_IDS
from evaluation.semantic_orchestration.analysis.stats import (
    confusion_matrix,
    majority_vote,
    pairwise_agreement,
    wilson_ci,
)

CATEGORIES = [
    "explicit_method",
    "formal_estimand",
    "biomedical_domain",
    "indirect_colloquial",
    "noisy_verbose",
    "near_boundary",
]
CAPABILITIES = ["causal_ate", "survival_adjusted_curves"]

LabelFn = Callable[[Dict[str, Any]], Any]


def label_matrix_by_prompt(rows: List[Dict[str, Any]], label_fn: LabelFn) -> Dict[str, Dict[int, Any]]:
    out: Dict[str, Dict[int, Any]] = {}
    for r in rows:
        out.setdefault(r["prompt_id"], {})[r["repeat_index"]] = label_fn(r)
    return out


def majority_labels_by_prompt(rows: List[Dict[str, Any]], label_fn: LabelFn) -> Dict[str, Tuple[Any, bool]]:
    by_prompt = label_matrix_by_prompt(rows, label_fn)
    out: Dict[str, Tuple[Any, bool]] = {}
    for pid, reps in by_prompt.items():
        labels = [reps[k] for k in sorted(reps.keys())]
        out[pid] = majority_vote(labels)
    return out


def _cell_accuracy(n_correct: int, n: int) -> Dict[str, Any]:
    p_hat, lo, hi = wilson_ci(n_correct, n) if n else (float("nan"),) * 3
    return {"n": n, "n_correct": n_correct, "accuracy": p_hat, "wilson_ci_lower": lo, "wilson_ci_upper": hi}


def majority_vote_accuracy_stratified(
    rows: List[Dict[str, Any]],
    label_fn: LabelFn,
    gold_by_id: Dict[str, str],
    capability_by_id: Dict[str, str],
    category_by_id: Dict[str, str],
) -> Dict[str, Any]:
    """
    Prompt-level majority-vote accuracy: overall, by gold capability, and
    by linguistic category (each prompt contributes exactly once, using
    its 3-repeat majority label).
    """
    majorities = majority_labels_by_prompt(rows, label_fn)

    def _accuracy_for(prompt_ids: List[str]) -> Dict[str, Any]:
        n = len(prompt_ids)
        n_correct = sum(1 for pid in prompt_ids if majorities[pid][0] == gold_by_id[pid])
        return _cell_accuracy(n_correct, n)

    all_ids = sorted(majorities.keys())
    overall = _accuracy_for(all_ids)
    by_capability = {
        cap: _accuracy_for([pid for pid in all_ids if capability_by_id.get(pid) == cap]) for cap in CAPABILITIES
    }
    by_category = {
        cat: _accuracy_for([pid for pid in all_ids if category_by_id.get(pid) == cat]) for cat in CATEGORIES
    }

    n_clean = sum(1 for pid in all_ids if majorities[pid][1])
    return {
        "overall": overall,
        "by_capability": by_capability,
        "by_category": by_category,
        "clean_majority_rate": n_clean / len(all_ids) if all_ids else float("nan"),
    }


def consistency_stratified(
    rows: List[Dict[str, Any]],
    label_fn: LabelFn,
    capability_by_id: Dict[str, str],
    category_by_id: Dict[str, str],
) -> Dict[str, Any]:
    """
    3-run pairwise agreement + full (3/3) consistency rate: overall, by
    gold capability, and by linguistic category.
    """
    by_prompt = label_matrix_by_prompt(rows, label_fn)
    repeats_present = sorted({rep for reps in by_prompt.values() for rep in reps})
    complete_prompts = [pid for pid, reps in by_prompt.items() if set(reps.keys()) == set(repeats_present)]

    def _agreement_for(prompt_ids: List[str]) -> Dict[str, Any]:
        if len(repeats_present) < 2 or not prompt_ids:
            return {"n_prompts": len(prompt_ids)}
        runs = [[by_prompt[pid][rep] for pid in prompt_ids] for rep in repeats_present]
        agreement = pairwise_agreement(runs)
        agreement["n_prompts"] = len(prompt_ids)
        return agreement

    overall = _agreement_for(complete_prompts)
    by_capability = {
        cap: _agreement_for([pid for pid in complete_prompts if capability_by_id.get(pid) == cap]) for cap in CAPABILITIES
    }
    by_category = {
        cat: _agreement_for([pid for pid in complete_prompts if category_by_id.get(pid) == cat]) for cat in CATEGORIES
    }

    return {
        "repeats_present": repeats_present,
        "n_prompts_with_all_repeats": len(complete_prompts),
        "overall": overall,
        "by_capability": by_capability,
        "by_category": by_category,
    }


def majority_vote_confusion_matrix(
    rows: List[Dict[str, Any]],
    label_fn: LabelFn,
    gold_by_id: Dict[str, str],
) -> Dict[str, Any]:
    """Confusion matrix over the 120 prompts using each prompt's 3-repeat
    majority label, rather than the 360 individual repeat-level rows."""
    majorities = majority_labels_by_prompt(rows, label_fn)
    prompt_ids = sorted(majorities.keys())
    gold = [gold_by_id[pid] for pid in prompt_ids]
    pred = [majorities[pid][0] for pid in prompt_ids]
    return confusion_matrix(gold, pred, label_order=list(ALL_CAPABILITY_IDS))
