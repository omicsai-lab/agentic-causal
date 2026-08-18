"""
PRIMARY manuscript comparison: GPT-5.4 planner recommended-tool
performance (prompt-level majority vote across the 3 planner repeats) vs.
the repository's existing deterministic `_fallback_plan()` baseline, on
the same 120 supported prompts.

This is intentionally separate from `analysis/agreement.py`
(planner-vs-router agreement/McNemar), which compares the planner against
the router -- a different pair of components. Do not conflate the two.

Inputs:
  - `results/raw/planner_supported.jsonl` (GPT-5.4 planner, 3 repeats x
    120 prompts = 360 rows) -- live results, read-only.
  - `results/processed/deterministic_baseline.json` (already computed by
    `harness/run_deterministic_baseline.py`, itself making zero network
    calls) -- read as-is, not recomputed here.
  - `benchmark/prompts_supported.jsonl` (frozen benchmark) -- for gold
    labels, capability, and category, read-only.

Majority vote and all statistics reuse the existing primitives in
`analysis/stats.py` unchanged (wilson_ci, exact_mcnemar,
paired_bootstrap_ci, majority_vote) -- no statistical formula is altered
or reimplemented here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.semantic_orchestration.analysis.stats import (
    BOOTSTRAP_SEED,
    exact_mcnemar,
    majority_vote,
    paired_bootstrap_ci,
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


def gpt_majority_by_prompt(planner_rows: List[Dict[str, Any]], field: str = "recommended_tool") -> Dict[str, Dict[str, Any]]:
    """
    Collapse the 3 planner repeats per prompt into one majority-vote label
    per prompt, using the exact `majority_vote()` primitive (first-
    occurrence tiebreak, flagged non-clean on a tie).
    """
    by_prompt: Dict[str, Dict[int, Any]] = {}
    for r in planner_rows:
        by_prompt.setdefault(r["prompt_id"], {})[r["repeat_index"]] = r["plan"].get(field)

    out: Dict[str, Dict[str, Any]] = {}
    for pid, reps in by_prompt.items():
        labels = [reps[k] for k in sorted(reps.keys())]
        winner, is_clean = majority_vote(labels)
        out[pid] = {"majority_label": winner, "is_clean_majority": is_clean, "n_repeats": len(labels), "repeat_labels": labels}
    return out


def build_paired_items(
    planner_rows: List[Dict[str, Any]],
    deterministic_planner_rows: List[Dict[str, Any]],
    supported_prompts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One row per supported prompt (n=120), pairing the GPT-5.4 planner's
    majority-vote `recommended_tool` correctness against the deterministic
    `_fallback_plan()` baseline's `recommended_tool` correctness for the
    SAME prompt -- the matched-pairs structure McNemar and the paired
    bootstrap both require.
    """
    gold_by_id = {p["id"]: p["gold_capability_id"] for p in supported_prompts}
    capability_by_id = {p["id"]: p["capability"] for p in supported_prompts}
    category_by_id = {p["id"]: p["category"] for p in supported_prompts}

    gpt_majority = gpt_majority_by_prompt(planner_rows, field="recommended_tool")
    det_by_id = {r["prompt_id"]: r for r in deterministic_planner_rows}

    missing_gpt = set(gold_by_id) - set(gpt_majority)
    missing_det = set(gold_by_id) - set(det_by_id)
    if missing_gpt:
        raise ValueError(f"planner_supported.jsonl missing prompt_ids: {sorted(missing_gpt)}")
    if missing_det:
        raise ValueError(f"deterministic_baseline.json missing prompt_ids: {sorted(missing_det)}")

    paired: List[Dict[str, Any]] = []
    for pid in sorted(gold_by_id.keys()):
        gold = gold_by_id[pid]
        gpt_label = gpt_majority[pid]["majority_label"]
        det_row = det_by_id[pid]
        paired.append(
            {
                "prompt_id": pid,
                "capability": capability_by_id[pid],
                "category": category_by_id[pid],
                "gold_capability_id": gold,
                "gpt_majority_label": gpt_label,
                "gpt_majority_correct": gpt_label == gold,
                "gpt_majority_is_clean": gpt_majority[pid]["is_clean_majority"],
                "gpt_repeat_labels": gpt_majority[pid]["repeat_labels"],
                "deterministic_label": det_row.get("recommended_tool"),
                "deterministic_correct": bool(det_row.get("recommended_tool_correct")),
            }
        )
    return paired


def _summarize(paired_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(paired_subset)
    gpt_correct = [1.0 if p["gpt_majority_correct"] else 0.0 for p in paired_subset]
    det_correct = [1.0 if p["deterministic_correct"] else 0.0 for p in paired_subset]

    n_gpt = int(sum(gpt_correct))
    n_det = int(sum(det_correct))

    gpt_p, gpt_lo, gpt_hi = wilson_ci(n_gpt, n) if n else (float("nan"),) * 3
    det_p, det_lo, det_hi = wilson_ci(n_det, n) if n else (float("nan"),) * 3

    diff_pp = 100.0 * ((n_gpt / n) - (n_det / n)) if n else float("nan")

    b = sum(1 for g, d in zip(gpt_correct, det_correct) if g == 1.0 and d == 0.0)
    c = sum(1 for g, d in zip(gpt_correct, det_correct) if g == 0.0 and d == 1.0)
    mcnemar = exact_mcnemar(b, c)

    diffs = [g - d for g, d in zip(gpt_correct, det_correct)]
    bootstrap = paired_bootstrap_ci(diffs, seed=BOOTSTRAP_SEED)

    return {
        "n": n,
        "gpt_majority_vote_accuracy": {"n": n, "n_correct": n_gpt, "accuracy": gpt_p, "wilson_ci_lower": gpt_lo, "wilson_ci_upper": gpt_hi},
        "deterministic_baseline_accuracy": {"n": n, "n_correct": n_det, "accuracy": det_p, "wilson_ci_lower": det_lo, "wilson_ci_upper": det_hi},
        "absolute_difference_percentage_points": diff_pp,
        "mcnemar": mcnemar,
        "paired_bootstrap_accuracy_diff": bootstrap,
    }


def planner_majority_vs_deterministic_report(paired: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _summarize(paired)
    by_capability = {cap: _summarize([p for p in paired if p["capability"] == cap]) for cap in CAPABILITIES}
    by_category = {cat: _summarize([p for p in paired if p["category"] == cat]) for cat in CATEGORIES}

    return {
        "description": (
            "GPT-5.4 planner recommended_tool, prompt-level majority vote across "
            "3 repeats, vs. the repository's existing deterministic _fallback_plan() "
            "baseline, on the same 120 supported prompts. NOT the same comparison as "
            "planner-vs-router agreement/McNemar (see analysis/agreement.py)."
        ),
        "overall": overall,
        "by_capability": by_capability,
        "by_category": by_category,
        "paired_items": paired,
    }
