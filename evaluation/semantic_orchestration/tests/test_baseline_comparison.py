"""
Tests for the PRIMARY manuscript comparison: GPT-5.4 planner majority
vote vs. the deterministic _fallback_plan() baseline. Uses a small
synthetic paired dataset with known correctness counts so the McNemar b/c
and bootstrap mean can be checked by hand, independent of the real live
results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.baseline_comparison import (  # noqa: E402
    _summarize,
    build_paired_items,
    gpt_majority_by_prompt,
    planner_majority_vs_deterministic_report,
)


def _mk_planner_rows(prompt_id, labels, capability="causal_ate", category="explicit_method", gold="causal_ate"):
    return [
        {
            "prompt_id": prompt_id,
            "repeat_index": i + 1,
            "capability": capability,
            "category": category,
            "gold_capability_id": gold,
            "plan": {"recommended_tool": label},
        }
        for i, label in enumerate(labels)
    ]


def test_gpt_majority_by_prompt_picks_majority_label():
    rows = _mk_planner_rows("p1", ["causal_ate", "causal_ate", "binary_edrip"])
    out = gpt_majority_by_prompt(rows)
    assert out["p1"]["majority_label"] == "causal_ate"
    assert out["p1"]["is_clean_majority"] is True
    assert out["p1"]["repeat_labels"] == ["causal_ate", "causal_ate", "binary_edrip"]


def test_build_paired_items_matches_prompt_count_and_raises_on_missing():
    planner_rows = _mk_planner_rows("p1", ["causal_ate", "causal_ate", "causal_ate"]) + _mk_planner_rows(
        "p2", ["survival_adjusted_curves"] * 3, capability="survival_adjusted_curves", category="near_boundary", gold="survival_adjusted_curves"
    )
    det_rows = [
        {"prompt_id": "p1", "recommended_tool": "causal_ate", "recommended_tool_correct": True},
        {"prompt_id": "p2", "recommended_tool": "summary_stats", "recommended_tool_correct": False},
    ]
    supported_prompts = [
        {"id": "p1", "gold_capability_id": "causal_ate", "capability": "causal_ate", "category": "explicit_method"},
        {"id": "p2", "gold_capability_id": "survival_adjusted_curves", "capability": "survival_adjusted_curves", "category": "near_boundary"},
    ]

    paired = build_paired_items(planner_rows, det_rows, supported_prompts)
    assert len(paired) == 2
    p1 = next(p for p in paired if p["prompt_id"] == "p1")
    assert p1["gpt_majority_correct"] is True
    assert p1["deterministic_correct"] is True
    p2 = next(p for p in paired if p["prompt_id"] == "p2")
    assert p2["gpt_majority_correct"] is True
    assert p2["deterministic_correct"] is False

    # Missing a prompt from det_rows must raise, not silently drop it.
    incomplete_supported = supported_prompts + [
        {"id": "p3", "gold_capability_id": "causal_ate", "capability": "causal_ate", "category": "explicit_method"}
    ]
    try:
        build_paired_items(planner_rows, det_rows, incomplete_supported)
        assert False, "expected ValueError for missing prompt_id"
    except ValueError:
        pass


def test_summarize_known_mcnemar_and_bootstrap():
    # 4 items: gpt correct in all 4, det correct in only 2 -> b=2, c=0.
    paired = [
        {"gpt_majority_correct": True, "deterministic_correct": True},
        {"gpt_majority_correct": True, "deterministic_correct": True},
        {"gpt_majority_correct": True, "deterministic_correct": False},
        {"gpt_majority_correct": True, "deterministic_correct": False},
    ]
    summary = _summarize(paired)
    assert summary["n"] == 4
    assert summary["gpt_majority_vote_accuracy"]["n_correct"] == 4
    assert summary["deterministic_baseline_accuracy"]["n_correct"] == 2
    assert summary["absolute_difference_percentage_points"] == 50.0
    assert summary["mcnemar"]["b"] == 2
    assert summary["mcnemar"]["c"] == 0
    assert summary["paired_bootstrap_accuracy_diff"]["mean_diff"] == 0.5
    assert summary["paired_bootstrap_accuracy_diff"]["seed"] == 20260818


def test_report_overall_matches_sum_of_by_capability():
    planner_rows = (
        _mk_planner_rows("p1", ["causal_ate"] * 3)
        + _mk_planner_rows("p2", ["binary_edrip"] * 3)  # wrong on purpose
        + _mk_planner_rows("p3", ["survival_adjusted_curves"] * 3, capability="survival_adjusted_curves", category="near_boundary", gold="survival_adjusted_curves")
    )
    det_rows = [
        {"prompt_id": "p1", "recommended_tool": "causal_ate", "recommended_tool_correct": True},
        {"prompt_id": "p2", "recommended_tool": "causal_ate", "recommended_tool_correct": True},
        {"prompt_id": "p3", "recommended_tool": "causal_ate", "recommended_tool_correct": False},
    ]
    supported_prompts = [
        {"id": "p1", "gold_capability_id": "causal_ate", "capability": "causal_ate", "category": "explicit_method"},
        {"id": "p2", "gold_capability_id": "causal_ate", "capability": "causal_ate", "category": "explicit_method"},
        {"id": "p3", "gold_capability_id": "survival_adjusted_curves", "capability": "survival_adjusted_curves", "category": "near_boundary"},
    ]
    paired = build_paired_items(planner_rows, det_rows, supported_prompts)
    report = planner_majority_vs_deterministic_report(paired)

    assert report["overall"]["n"] == 3
    n_cap_sum = sum(c["n"] for c in report["by_capability"].values())
    assert n_cap_sum == 3
    assert report["by_capability"]["causal_ate"]["n"] == 2
    assert report["by_capability"]["survival_adjusted_curves"]["n"] == 1
    assert "description" in report
    # The description must explicitly disclaim conflation with the
    # separate planner-vs-router comparison in analysis/agreement.py.
    assert "planner-vs-router" in report["description"]
    assert "_fallback_plan" in report["description"]
