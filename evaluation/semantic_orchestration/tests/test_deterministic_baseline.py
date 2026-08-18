from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.run_deterministic_baseline import (  # noqa: E402
    run_planner_baseline,
    run_router_baseline,
)
from evaluation.semantic_orchestration.harness.io_utils import load_supported_prompts  # noqa: E402


def test_router_deterministic_fallback_always_predicts_binary_edrip():
    prompts = load_supported_prompts()[:10]
    rows = run_router_baseline(prompts)
    assert len(rows) == 10
    assert all(r["predicted_capability_id"] == "binary_edrip" for r in rows)
    # Gold labels are always causal_ate/survival_adjusted_curves, never
    # binary_edrip, so the deterministic router fallback has 0% accuracy
    # on the supported set -- this is the expected, documented result.
    assert all(r["correct"] is False for r in rows)


def test_planner_deterministic_fallback_gets_some_explicit_method_prompts_right():
    """
    _fallback_plan() is a real keyword-based rule engine (checks for
    'survival'/'kaplan'/... and 'causal effect'/'treatment effect'/'ate').
    explicit_method category prompts are specifically designed to name the
    method plainly, so at least some should trigger the correct keyword
    rule even without any LLM call.
    """
    prompts = [p for p in load_supported_prompts() if p["category"] == "explicit_method"]
    rows = run_planner_baseline(prompts)
    assert len(rows) == 20
    n_correct = sum(1 for r in rows if r["recommended_tool_correct"])
    assert n_correct > 0, "expected at least some explicit_method prompts to trigger the keyword-based fallback plan correctly"


def test_planner_deterministic_fallback_is_pure_and_reproducible():
    prompts = load_supported_prompts()[:5]
    rows1 = run_planner_baseline(prompts)
    rows2 = run_planner_baseline(prompts)
    assert rows1 == rows2
