"""
Guard against silent drift in the frozen capability registry and the two
router/planner fallback code paths that AUDIT.md documents. If any of
these tests fail, the audit and/or harness/config.py fallback-marker list
must be re-checked and updated together -- do not just patch the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.semantic_orchestration.harness.config import (  # noqa: E402
    ROUTER_FALLBACK_CAPABILITY_ID,
    ROUTER_FALLBACK_REASON_MARKERS,
)

EXPECTED_CAPABILITY_IDS = {
    "binary_edrip",
    "causal_ate",
    "hello_world",
    "linear_regression",
    "logistic_regression",
    "summary_stats",
    "survival_adjusted_curves",
}

SUPPORTED_CAPABILITY_IDS = {"causal_ate", "survival_adjusted_curves"}


def test_capability_registry_has_exactly_seven_expected_ids():
    from src.agent.router_llm import load_capabilities

    caps = load_capabilities()
    ids = {c.get("capability_id") for c in caps}
    assert ids == EXPECTED_CAPABILITY_IDS, f"capability registry drifted: {ids}"


def test_router_alphabetical_fallback_id_is_binary_edrip():
    """
    router_llm.llm_choose_capability's internal fallback_id = allowed[0],
    where `allowed` is derived from sorted(cap_dir.glob('cap_*.json')).
    This must remain 'binary_edrip' -- if a new cap_*.json file sorts
    before it alphabetically, the deterministic-fallback baseline results
    in EVALUATION_REPORT.md / MANUSCRIPT_FACTS.md become stale.
    """
    from src.agent.router_llm import load_capabilities, _capability_ids

    caps = load_capabilities()
    allowed = _capability_ids(caps)
    assert allowed[0] == ROUTER_FALLBACK_CAPABILITY_ID


def test_supported_capabilities_are_subset_of_registry():
    assert SUPPORTED_CAPABILITY_IDS.issubset(EXPECTED_CAPABILITY_IDS)


def test_router_fallback_reason_markers_still_present_in_source():
    """
    Golden-string check against the actual router_llm.py source text (not
    just behavior), so a wording change in the fallback reason strings is
    caught even if we haven't re-run a live (costly) evaluation recently.
    """
    src_path = REPO_ROOT / "src" / "agent" / "router_llm.py"
    source = src_path.read_text(encoding="utf-8")
    for marker in ROUTER_FALLBACK_REASON_MARKERS:
        assert marker in source, f"fallback marker no longer found in router_llm.py source: {marker!r}"


def test_binary_edrip_capability_json_lists_covariates_optional_but_tool_requires_it():
    """
    AUDIT.md, Finding 5: cap_binary_edrip.json lists 'covariates' as
    optional, but BinaryEDRIPTool.validate() rejects a request with no
    covariates. This is a pre-existing registry/tool inconsistency in the
    frozen application; this test documents and pins it rather than
    "fixing" application code that is out of scope for this evaluation.
    """
    import json

    cap_path = REPO_ROOT / "src" / "agent" / "capabilities" / "cap_binary_edrip.json"
    spec = json.loads(cap_path.read_text(encoding="utf-8"))
    assert "covariates" in spec.get("optional_fields", [])
    assert "covariates" not in spec.get("required_fields", [])

    tool_src = (REPO_ROOT / "src" / "agent" / "tools" / "tool_binary_edrip.py").read_text(encoding="utf-8")
    assert "binary_edrip requires covariates" in tool_src


def test_causal_ate_and_binary_edrip_capability_jsons_share_doubly_robust_keyword():
    """
    AUDIT.md, Finding 2: pins the exact keyword overlap the benchmark
    design works around (see BINARY_EDRIP_OVERLAP_TERMS in
    prompts_supported.py).
    """
    import json

    causal_ate_spec = json.loads((REPO_ROOT / "src" / "agent" / "capabilities" / "cap_causal_ate.json").read_text(encoding="utf-8"))
    binary_edrip_spec = json.loads((REPO_ROOT / "src" / "agent" / "capabilities" / "cap_binary_edrip.json").read_text(encoding="utf-8"))

    assert "doubly robust" in causal_ate_spec.get("keywords", [])
    assert "doubly robust" in binary_edrip_spec.get("keywords", [])


def test_planner_fallback_plan_has_no_explicit_fallback_marker():
    """
    AUDIT.md, Finding 4: planner_llm._fallback_plan()'s returned dict has
    no field distinguishing it from a genuine model response, unlike the
    router's `reason` string. If this ever changes (e.g. a `source` or
    `is_fallback` key is added), harness/model_client.py's
    likely_fallback_by_content_match heuristic should be revisited.
    """
    from src.agent.planner_llm import _fallback_plan

    plan = _fallback_plan("estimate the average treatment effect of drug on outcome")
    assert set(plan.keys()) == {
        "analysis_goal",
        "task_type",
        "target_estimand",
        "outcome_type",
        "recommended_tool",
        "required_fields",
        "optional_fields",
        "assumptions",
        "reasoning",
    }
    assert "fallback" not in plan  # no explicit marker key
    assert "is_fallback" not in plan
    assert "source" not in plan
