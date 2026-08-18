from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.challenge_descriptive import (  # noqa: E402
    challenge_report,
    planner_challenge_detail,
    router_challenge_detail,
)


def _router_rows():
    return [
        {"prompt_id": "c1", "repeat_index": 1, "challenge_type": "underspecified", "predicted_capability_id": "causal_ate", "reason": "r1", "fallback_detected": False, "api_call_succeeded": True, "returned_models": ["gpt-5.4-2026-03-05"], "expected_behavior": "eb1"},
        {"prompt_id": "c2", "repeat_index": 1, "challenge_type": "unsupported_method", "predicted_capability_id": "causal_ate", "reason": "r2", "fallback_detected": False, "api_call_succeeded": True, "returned_models": ["gpt-5.4-2026-03-05"], "expected_behavior": "eb2"},
    ]


def _planner_rows():
    return [
        {"prompt_id": "c1", "repeat_index": 1, "challenge_type": "underspecified", "plan": {"recommended_tool": "causal_ate", "task_type": "causal_effect", "outcome_type": "unknown", "reasoning": "x"}, "likely_fallback_by_content_match": False, "api_call_succeeded": True, "returned_models": ["gpt-5.4-2026-03-05"], "expected_behavior": "eb1"},
        {"prompt_id": "c2", "repeat_index": 1, "challenge_type": "unsupported_method", "plan": {"recommended_tool": "causal_ate", "task_type": "causal_effect", "outcome_type": "unknown", "reasoning": "y"}, "likely_fallback_by_content_match": False, "api_call_succeeded": True, "returned_models": ["gpt-5.4-2026-03-05"], "expected_behavior": "eb2"},
    ]


def test_router_challenge_detail_shape():
    detail = router_challenge_detail(_router_rows())
    assert len(detail) == 2
    assert detail[0]["prompt_id"] == "c1"
    assert set(detail[0].keys()) >= {"prompt_id", "challenge_type", "predicted_capability_id", "reason"}


def test_planner_challenge_detail_extracts_plan_fields():
    detail = planner_challenge_detail(_planner_rows())
    assert detail[0]["recommended_tool"] == "causal_ate"
    assert detail[0]["task_type"] == "causal_effect"


def test_challenge_report_has_no_accuracy_field_and_states_descriptive_only():
    report = challenge_report(_router_rows(), _planner_rows())
    assert "accuracy" not in json_keys_recursive(report)
    assert "descriptive" in report["note"].lower()
    assert "no accuracy" in report["note"].lower()
    assert "abstention" in report["note"].lower()


def test_challenge_report_distributions_and_counts():
    report = challenge_report(_router_rows(), _planner_rows())
    assert report["n_router_calls"] == 2
    assert report["n_planner_calls"] == 2
    assert report["router_predicted_capability_distribution_by_challenge_type"]["underspecified"]["causal_ate"] == 1
    assert report["planner_recommended_tool_distribution_by_challenge_type"]["unsupported_method"]["causal_ate"] == 1


def json_keys_recursive(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= json_keys_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= json_keys_recursive(item)
    return keys
