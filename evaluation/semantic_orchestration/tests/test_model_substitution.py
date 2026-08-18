"""
Regression tests for the preflight/telemetry model-substitution check
(harness/model_client.py::is_same_model, CallTelemetry.model_substituted).

Requested model "gpt-5.4" must accept itself and any dated snapshot
("gpt-5.4-YYYY-MM-DD"), but reject other model-family suffixes
("-mini", "-nano", "-pro") and unrelated model families entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.model_client import (  # noqa: E402
    CallTelemetry,
    is_same_model,
)


def test_exact_match_is_same_model():
    assert is_same_model("gpt-5.4", "gpt-5.4") is True


def test_dated_snapshot_is_same_model():
    assert is_same_model("gpt-5.4", "gpt-5.4-2026-03-05") is True
    assert is_same_model("gpt-5.4", "gpt-5.4-2025-12-31") is True


def test_mini_nano_pro_variants_are_substitution():
    assert is_same_model("gpt-5.4", "gpt-5.4-mini") is False
    assert is_same_model("gpt-5.4", "gpt-5.4-nano") is False
    assert is_same_model("gpt-5.4", "gpt-5.4-pro") is False


def test_other_model_families_are_substitution():
    assert is_same_model("gpt-5.4", "gpt-4o") is False
    assert is_same_model("gpt-5.4", "gpt-4o-mini") is False
    assert is_same_model("gpt-5.4", "gpt-5.4.1") is False
    assert is_same_model("gpt-5.4", "gpt-5.40") is False


def test_malformed_dated_suffix_is_substitution():
    # Not a valid YYYY-MM-DD shape -> not treated as a dated snapshot.
    assert is_same_model("gpt-5.4", "gpt-5.4-2026-3-5") is False
    assert is_same_model("gpt-5.4", "gpt-5.4-20260305") is False
    assert is_same_model("gpt-5.4", "gpt-5.4-preview") is False


def test_empty_requested_or_returned_falls_back_to_equality():
    assert is_same_model("", "") is True
    assert is_same_model("gpt-5.4", "") is False
    assert is_same_model("", "gpt-5.4") is False


def test_call_telemetry_model_substituted_property_uses_same_model_logic():
    t = CallTelemetry(requested_model="gpt-5.4", returned_models=["gpt-5.4-2026-03-05"])
    assert t.model_substituted is False

    t2 = CallTelemetry(requested_model="gpt-5.4", returned_models=["gpt-5.4-mini"])
    assert t2.model_substituted is True

    t3 = CallTelemetry(requested_model="gpt-5.4", returned_models=[])
    assert t3.model_substituted is False

    t4 = CallTelemetry(requested_model="gpt-5.4", returned_models=["gpt-5.4", "gpt-5.4-2026-01-01"])
    assert t4.model_substituted is False

    t5 = CallTelemetry(requested_model="gpt-5.4", returned_models=["gpt-5.4", "gpt-5.4-nano"])
    assert t5.model_substituted is True
