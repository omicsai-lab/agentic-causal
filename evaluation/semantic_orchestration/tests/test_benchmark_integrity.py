"""
Static integrity checks on the frozen benchmark. These must all pass
before BENCHMARK_MANIFEST.json is written (see benchmark/build_benchmark.py
and the freeze step in AUDIT.md / EVALUATION_PROTOCOL.md).
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"

from evaluation.semantic_orchestration.benchmark.prompts_supported import (  # noqa: E402
    ALL_SUPPORTED_PROMPTS,
    CATEGORIES,
    CAUSAL_ATE_PROMPTS,
    SURVIVAL_PROMPTS,
    BINARY_EDRIP_OVERLAP_TERMS,
)
from evaluation.semantic_orchestration.benchmark.prompts_challenge import (  # noqa: E402
    ALL_CHALLENGE_PROMPTS,
    CHALLENGE_TYPES,
    UNDERSPECIFIED_PROMPTS,
    UNSUPPORTED_METHOD_PROMPTS,
    OUT_OF_SCOPE_MIXED_PROMPTS,
)

NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.70


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# --------------------------------------------------------------------------
# Supported set
# --------------------------------------------------------------------------

def test_supported_prompt_counts():
    assert len(ALL_SUPPORTED_PROMPTS) == 120
    assert len(CAUSAL_ATE_PROMPTS) == 60
    assert len(SURVIVAL_PROMPTS) == 60


def test_supported_category_cell_counts_are_exactly_ten():
    for capability, prompts in [("causal_ate", CAUSAL_ATE_PROMPTS), ("survival_adjusted_curves", SURVIVAL_PROMPTS)]:
        for cat in CATEGORIES:
            n = sum(1 for p in prompts if p["category"] == cat)
            assert n == 10, f"{capability}/{cat} has {n} prompts, expected 10"


def test_supported_categories_match_spec():
    assert CATEGORIES == [
        "explicit_method",
        "formal_estimand",
        "biomedical_domain",
        "indirect_colloquial",
        "noisy_verbose",
        "near_boundary",
    ]


def test_supported_gold_capability_id_matches_capability():
    for p in ALL_SUPPORTED_PROMPTS:
        assert p["gold_capability_id"] == p["capability"]
        assert p["gold_capability_id"] in ("causal_ate", "survival_adjusted_curves")


def test_supported_prompt_ids_are_unique():
    ids = [p["id"] for p in ALL_SUPPORTED_PROMPTS]
    assert len(ids) == len(set(ids))


def test_supported_ids_encode_capability_and_category():
    for p in ALL_SUPPORTED_PROMPTS:
        assert p["category"] in p["id"]
        cap_token = "causal_ate" if p["capability"] == "causal_ate" else "survival"
        assert p["id"].startswith(f"sup_{cap_token}_{p['category']}_")


def test_no_causal_ate_prompt_uses_binary_edrip_overlap_language():
    """
    See AUDIT.md, Finding 2: causal_ate and binary_edrip share several
    keywords (e.g. "doubly robust"). The genuine overlap zone is a
    *binary* outcome combined with robustness/flexibility/semiparametric
    framing. No supported prompt (either capability) may use that framing
    language at all, since it is reserved for describing binary_edrip.
    """
    for p in ALL_SUPPORTED_PROMPTS:
        text_lower = p["text"].lower()
        for term in BINARY_EDRIP_OVERLAP_TERMS:
            assert term not in text_lower, f"{p['id']} contains reserved overlap term {term!r}"


def test_no_supported_prompt_uses_reserved_binary_edrip_terms():
    for p in ALL_SUPPORTED_PROMPTS:
        text_lower = p["text"].lower()
        assert "edrip" not in text_lower, p["id"]
        assert "semiparametric" not in text_lower, p["id"]


def test_no_near_duplicate_supported_prompts():
    prompts = [(p["id"], _tokens(p["text"])) for p in ALL_SUPPORTED_PROMPTS]
    violations = []
    for (id1, t1), (id2, t2) in itertools.combinations(prompts, 2):
        j = len(t1 & t2) / len(t1 | t2)
        if j > NEAR_DUPLICATE_JACCARD_THRESHOLD:
            violations.append((j, id1, id2))
    assert not violations, f"near-duplicate prompt pairs found: {violations}"


def test_survival_prompts_do_not_request_single_summary_effect_except_near_boundary():
    """
    Sanity check on the survival near_boundary design: those 10 prompts
    deliberately mention 'treatment effect' language while still asking
    for curves, to test the boundary against causal_ate. No other
    survival category should mention 'average treatment effect' at all,
    to keep gold labels unambiguous outside the near_boundary cell.
    """
    for p in SURVIVAL_PROMPTS:
        if p["category"] == "near_boundary":
            continue
        assert "average treatment effect" not in p["text"].lower()


# --------------------------------------------------------------------------
# Challenge set
# --------------------------------------------------------------------------

def test_challenge_prompt_counts():
    assert len(ALL_CHALLENGE_PROMPTS) == 30
    assert len(UNDERSPECIFIED_PROMPTS) == 10
    assert len(UNSUPPORTED_METHOD_PROMPTS) == 10
    assert len(OUT_OF_SCOPE_MIXED_PROMPTS) == 10


def test_challenge_types_match_spec():
    assert CHALLENGE_TYPES == ["underspecified", "unsupported_method", "out_of_scope_mixed"]


def test_challenge_prompts_have_no_forced_gold_capability():
    for p in ALL_CHALLENGE_PROMPTS:
        assert "gold_capability_id" not in p or p.get("gold_capability_id") is None
        assert p.get("expected_behavior")


def test_challenge_prompt_ids_are_unique():
    ids = [p["id"] for p in ALL_CHALLENGE_PROMPTS]
    assert len(ids) == len(set(ids))


def test_unsupported_method_prompts_do_not_describe_binary_edrip():
    """
    binary_edrip IS a registered, implemented capability. The
    unsupported_method challenge prompts must describe methods that are
    NOT implemented by any of the 7 registered capabilities, so they must
    not accidentally describe binary_edrip (doubly robust / semiparametric
    estimation for a binary outcome).
    """
    for p in UNSUPPORTED_METHOD_PROMPTS:
        text_lower = p["text"].lower()
        assert "edrip" not in text_lower
        assert not ("doubly robust" in text_lower and "binary" in text_lower)


def test_no_near_duplicate_challenge_prompts():
    prompts = [(p["id"], _tokens(p["text"])) for p in ALL_CHALLENGE_PROMPTS]
    violations = []
    for (id1, t1), (id2, t2) in itertools.combinations(prompts, 2):
        j = len(t1 & t2) / len(t1 | t2)
        if j > NEAR_DUPLICATE_JACCARD_THRESHOLD:
            violations.append((j, id1, id2))
    assert not violations, f"near-duplicate challenge prompt pairs found: {violations}"


# --------------------------------------------------------------------------
# Frozen JSONL matches the literal source data exactly (rebuild is a no-op)
# --------------------------------------------------------------------------

def test_frozen_jsonl_matches_rebuilt_output():
    sys.path.insert(0, str(BENCHMARK_DIR))
    import build_benchmark  # type: ignore

    expected_supported = "\n".join(
        json.dumps(build_benchmark._ordered(p, build_benchmark.SUPPORTED_FIELD_ORDER), ensure_ascii=False)
        for p in ALL_SUPPORTED_PROMPTS
    )
    actual_supported = (BENCHMARK_DIR / "prompts_supported.jsonl").read_text(encoding="utf-8").strip()

    # gold_outcome_type defaulting for survival prompts must match what
    # build_benchmark.py applies.
    expected_lines = []
    for p in ALL_SUPPORTED_PROMPTS:
        entry = build_benchmark._ordered(p, build_benchmark.SUPPORTED_FIELD_ORDER)
        entry.setdefault(
            "gold_outcome_type",
            "time_to_event" if entry["capability"] == "survival_adjusted_curves" else "unknown",
        )
        expected_lines.append(json.dumps(entry, ensure_ascii=False))
    expected_supported = "\n".join(expected_lines)

    assert actual_supported == expected_supported, "prompts_supported.jsonl is stale; re-run build_benchmark.py"

    expected_lines = []
    for p in ALL_CHALLENGE_PROMPTS:
        entry = build_benchmark._ordered(p, build_benchmark.CHALLENGE_FIELD_ORDER)
        entry.setdefault("gold_capability_id", None)
        expected_lines.append(json.dumps(entry, ensure_ascii=False))
    expected_challenge = "\n".join(expected_lines)
    actual_challenge = (BENCHMARK_DIR / "prompts_challenge.jsonl").read_text(encoding="utf-8").strip()

    assert actual_challenge == expected_challenge, "prompts_challenge.jsonl is stale; re-run build_benchmark.py"


def test_jsonl_files_have_expected_line_counts():
    sup_lines = (BENCHMARK_DIR / "prompts_supported.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chal_lines = (BENCHMARK_DIR / "prompts_challenge.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sup_lines) == 120
    assert len(chal_lines) == 30
    for line in sup_lines + chal_lines:
        json.loads(line)  # must be valid JSON
