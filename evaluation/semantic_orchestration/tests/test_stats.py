from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.stats import (  # noqa: E402
    confusion_matrix,
    exact_mcnemar,
    majority_vote,
    paired_bootstrap_ci,
    pairwise_agreement,
    wilson_ci,
)


def test_wilson_ci_matches_known_value():
    # n=100, x=50 -> Wilson interval is a well-known textbook example,
    # approximately [0.404, 0.596] for a 95% CI.
    p_hat, lo, hi = wilson_ci(50, 100)
    assert p_hat == 0.5
    assert abs(lo - 0.404) < 0.005
    assert abs(hi - 0.596) < 0.005


def test_wilson_ci_bounds_are_within_0_1():
    p_hat, lo, hi = wilson_ci(120, 120)
    assert 0.0 <= lo <= hi <= 1.0
    p_hat, lo, hi = wilson_ci(0, 120)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_zero_n_returns_nan():
    p_hat, lo, hi = wilson_ci(0, 0)
    assert p_hat != p_hat  # NaN


def test_exact_mcnemar_symmetric_case_gives_p_value_1():
    result = exact_mcnemar(5, 5)
    assert result["n_discordant"] == 10
    assert abs(result["p_value"] - 1.0) < 1e-9


def test_exact_mcnemar_extreme_case_gives_small_p_value():
    # All discordant pairs favor one side: b=20, c=0 should be significant.
    result = exact_mcnemar(20, 0)
    assert result["p_value"] < 0.001


def test_exact_mcnemar_no_discordant_pairs():
    result = exact_mcnemar(0, 0)
    assert result["p_value"] == 1.0
    assert result["n_discordant"] == 0


def test_paired_bootstrap_ci_is_deterministic_given_seed():
    diffs = [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    r1 = paired_bootstrap_ci(diffs, n_resamples=500, seed=20260818)
    r2 = paired_bootstrap_ci(diffs, n_resamples=500, seed=20260818)
    assert r1 == r2


def test_paired_bootstrap_ci_different_seed_can_differ():
    diffs = [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    r1 = paired_bootstrap_ci(diffs, n_resamples=500, seed=1)
    r2 = paired_bootstrap_ci(diffs, n_resamples=500, seed=2)
    assert r1["ci_lower"] != r2["ci_lower"] or r1["ci_upper"] != r2["ci_upper"]


def test_paired_bootstrap_ci_mean_matches_simple_mean():
    diffs = [1.0, 1.0, 1.0, -1.0]
    r = paired_bootstrap_ci(diffs, n_resamples=1000, seed=20260818)
    assert abs(r["mean_diff"] - 0.5) < 1e-9


def test_confusion_matrix_basic():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    cm = confusion_matrix(gold, pred, label_order=["a", "b"])
    assert cm["matrix"]["a"]["a"] == 1
    assert cm["matrix"]["a"]["b"] == 1
    assert cm["matrix"]["b"]["b"] == 2
    assert cm["matrix"]["b"]["a"] == 0


def test_pairwise_agreement_all_agree():
    runs = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
    r = pairwise_agreement(runs)
    assert r["mean_pairwise_agreement"] == 1.0
    assert r["full_consistency_rate"] == 1.0


def test_pairwise_agreement_partial():
    runs = [["a", "b"], ["a", "x"], ["a", "b"]]
    r = pairwise_agreement(runs)
    # item 0: all agree ("a"), item 1: "b","x","b" -> not all agree
    assert r["full_consistency_rate"] == 0.5
    assert 0.0 < r["mean_pairwise_agreement"] < 1.0


def test_majority_vote_clean_majority():
    winner, is_clean = majority_vote(["a", "a", "b"])
    assert winner == "a"
    assert is_clean is True


def test_majority_vote_tie_not_clean():
    winner, is_clean = majority_vote(["a", "b"])
    assert is_clean is False
    assert winner == "a"  # first-occurrence tiebreak, deterministic
