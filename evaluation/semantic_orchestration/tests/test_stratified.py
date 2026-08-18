from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.stratified import (  # noqa: E402
    consistency_stratified,
    majority_vote_accuracy_stratified,
    majority_vote_confusion_matrix,
)


def _rows(prompt_id, labels, capability, category):
    return [{"prompt_id": prompt_id, "repeat_index": i + 1, "predicted_capability_id": label} for i, label in enumerate(labels)]


def _label_fn(r):
    return r["predicted_capability_id"]


def _fixture():
    rows = (
        _rows("p1", ["causal_ate", "causal_ate", "causal_ate"], "causal_ate", "explicit_method")
        + _rows("p2", ["binary_edrip", "causal_ate", "causal_ate"], "causal_ate", "explicit_method")  # majority still causal_ate
        + _rows("p3", ["survival_adjusted_curves"] * 3, "survival_adjusted_curves", "near_boundary")
        + _rows("p4", ["causal_ate", "causal_ate", "survival_adjusted_curves"], "survival_adjusted_curves", "near_boundary")  # majority wrong
    )
    gold_by_id = {
        "p1": "causal_ate",
        "p2": "causal_ate",
        "p3": "survival_adjusted_curves",
        "p4": "survival_adjusted_curves",
    }
    capability_by_id = {"p1": "causal_ate", "p2": "causal_ate", "p3": "survival_adjusted_curves", "p4": "survival_adjusted_curves"}
    category_by_id = {"p1": "explicit_method", "p2": "explicit_method", "p3": "near_boundary", "p4": "near_boundary"}
    return rows, gold_by_id, capability_by_id, category_by_id


def test_majority_vote_accuracy_stratified_overall_and_by_capability():
    rows, gold_by_id, capability_by_id, category_by_id = _fixture()
    report = majority_vote_accuracy_stratified(rows, _label_fn, gold_by_id, capability_by_id, category_by_id)

    assert report["overall"]["n"] == 4
    assert report["overall"]["n_correct"] == 3  # p1, p2, p3 correct; p4 majority is causal_ate != gold

    assert report["by_capability"]["causal_ate"]["n"] == 2
    assert report["by_capability"]["causal_ate"]["n_correct"] == 2
    assert report["by_capability"]["survival_adjusted_curves"]["n"] == 2
    assert report["by_capability"]["survival_adjusted_curves"]["n_correct"] == 1

    assert report["by_category"]["explicit_method"]["n_correct"] == 2
    assert report["by_category"]["near_boundary"]["n_correct"] == 1
    # categories not present in this fixture must still report n=0, not KeyError
    assert report["by_category"]["noisy_verbose"]["n"] == 0


def test_consistency_stratified_flags_p2_and_p4_as_non_unanimous():
    rows, gold_by_id, capability_by_id, category_by_id = _fixture()
    report = consistency_stratified(rows, _label_fn, capability_by_id, category_by_id)

    assert report["n_prompts_with_all_repeats"] == 4
    # p1 and p3 unanimous; p2 and p4 are not -> full_consistency_rate == 0.5
    assert report["overall"]["full_consistency_rate"] == 0.5


def test_majority_vote_confusion_matrix_counts_p4_as_survival_to_causal_ate():
    rows, gold_by_id, capability_by_id, category_by_id = _fixture()
    cm = majority_vote_confusion_matrix(rows, _label_fn, gold_by_id)
    assert cm["matrix"]["survival_adjusted_curves"]["causal_ate"] == 1
    assert cm["matrix"]["survival_adjusted_curves"]["survival_adjusted_curves"] == 1
    assert cm["matrix"]["causal_ate"]["causal_ate"] == 2
