"""
Regression test for the figure-generation yerr bug: matplotlib requires
non-negative error-bar magnitudes, but floating-point rounoff in the
Wilson CI computation can produce a bound that differs from the point
estimate by a tiny negative amount at the 0%/100% boundary (e.g.
wilson_ci_upper = 0.9999999999999998 when accuracy = 1.0). See
analysis/figures.py::_yerr_magnitude.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.figures import (  # noqa: E402
    _yerr_magnitude,
    plot_accuracy_by_category,
)


def test_yerr_magnitude_clamps_tiny_negative_rounoff():
    assert _yerr_magnitude(-2.220446049250313e-16) == 0.0


def test_yerr_magnitude_preserves_genuine_positive_distance():
    assert _yerr_magnitude(0.0630) == 0.0630


def test_yerr_magnitude_zero_stays_zero():
    assert _yerr_magnitude(0.0) == 0.0


def test_plot_accuracy_by_category_handles_100pct_wilson_rounoff(tmp_path):
    """
    Reproduces the exact real-world case: accuracy == 1.0 exactly, while
    the computed wilson_ci_upper is a hair below 1.0 due to floating-point
    rounoff, which previously made (wilson_ci_upper - accuracy) a tiny
    negative number and crashed matplotlib's ax.bar(..., yerr=...).
    """
    report = {
        "by_category": {
            "explicit_method": {
                "n": 60,
                "n_correct": 60,
                "accuracy": 1.0,
                "wilson_ci_lower": 0.93982814785791,
                "wilson_ci_upper": 0.9999999999999998,  # < 1.0 due to rounoff
            },
            "indirect_colloquial": {
                "n": 60,
                "n_correct": 57,
                "accuracy": 0.95,
                "wilson_ci_lower": 0.8629948352365148,
                "wilson_ci_upper": 0.9828504978356043,
            },
        }
    }
    out_path = tmp_path / "router_accuracy_by_category.png"
    plot_accuracy_by_category(report, out_path, "test")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
