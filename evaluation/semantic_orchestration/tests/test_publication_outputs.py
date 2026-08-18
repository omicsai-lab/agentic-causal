"""
Tests for the generic CSV/Markdown/LaTeX row-table writers
(analysis/tables.py) and the PNG+PDF+CSV figure saving
(analysis/figures.py::_save_figure_all_formats), plus a guarded
end-to-end check that `analysis.run_all` produces the new finalized
sections when the real live results are present on disk.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.analysis.tables import (  # noqa: E402
    write_csv_table,
    write_latex_table,
    write_markdown_table,
    write_table_all_formats,
)
from evaluation.semantic_orchestration.analysis.figures import (  # noqa: E402
    _save_figure_all_formats,
    plot_baseline_comparison,
)
from evaluation.semantic_orchestration.harness.io_utils import RESULTS_PROCESSED_DIR, RESULTS_RAW_DIR  # noqa: E402

ROWS = [
    {"cell": "overall", "n": 120, "accuracy": 0.9833},
    {"cell": "causal_ate", "n": 60, "accuracy": 0.9667},
]
FIELDS = ["cell", "n", "accuracy"]


def test_write_csv_table_roundtrips(tmp_path):
    path = tmp_path / "t.csv"
    write_csv_table(ROWS, path, FIELDS)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["cell"] == "overall"
    assert rows[1]["n"] == "60"


def test_write_markdown_table_has_header_and_rows(tmp_path):
    path = tmp_path / "t.md"
    write_markdown_table(ROWS, path, FIELDS)
    text = path.read_text(encoding="utf-8")
    assert "| cell | n | accuracy |" in text
    assert "| overall | 120 | 0.9833 |" in text


def test_write_latex_table_has_tabular_and_escapes_underscore(tmp_path):
    path = tmp_path / "t.tex"
    write_latex_table([{"cell": "causal_ate", "n": 60, "accuracy": 0.9667}], path, FIELDS, caption="My_Caption")
    text = path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in text
    assert "causal\\_ate" in text
    assert "My\\_Caption" in text


def test_write_latex_table_caption_is_not_double_escaped_if_caller_passes_raw_underscore(tmp_path):
    r"""
    Regression test: analysis/publication_tables.py previously pre-escaped
    underscores in some caption strings (e.g. "recommended\\_tool") before
    passing them to write_latex_table, which itself calls _latex_escape()
    on the caption -- corrupting "\_" into "\textbackslash{}\_". Callers
    must pass raw text; write_latex_table is solely responsible for
    escaping.
    """
    path = tmp_path / "t.tex"
    write_latex_table([{"cell": "x", "n": 1, "accuracy": 1.0}], path, FIELDS, caption="Planner recommended_tool accuracy")
    text = path.read_text(encoding="utf-8")
    assert "\\caption{Planner recommended\\_tool accuracy}" in text
    assert "textbackslash" not in text


def test_publication_tables_module_does_not_pre_escape_captions():
    import re
    from pathlib import Path as _Path

    src = (_Path(__file__).resolve().parents[1] / "analysis" / "publication_tables.py").read_text(encoding="utf-8")
    # No caption string literal in this module should contain a literal
    # backslash-underscore -- that is write_latex_table's job, not the
    # caller's.
    assert not re.search(r'caption="[^"]*\\\\_', src), "found a pre-escaped underscore in a caption string"


def test_write_table_all_formats_writes_three_files(tmp_path):
    paths = write_table_all_formats(ROWS, tmp_path, "mytable", FIELDS)
    assert set(paths.keys()) == {"csv", "md", "tex"}
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 0


def test_save_figure_all_formats_writes_png_and_pdf(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2])
    out = _save_figure_all_formats(fig, tmp_path / "fig.png")
    assert out["png"].exists()
    assert out["pdf"].exists()
    assert out["png"].suffix == ".png"
    assert out["pdf"].suffix == ".pdf"


def _synthetic_baseline_report():
    def cell(n, n_correct_gpt, n_correct_det):
        from evaluation.semantic_orchestration.analysis.stats import wilson_ci, exact_mcnemar, paired_bootstrap_ci

        gp, gl, gh = wilson_ci(n_correct_gpt, n)
        dp, dl, dh = wilson_ci(n_correct_det, n)
        return {
            "n": n,
            "gpt_majority_vote_accuracy": {"n": n, "n_correct": n_correct_gpt, "accuracy": gp, "wilson_ci_lower": gl, "wilson_ci_upper": gh},
            "deterministic_baseline_accuracy": {"n": n, "n_correct": n_correct_det, "accuracy": dp, "wilson_ci_lower": dl, "wilson_ci_upper": dh},
            "absolute_difference_percentage_points": 100.0 * (n_correct_gpt - n_correct_det) / n,
            "mcnemar": exact_mcnemar(2, 0),
            "paired_bootstrap_accuracy_diff": paired_bootstrap_ci([1.0, 0.0], seed=20260818),
        }

    return {
        "overall": cell(10, 10, 7),
        "by_capability": {"causal_ate": cell(5, 5, 3), "survival_adjusted_curves": cell(5, 5, 4)},
        "by_category": {c: cell(2, 2, 1) for c in ["explicit_method", "formal_estimand", "biomedical_domain", "indirect_colloquial", "noisy_verbose", "near_boundary"]},
    }


def test_plot_baseline_comparison_writes_png_pdf_csv(tmp_path):
    report = _synthetic_baseline_report()
    out = plot_baseline_comparison(report, tmp_path / "baseline.png", "test")
    assert out["png"].exists()
    assert out["pdf"].exists()
    csv_path = tmp_path / "baseline.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["cell"] == "overall"


# --------------------------------------------------------------------------
# Guarded end-to-end check against the real live results, if present.
# --------------------------------------------------------------------------

def test_run_all_produces_finalized_sections_on_real_data_if_present():
    router_sup = RESULTS_RAW_DIR / "router_supported.jsonl"
    planner_sup = RESULTS_RAW_DIR / "planner_supported.jsonl"
    det_baseline = RESULTS_PROCESSED_DIR / "deterministic_baseline.json"
    analysis_summary = RESULTS_PROCESSED_DIR / "analysis_summary.json"

    if not (router_sup.exists() and planner_sup.exists() and det_baseline.exists()):
        return  # nothing to check yet; run_all.py itself no-ops in this case

    assert analysis_summary.exists(), "run analysis.run_all before running this test"

    import json

    d = json.loads(analysis_summary.read_text(encoding="utf-8"))

    for key in [
        "router_accuracy",
        "planner_accuracy",
        "router_accuracy_majority_vote",
        "planner_accuracy_majority_vote",
        "router_consistency_stratified",
        "planner_consistency_stratified",
        "router_confusion_matrix_majority_vote",
        "planner_confusion_matrix_majority_vote",
        "planner_vs_deterministic_baseline",
        "live_run_summaries",
    ]:
        assert key in d, f"missing finalized section: {key}"

    bc = d["planner_vs_deterministic_baseline"]
    assert bc["overall"]["n"] == 120
    mc = bc["overall"]["mcnemar"]
    assert mc["b"] + mc["c"] == mc["n_discordant"]
    assert 0.0 <= bc["overall"]["gpt_majority_vote_accuracy"]["accuracy"] <= 1.0
    assert 0.0 <= bc["overall"]["deterministic_baseline_accuracy"]["accuracy"] <= 1.0
    assert bc["overall"]["paired_bootstrap_accuracy_diff"]["seed"] == 20260818

    n_cap_sum = sum(c["n"] for c in bc["by_capability"].values())
    assert n_cap_sum == 120
    n_cat_sum = sum(c["n"] for c in bc["by_category"].values())
    assert n_cat_sum == 120

    for comp, summary in d["live_run_summaries"].items():
        assert summary["requested_models"] == ["gpt-5.4"]
