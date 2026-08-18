"""
Figure generation for the semantic-orchestration evaluation, using
matplotlib (already pinned in requirements.txt, no new dependency).

Every figure is saved as both PNG and PDF (publication-ready vector
format), alongside a CSV of the exact data underlying the plot, so a
reader can reproduce or re-style the figure without re-running the
analysis pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _yerr_magnitude(distance: float) -> float:
    """
    Clamp a computed (estimate - lower) or (upper - estimate) distance to
    a non-negative matplotlib `yerr` magnitude.

    Wilson CI bounds mathematically always satisfy lower <= estimate <=
    upper, so both distances are conceptually >= 0. But floating-point
    rounoff inside the CI computation (see analysis/stats.py::wilson_ci)
    can yield a bound that differs from the estimate by a tiny negative
    amount at the 0%/100% boundary (e.g. upper = 0.9999999999999998 when
    estimate = 1.0, giving distance = -2.22e-16). This clamps only the
    plotted error-bar magnitude to zero in that case; it does not alter
    the CI values themselves or any statistic in analysis/stats.py.
    """
    return max(0.0, distance)


def _save_figure_all_formats(fig: "plt.Figure", out_path: Path) -> Dict[str, Path]:
    """Save `out_path` (expected to end in .png) plus a sibling .pdf."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path if out_path.suffix == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _write_underlying_csv(rows: List[Dict[str, Any]], fieldnames: List[str], out_path: Path) -> Path:
    csv_path = (out_path if out_path.suffix == ".png" else out_path.with_suffix(".png")).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def plot_accuracy_by_category(report: Dict[str, Any], out_path: Path, title: str) -> Dict[str, Path]:
    categories = list(report["by_category"].keys())
    accs = [report["by_category"][c]["accuracy"] for c in categories]
    los = [
        _yerr_magnitude(report["by_category"][c]["accuracy"] - report["by_category"][c]["wilson_ci_lower"])
        for c in categories
    ]
    his = [
        _yerr_magnitude(report["by_category"][c]["wilson_ci_upper"] - report["by_category"][c]["accuracy"])
        for c in categories
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(categories, accs, yerr=[los, his], capsize=4, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    csv_rows = [
        {
            "category": c,
            "accuracy": report["by_category"][c]["accuracy"],
            "wilson_ci_lower": report["by_category"][c]["wilson_ci_lower"],
            "wilson_ci_upper": report["by_category"][c]["wilson_ci_upper"],
            "yerr_lower": lo,
            "yerr_upper": hi,
            "n": report["by_category"][c]["n"],
        }
        for c, lo, hi in zip(categories, los, his)
    ]
    _write_underlying_csv(csv_rows, ["category", "accuracy", "wilson_ci_lower", "wilson_ci_upper", "yerr_lower", "yerr_upper", "n"], out_path)
    return _save_figure_all_formats(fig, out_path)


def plot_confusion_heatmap(cm: Dict[str, Any], out_path: Path, title: str) -> Dict[str, Path]:
    labels = cm["labels"]
    matrix = [[cm["matrix"][g].get(p, 0) for p in labels] for g in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i][j]
            if val:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > (max(max(r) for r in matrix) / 2) else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    csv_rows = [{"gold": labels[i], "predicted": labels[j], "count": matrix[i][j]} for i in range(len(labels)) for j in range(len(labels))]
    _write_underlying_csv(csv_rows, ["gold", "predicted", "count"], out_path)
    return _save_figure_all_formats(fig, out_path)


def plot_repeat_consistency(agreement: Dict[str, Any], out_path: Path, title: str) -> Dict[str, Path] | None:
    rates = agreement.get("pairwise_rates", [])
    if not rates:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([f"pair {i+1}" for i in range(len(rates))], rates, color="#55A868")
    ax.axhline(agreement.get("full_consistency_rate", 0), color="#C44E52", linestyle="--", label="full (3/3) consistency")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    csv_rows = [{"pair": f"pair_{i+1}", "agreement_rate": r} for i, r in enumerate(rates)]
    csv_rows.append({"pair": "full_3_of_3_consistency_rate", "agreement_rate": agreement.get("full_consistency_rate")})
    _write_underlying_csv(csv_rows, ["pair", "agreement_rate"], out_path)
    return _save_figure_all_formats(fig, out_path)


def plot_baseline_comparison(report: Dict[str, Any], out_path: Path, title: str) -> Dict[str, Path]:
    """
    PRIMARY manuscript figure: GPT-5.4 planner (prompt-level majority
    vote) vs. the deterministic `_fallback_plan()` baseline,
    recommended_tool accuracy, grouped by cell (overall, each gold
    capability, each linguistic category). Wilson CI error bars (clamped
    via _yerr_magnitude for the same floating-point-rounoff reason as
    plot_accuracy_by_category).
    """
    cells: List[str] = ["overall"] + list(report["by_capability"].keys()) + list(report["by_category"].keys())
    cell_reports = [report["overall"]] + list(report["by_capability"].values()) + list(report["by_category"].values())

    gpt_acc = [c["gpt_majority_vote_accuracy"]["accuracy"] for c in cell_reports]
    gpt_lo = [_yerr_magnitude(c["gpt_majority_vote_accuracy"]["accuracy"] - c["gpt_majority_vote_accuracy"]["wilson_ci_lower"]) for c in cell_reports]
    gpt_hi = [_yerr_magnitude(c["gpt_majority_vote_accuracy"]["wilson_ci_upper"] - c["gpt_majority_vote_accuracy"]["accuracy"]) for c in cell_reports]

    det_acc = [c["deterministic_baseline_accuracy"]["accuracy"] for c in cell_reports]
    det_lo = [_yerr_magnitude(c["deterministic_baseline_accuracy"]["accuracy"] - c["deterministic_baseline_accuracy"]["wilson_ci_lower"]) for c in cell_reports]
    det_hi = [_yerr_magnitude(c["deterministic_baseline_accuracy"]["wilson_ci_upper"] - c["deterministic_baseline_accuracy"]["accuracy"]) for c in cell_reports]

    x = range(len(cells))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], gpt_acc, width=width, yerr=[gpt_lo, gpt_hi], capsize=3, label="GPT-5.4 planner (majority vote)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], det_acc, width=width, yerr=[det_lo, det_hi], capsize=3, label="Deterministic _fallback_plan()", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cells, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recommended_tool accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    csv_rows = []
    for cell, ga, gl, gh, da, dl, dh in zip(cells, gpt_acc, gpt_lo, gpt_hi, det_acc, det_lo, det_hi):
        csv_rows.append(
            {
                "cell": cell,
                "gpt_majority_accuracy": ga,
                "gpt_yerr_lower": gl,
                "gpt_yerr_upper": gh,
                "deterministic_accuracy": da,
                "det_yerr_lower": dl,
                "det_yerr_upper": dh,
            }
        )
    _write_underlying_csv(
        csv_rows,
        ["cell", "gpt_majority_accuracy", "gpt_yerr_lower", "gpt_yerr_upper", "deterministic_accuracy", "det_yerr_lower", "det_yerr_upper"],
        out_path,
    )
    return _save_figure_all_formats(fig, out_path)
