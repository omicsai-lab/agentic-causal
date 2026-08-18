"""
Render analysis results (plain dicts, as produced by router_accuracy.py /
planner_accuracy.py / consistency.py / agreement.py) into Markdown tables
suitable for pasting into EVALUATION_REPORT.md / MANUSCRIPT_FACTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{100 * x:.1f}%"


def _fmt_ci(cell: Dict[str, Any]) -> str:
    if cell.get("n", 0) == 0:
        return "n/a"
    return f"{_fmt_pct(cell['accuracy'])} [{_fmt_pct(cell['wilson_ci_lower'])}, {_fmt_pct(cell['wilson_ci_upper'])}] (n={cell['n']})"


def router_accuracy_table(report: Dict[str, Any]) -> str:
    lines = ["| Cell | Accuracy [95% Wilson CI] (n) |", "|---|---|"]
    lines.append(f"| Overall | {_fmt_ci(report['overall'])} |")
    for cap, cell in report["by_capability"].items():
        lines.append(f"| capability={cap} | {_fmt_ci(cell)} |")
    for cat, cell in report["by_category"].items():
        lines.append(f"| category={cat} | {_fmt_ci(cell)} |")
    for key, cell in report["by_capability_category"].items():
        lines.append(f"| {key} | {_fmt_ci(cell)} |")
    return "\n".join(lines)


def planner_accuracy_table(report: Dict[str, Any]) -> str:
    lines = ["| Field | Accuracy [95% Wilson CI] |", "|---|---|"]
    for field, cell in report["overall_by_field"].items():
        lines.append(f"| {field} | {_fmt_ci(cell)} |")
    lines.append(f"| all_fields_correct | {_fmt_ci(report['overall_all_fields_correct'])} |")
    return "\n".join(lines)


def consistency_table(router_report: Dict[str, Any], planner_report: Dict[str, Any]) -> str:
    lines = ["| Component | Mean pairwise agreement | Full (3/3) consistency | Majority-vote accuracy |", "|---|---|---|---|"]
    ra = router_report.get("agreement", {})
    lines.append(
        f"| Router | {_fmt_pct(ra.get('mean_pairwise_agreement', float('nan')))} | "
        f"{_fmt_pct(ra.get('full_consistency_rate', float('nan')))} | "
        f"{_fmt_ci(router_report['majority_vote_accuracy'])} |"
    )
    pa = planner_report.get("agreement", {})
    lines.append(
        f"| Planner ({planner_report.get('field')}) | {_fmt_pct(pa.get('mean_pairwise_agreement', float('nan')))} | "
        f"{_fmt_pct(pa.get('full_consistency_rate', float('nan')))} | "
        f"{_fmt_ci(planner_report['majority_vote_accuracy'])} |"
    )
    return "\n".join(lines)


def confusion_matrix_table(cm: Dict[str, Any]) -> str:
    labels = cm["labels"]
    header = "| gold \\ pred | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    lines = [header, sep]
    for g in labels:
        row = cm["matrix"].get(g, {})
        lines.append("| " + g + " | " + " | ".join(str(row.get(p, 0)) for p in labels) + " |")
    return "\n".join(lines)


def write_all_tables(out_dir: Path, **reports: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "router_accuracy" in reports:
        (out_dir / "router_accuracy.md").write_text(router_accuracy_table(reports["router_accuracy"]) + "\n", encoding="utf-8")
    if "planner_accuracy" in reports:
        (out_dir / "planner_accuracy.md").write_text(planner_accuracy_table(reports["planner_accuracy"]) + "\n", encoding="utf-8")
    if "router_confusion" in reports:
        (out_dir / "router_confusion_matrix.md").write_text(confusion_matrix_table(reports["router_confusion"]) + "\n", encoding="utf-8")
