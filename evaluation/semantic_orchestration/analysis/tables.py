"""
Render analysis results (plain dicts, as produced by router_accuracy.py /
planner_accuracy.py / consistency.py / agreement.py) into Markdown tables
suitable for pasting into EVALUATION_REPORT.md / MANUSCRIPT_FACTS.md.

Also provides generic flat-row-list writers (CSV / Markdown / LaTeX) used
by `analysis/publication_tables.py` to emit the same underlying data in
all three publication formats from a single row-list representation, so
the numbers in each format are guaranteed identical (built from the same
rows, not re-derived per format).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def write_csv_table(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _latex_escape(s: Any) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def write_markdown_table(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "|" + "|".join(["---"] * len(fieldnames)) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: List[str], caption: str = "", label: str = "") -> None:
    lines = ["\\begin{table}[ht]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{" + "l" * len(fieldnames) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(f) for f in fieldnames) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(_latex_escape(row.get(k, "")) for k in fieldnames) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_all_formats(rows: Sequence[Dict[str, Any]], out_dir: Path, basename: str, fieldnames: List[str], caption: str = "", label: str = "") -> Dict[str, Path]:
    """Write the same row-list to <basename>.{csv,md,tex} and return the
    three paths, keyed by format."""
    csv_path = out_dir / f"{basename}.csv"
    md_path = out_dir / f"{basename}.md"
    tex_path = out_dir / f"{basename}.tex"
    write_csv_table(rows, csv_path, fieldnames)
    write_markdown_table(rows, md_path, fieldnames)
    write_latex_table(rows, tex_path, fieldnames, caption=caption or basename.replace("_", " "), label=f"tab:{basename}" if not label else label)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def write_all_tables(out_dir: Path, **reports: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "router_accuracy" in reports:
        (out_dir / "router_accuracy.md").write_text(router_accuracy_table(reports["router_accuracy"]) + "\n", encoding="utf-8")
    if "planner_accuracy" in reports:
        (out_dir / "planner_accuracy.md").write_text(planner_accuracy_table(reports["planner_accuracy"]) + "\n", encoding="utf-8")
    if "router_confusion" in reports:
        (out_dir / "router_confusion_matrix.md").write_text(confusion_matrix_table(reports["router_confusion"]) + "\n", encoding="utf-8")
