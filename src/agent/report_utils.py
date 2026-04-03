from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

# import plot registry and trigger autodiscovery
import src.agent.plots  # noqa: F401
from src.agent.plots.registry import get_plotter

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _fmt_num(x: Any, digits: int = 4) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _parse_ate_from_stdout(stdout: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    try:
        lines = stdout.splitlines()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.match(r"^1\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+$", s):
                parts = s.split()
                if len(parts) >= 5:
                    result["ate"] = float(parts[1])
                    result["ci_lower"] = float(parts[3])
                    result["ci_upper"] = float(parts[4])
                    return result
    except Exception:
        pass
    return result


def _extract_effect_and_ci(
    artifacts: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    effect_keys = ["ate", "effect", "estimate", "average_treatment_effect"]
    lo_keys = ["ci_lower", "lower_ci", "lcl"]
    hi_keys = ["ci_upper", "upper_ci", "ucl"]

    effect = None
    lo = None
    hi = None

    for k in effect_keys:
        v = _safe_float(artifacts.get(k))
        if v is not None:
            effect = v
            break

    for k in lo_keys:
        v = _safe_float(artifacts.get(k))
        if v is not None:
            lo = v
            break

    for k in hi_keys:
        v = _safe_float(artifacts.get(k))
        if v is not None:
            hi = v
            break

    return effect, lo, hi


def build_user_summary(
    *,
    status: str,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    stdout: str,
    stderr: str,
    artifacts: Dict[str, Any],
) -> str:
    if status != "ok":
        if stderr.strip():
            return f"Analysis failed. Error details: {stderr.strip()}"
        return "Analysis failed."

    tool_key = capability_id or selected_tool or ""

    if tool_key == "summary_stats" or selected_tool == "summary_stats":
        shape = artifacts.get("shape", {})
        n_rows = shape.get("n_rows", "?")
        n_cols = shape.get("n_cols", "?")
        return (
            f"Dataset summary completed successfully. "
            f"The dataset contains {n_rows} rows and {n_cols} columns. "
            f"This report describes the uploaded data and highlights its basic structure."
        )

    if tool_key == "linear_regression" or selected_tool == "linear_regression":
        outcome = artifacts.get("outcome", "the outcome")
        r2 = artifacts.get("r2_in_sample")
        return (
            f"Linear regression completed for {outcome}. "
            f"The in-sample R-squared is {_fmt_num(r2, 3)}. "
            f"This describes how much variation in the outcome is explained by the fitted model within the analyzed sample."
        )

    if tool_key == "logistic_regression" or selected_tool == "logistic_regression":
        outcome = artifacts.get("outcome", "the outcome")
        metrics = artifacts.get("metrics_in_sample", {})
        acc = metrics.get("accuracy")
        auc = metrics.get("auc")
        text = (
            f"Logistic regression completed for {outcome}. "
            f"The in-sample accuracy is {_fmt_num(acc, 3)}"
        )
        if auc is not None:
            text += f", and the AUC is {_fmt_num(auc, 3)}. "
        else:
            text += ". "
        text += "Higher AUC values indicate better separation between the two outcome groups in the fitted sample."
        return text

    if tool_key == "survival_adjusted_curves" or selected_tool == "adjustedcurves":
        return (
            "Adjusted survival analysis completed successfully. "
            "This report summarizes how survival probability changes over time across groups after statistical adjustment. "
            "If a plotting plugin is available for this tool, a figure is included below."
        )

    if tool_key in {"causal_ate", "binary_edrip"} or selected_tool in {"causalmodels", "binary_edrip"}:
        effect, lo, hi = _extract_effect_and_ci(artifacts)
        if effect is not None and lo is not None and hi is not None:
            return (
                f"Causal effect estimation completed successfully. "
                f"The estimated effect is {_fmt_num(effect)} with a 95% confidence interval from {_fmt_num(lo)} to {_fmt_num(hi)}. "
                f"In plain language, this summarizes the average difference expected under treatment versus no treatment after adjustment by the selected method."
            )
        if effect is not None:
            return (
                f"Causal effect estimation completed successfully. "
                f"The estimated effect is {_fmt_num(effect)}. "
                f"This value represents the average treatment effect estimated by the selected causal method."
            )
        return (
            "Causal analysis completed successfully. "
            "A structured result was produced, but no standard effect estimate was found for plain-language display."
        )

    if stdout.strip():
        first_line = stdout.strip().splitlines()[0]
        return f"Analysis completed successfully. Main output: {first_line}"

    return "Analysis completed successfully."


def build_interpretation_text(
    *,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    artifacts: Dict[str, Any],
) -> str:
    tool_key = capability_id or selected_tool or ""
    effect, lo, hi = _extract_effect_and_ci(artifacts)

    if tool_key in {"causal_ate", "binary_edrip"} or selected_tool in {"causalmodels", "binary_edrip"}:
        if effect is not None:
            txt = (
                f"The main quantity of interest is the estimated average treatment effect, equal to {_fmt_num(effect)}. "
                "A positive estimate suggests the treatment is associated with a higher expected outcome on average, while a negative estimate suggests a lower expected outcome."
            )
            if lo is not None and hi is not None:
                txt += (
                    f" The reported 95% confidence interval ranges from {_fmt_num(lo)} to {_fmt_num(hi)}. "
                    "Intervals farther from zero indicate stronger evidence for a directional effect, whereas intervals close to zero indicate greater uncertainty."
                )
            txt += (
                " This interpretation assumes the chosen causal method is appropriate for the data and that the relevant identifying assumptions are sufficiently plausible."
            )
            return txt

    if tool_key == "survival_adjusted_curves" or selected_tool == "adjustedcurves":
        return (
            "When available, the survival figure compares estimated survival probabilities across groups over time. "
            "Curves that remain higher over time indicate a higher estimated probability of remaining event-free."
        )

    if tool_key == "summary_stats" or selected_tool == "summary_stats":
        return (
            "This report is descriptive rather than inferential. "
            "Its goal is to help the user understand the dataset before modeling."
        )

    if tool_key == "linear_regression" or selected_tool == "linear_regression":
        return (
            "Regression coefficients describe how the fitted outcome changes with each covariate, holding the remaining covariates fixed within the model specification."
        )

    if tool_key == "logistic_regression" or selected_tool == "logistic_regression":
        return (
            "The logistic model estimates associations with a binary outcome. "
            "Odds ratios greater than 1 indicate higher modeled odds, while values below 1 indicate lower modeled odds."
        )

    return (
        "This section provides a user-oriented interpretation of the main output. "
        "It is intended as a reading aid rather than a substitute for formal statistical review."
    )


def build_method_text(
    *,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    artifacts: Dict[str, Any],
) -> str:
    tool_key = capability_id or selected_tool or ""

    if tool_key in {"causal_ate", "binary_edrip"} or selected_tool in {"causalmodels", "binary_edrip"}:
        return (
            "The system used a causal inference tool to estimate the average effect of treatment on the chosen outcome. "
            "The backend received the uploaded dataset and user-specified variables, ran the selected estimator, and converted the resulting outputs into a user-friendly report."
        )

    if tool_key == "survival_adjusted_curves" or selected_tool == "adjustedcurves":
        return (
            "The system used an adjusted survival analysis workflow. "
            "The backend estimated survival patterns over time while accounting for the user-specified grouping variables and any provided covariates."
        )

    if tool_key == "summary_stats" or selected_tool == "summary_stats":
        return (
            "The system summarized the uploaded CSV file by reporting its dimensions, variables, missingness patterns, and basic numeric summaries."
        )

    if tool_key == "linear_regression" or selected_tool == "linear_regression":
        return (
            "The system fit a linear regression model using the uploaded dataset and the specified outcome and covariates."
        )

    if tool_key == "logistic_regression" or selected_tool == "logistic_regression":
        return (
            "The system fit a logistic regression model for a binary outcome and computed basic in-sample evaluation metrics."
        )

    return (
        "The system processed the uploaded file, selected the requested analysis pathway, ran the backend tool, and produced a report-ready summary."
    )


def generate_graphs(
    *,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    artifacts: Dict[str, Any],
    out_dir: Path,
    generate_plots: bool,
    stdout: str = "",
    stderr: str = "",
) -> List[str]:
    if not generate_plots:
        return []

    if not capability_id:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = get_plotter(capability_id)
    if plotter is None:
        return []

    try:
        if not plotter.can_plot(artifacts, stdout=stdout, stderr=stderr):
            return []
    except Exception:
        return []

    try:
        paths = plotter.generate(
            artifacts=artifacts,
            out_dir=out_dir,
            stdout=stdout,
            stderr=stderr,
        )
        return paths or []
    except Exception:
        return []


def _metrics_table_data(artifacts: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = [["Metric", "Value"]]

    preferred_keys = [
        "ate",
        "ci_lower",
        "ci_upper",
        "effect",
        "estimate",
        "n_complete_cases",
        "r2_in_sample",
        "shape",
        "router_reason",
        "out_dir",
    ]

    added = set()

    for key in preferred_keys:
        if key in artifacts:
            rows.append([str(key), str(artifacts.get(key))])
            added.add(key)

    for k, v in artifacts.items():
        if k in added:
            continue
        if isinstance(v, (str, int, float, bool)):
            rows.append([str(k), str(v)])
        elif isinstance(v, dict) and k in {"metrics_in_sample", "shape"}:
            rows.append([str(k), str(v)])

        if len(rows) >= 18:
            break

    return rows


def create_pdf_report(
    *,
    pdf_path: Path,
    title: str,
    summary_text: str,
    status: str,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    stdout: str,
    stderr: str,
    artifacts: Dict[str, Any],
    graph_paths: Sequence[str],
) -> str:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9.2, leading=12))
    styles.add(ParagraphStyle(name="Summary", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionText", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8))

    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))

    meta = (
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
        f"Status: {status}<br/>"
        f"Tool: {selected_tool or 'N/A'}<br/>"
        f"Capability: {capability_id or 'N/A'}"
    )
    story.append(Paragraph(meta, styles["Small"]))
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    story.append(Paragraph(summary_text, styles["Summary"]))

    story.append(Paragraph("2. What the Analysis Did", styles["Heading2"]))
    story.append(
        Paragraph(
            build_method_text(
                selected_tool=selected_tool,
                capability_id=capability_id,
                artifacts=artifacts,
            ),
            styles["SectionText"],
        )
    )

    story.append(Paragraph("3. How to Interpret the Result", styles["Heading2"]))
    story.append(
        Paragraph(
            build_interpretation_text(
                selected_tool=selected_tool,
                capability_id=capability_id,
                artifacts=artifacts,
            ),
            styles["SectionText"],
        )
    )

    story.append(Paragraph("4. Key Reported Outputs", styles["Heading2"]))

    table_key_style = ParagraphStyle(
        name="TableKey",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=11,
        spaceAfter=0,
    )
    table_val_style = ParagraphStyle(
        name="TableVal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        spaceAfter=0,
    )

    raw_table_data = _metrics_table_data(artifacts)
    table_data: List[List[Any]] = []

    table_data.append([
        Paragraph("<b>Metric</b>", table_key_style),
        Paragraph("<b>Value</b>", table_key_style),
    ])

    for row in raw_table_data[1:]:
        key, val = row[0], row[1]
        if isinstance(val, str):
            val = val.replace("/", "/<wbr/>").replace("_", "_<wbr/>")
        table_data.append([
            Paragraph(str(key), table_key_style),
            Paragraph(str(val), table_val_style),
        ])

    table = Table(
        table_data,
        colWidths=[1.7 * inch, 4.3 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8D0DA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))

    if graph_paths:
        story.append(Paragraph("5. Figures", styles["Heading2"]))
        story.append(
            Paragraph(
                "The figure below provides a visual summary of the main modeled result. "
                "When a plotting plugin exists for the selected tool, it is used here to generate a deterministic visualization.",
                styles["SectionText"],
            )
        )
        for gp in graph_paths:
            gp_path = Path(gp)
            if gp_path.exists():
                story.append(Image(str(gp_path), width=6.1 * inch, height=3.5 * inch, kind="proportional"))
                story.append(Spacer(1, 0.12 * inch))

    raw_text = stderr.strip() if status != "ok" and stderr.strip() else stdout.strip()
    if raw_text:
        story.append(Paragraph("6. Technical Appendix", styles["Heading2"]))
        story.append(
            Paragraph(
                "The appendix below reproduces the backend tool output for technical reference. "
                "It is included so advanced users can inspect the raw estimator output and model details.",
                styles["SectionText"],
            )
        )
        cleaned = (
            raw_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(Paragraph(cleaned, styles["Small"]))

    doc.build(story)
    return str(pdf_path)


def create_user_outputs(
    *,
    out_dir: Path,
    status: str,
    selected_tool: Optional[str],
    capability_id: Optional[str],
    stdout: str,
    stderr: str,
    artifacts: Dict[str, Any],
    generate_plots: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep this fallback for plain-language summaries and tables
    if "ate" not in artifacts:
        parsed = _parse_ate_from_stdout(stdout)
        if parsed:
            artifacts.update(parsed)

    user_summary = build_user_summary(
        status=status,
        selected_tool=selected_tool,
        capability_id=capability_id,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
    )

    graph_paths = generate_graphs(
        selected_tool=selected_tool,
        capability_id=capability_id,
        artifacts=artifacts,
        out_dir=out_dir,
        generate_plots=generate_plots,
        stdout=stdout,
        stderr=stderr,
    )

    report_pdf = create_pdf_report(
        pdf_path=out_dir / "report.pdf",
        title="Causal Agent Project Report",
        summary_text=user_summary,
        status=status,
        selected_tool=selected_tool,
        capability_id=capability_id,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        graph_paths=graph_paths,
    )

    return {
        "user_summary": user_summary,
        "graph_paths": graph_paths,
        "report_pdf": report_pdf,
    }