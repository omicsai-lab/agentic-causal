"""
Build flat row-list representations of every finalized result and write
each to CSV, Markdown, and LaTeX via analysis/tables.py's generic
row-table writers, so all three formats are guaranteed to show identical
numbers (derived from the same row list, not re-derived per format).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.semantic_orchestration.analysis.tables import write_table_all_formats


def _fmt(x: Any) -> Any:
    if isinstance(x, float):
        if x != x:  # NaN
            return ""
        return round(x, 6)
    return x


def _acc_row(label: str, cell: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell": label,
        "n": cell.get("n", 0),
        "n_correct": cell.get("n_correct", 0),
        "accuracy": _fmt(cell.get("accuracy")),
        "wilson_ci_lower": _fmt(cell.get("wilson_ci_lower")),
        "wilson_ci_upper": _fmt(cell.get("wilson_ci_upper")),
    }


ACC_FIELDS = ["cell", "n", "n_correct", "accuracy", "wilson_ci_lower", "wilson_ci_upper"]


def router_accuracy_run_level_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [_acc_row("overall", report["overall"])]
    for cap, cell in report["by_capability"].items():
        rows.append(_acc_row(f"capability={cap}", cell))
    for cat, cell in report["by_category"].items():
        rows.append(_acc_row(f"category={cat}", cell))
    for key, cell in report["by_capability_category"].items():
        rows.append(_acc_row(key, cell))
    for rep, cell in report["by_repeat"].items():
        rows.append(_acc_row(f"repeat={rep}", cell))
    return rows


def majority_vote_accuracy_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [_acc_row("overall", report["overall"])]
    for cap, cell in report["by_capability"].items():
        rows.append(_acc_row(f"capability={cap}", cell))
    for cat, cell in report["by_category"].items():
        rows.append(_acc_row(f"category={cat}", cell))
    return rows


def planner_structured_field_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for field, cell in report["overall_by_field"].items():
        rows.append(_acc_row(field, cell))
    rows.append(_acc_row("all_fields_correct", report["overall_all_fields_correct"]))
    for cap, fields in report["by_capability"].items():
        for field, cell in fields.items():
            rows.append(_acc_row(f"capability={cap}::{field}", cell))
    for cat, fields in report["by_category"].items():
        for field, cell in fields.items():
            rows.append(_acc_row(f"category={cat}::{field}", cell))
    return rows


BASELINE_COMPARISON_FIELDS = [
    "cell",
    "n",
    "gpt_majority_accuracy",
    "gpt_ci_lower",
    "gpt_ci_upper",
    "deterministic_accuracy",
    "det_ci_lower",
    "det_ci_upper",
    "absolute_diff_pp",
    "mcnemar_b",
    "mcnemar_c",
    "mcnemar_p_value",
    "bootstrap_mean_diff",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "bootstrap_seed",
]


def baseline_comparison_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _row(label: str, cell: Dict[str, Any]) -> Dict[str, Any]:
        gpt = cell["gpt_majority_vote_accuracy"]
        det = cell["deterministic_baseline_accuracy"]
        mc = cell["mcnemar"]
        bs = cell["paired_bootstrap_accuracy_diff"]
        return {
            "cell": label,
            "n": cell["n"],
            "gpt_majority_accuracy": _fmt(gpt["accuracy"]),
            "gpt_ci_lower": _fmt(gpt["wilson_ci_lower"]),
            "gpt_ci_upper": _fmt(gpt["wilson_ci_upper"]),
            "deterministic_accuracy": _fmt(det["accuracy"]),
            "det_ci_lower": _fmt(det["wilson_ci_lower"]),
            "det_ci_upper": _fmt(det["wilson_ci_upper"]),
            "absolute_diff_pp": _fmt(cell["absolute_difference_percentage_points"]),
            "mcnemar_b": mc["b"],
            "mcnemar_c": mc["c"],
            "mcnemar_p_value": _fmt(mc["p_value"]),
            "bootstrap_mean_diff": _fmt(bs["mean_diff"]),
            "bootstrap_ci_lower": _fmt(bs["ci_lower"]),
            "bootstrap_ci_upper": _fmt(bs["ci_upper"]),
            "bootstrap_seed": bs["seed"],
        }

    rows = [_row("overall", report["overall"])]
    for cap, cell in report["by_capability"].items():
        rows.append(_row(f"capability={cap}", cell))
    for cat, cell in report["by_category"].items():
        rows.append(_row(f"category={cat}", cell))
    return rows


def confusion_matrix_rows(cm: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    labels = cm["labels"]
    rows = []
    for g in labels:
        row = {"gold": g}
        row.update({p: cm["matrix"].get(g, {}).get(p, 0) for p in labels})
        rows.append(row)
    return rows, ["gold"] + labels


CONSISTENCY_FIELDS = ["cell", "n_prompts", "mean_pairwise_agreement", "full_consistency_rate"] + [
    f"pair_{i+1}_agreement" for i in range(3)
]


def consistency_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _row(label: str, agreement: Dict[str, Any]) -> Dict[str, Any]:
        rates = agreement.get("pairwise_rates", [])
        row = {
            "cell": label,
            "n_prompts": agreement.get("n_prompts", 0),
            "mean_pairwise_agreement": _fmt(agreement.get("mean_pairwise_agreement")),
            "full_consistency_rate": _fmt(agreement.get("full_consistency_rate")),
        }
        for i in range(3):
            row[f"pair_{i+1}_agreement"] = _fmt(rates[i]) if i < len(rates) else ""
        return row

    rows = [_row("overall", report["overall"])]
    for cap, agreement in report["by_capability"].items():
        rows.append(_row(f"capability={cap}", agreement))
    for cat, agreement in report["by_category"].items():
        rows.append(_row(f"category={cat}", agreement))
    return rows


CHALLENGE_DETAIL_ROUTER_FIELDS = [
    "prompt_id",
    "challenge_type",
    "predicted_capability_id",
    "fallback_detected",
    "api_call_succeeded",
    "returned_models",
    "reason",
]

CHALLENGE_DETAIL_PLANNER_FIELDS = [
    "prompt_id",
    "challenge_type",
    "recommended_tool",
    "task_type",
    "outcome_type",
    "likely_fallback_by_content_match",
    "api_call_succeeded",
    "returned_models",
    "reasoning",
]

CHALLENGE_DIST_FIELDS = ["component", "challenge_type", "predicted_label", "count"]


def challenge_distribution_rows(router_dist: Dict[str, Dict[str, int]], planner_dist: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    rows = []
    for challenge_type, counts in router_dist.items():
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            rows.append({"component": "router", "challenge_type": challenge_type, "predicted_label": label, "count": n})
    for challenge_type, counts in planner_dist.items():
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            rows.append({"component": "planner", "challenge_type": challenge_type, "predicted_label": label, "count": n})
    return rows


LIVE_RUN_SUMMARY_FIELDS = [
    "component",
    "n_calls",
    "n_api_success",
    "n_api_failure",
    "n_fallback_or_likely_fallback",
    "n_model_substituted",
    "requested_models",
    "returned_models_observed",
]


def live_run_summary_rows(summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for component, s in summaries.items():
        rows.append(
            {
                "component": component,
                "n_calls": s["n_calls"],
                "n_api_success": s["n_api_success"],
                "n_api_failure": s["n_api_failure"],
                "n_fallback_or_likely_fallback": s["n_fallback_or_likely_fallback"],
                "n_model_substituted": s["n_model_substituted"],
                "requested_models": ";".join(sorted(s["requested_models"])),
                "returned_models_observed": ";".join(sorted(s["returned_models_observed"])),
            }
        )
    return rows


def write_all_publication_tables(out_dir: Path, **kwargs: Any) -> Dict[str, Dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Dict[str, Path]] = {}

    if "router_accuracy" in kwargs:
        written["router_accuracy_run_level"] = write_table_all_formats(
            router_accuracy_run_level_rows(kwargs["router_accuracy"]), out_dir, "router_accuracy_run_level", ACC_FIELDS,
            caption="Router accuracy, run level (n=360 calls, 3 repeats x 120 prompts)",
        )

    if "router_majority" in kwargs:
        written["router_accuracy_majority_vote"] = write_table_all_formats(
            majority_vote_accuracy_rows(kwargs["router_majority"]), out_dir, "router_accuracy_majority_vote", ACC_FIELDS,
            caption="Router accuracy, prompt-level majority vote (n=120 prompts)",
        )

    if "planner_accuracy" in kwargs:
        written["planner_structured_field_accuracy"] = write_table_all_formats(
            planner_structured_field_rows(kwargs["planner_accuracy"]), out_dir, "planner_structured_field_accuracy", ACC_FIELDS,
            caption="Planner structured-field accuracy, run level (n=360 calls)",
        )

    if "planner_majority" in kwargs:
        written["planner_accuracy_majority_vote"] = write_table_all_formats(
            majority_vote_accuracy_rows(kwargs["planner_majority"]), out_dir, "planner_accuracy_majority_vote", ACC_FIELDS,
            caption="Planner recommended_tool accuracy, prompt-level majority vote (n=120 prompts)",
        )

    if "baseline_comparison" in kwargs:
        written["planner_vs_deterministic_baseline"] = write_table_all_formats(
            baseline_comparison_rows(kwargs["baseline_comparison"]), out_dir, "planner_vs_deterministic_baseline", BASELINE_COMPARISON_FIELDS,
            caption="GPT-5.4 planner (majority vote) vs. deterministic _fallback_plan() baseline, recommended_tool accuracy",
        )

    if "router_confusion_run_level" in kwargs:
        rows, fields = confusion_matrix_rows(kwargs["router_confusion_run_level"])
        written["router_confusion_matrix_run_level"] = write_table_all_formats(
            rows, out_dir, "router_confusion_matrix_run_level", fields, caption="Router confusion matrix, run level (n=360 calls)"
        )

    if "router_confusion_majority" in kwargs:
        rows, fields = confusion_matrix_rows(kwargs["router_confusion_majority"])
        written["router_confusion_matrix_majority_vote"] = write_table_all_formats(
            rows, out_dir, "router_confusion_matrix_majority_vote", fields, caption="Router confusion matrix, prompt-level majority vote (n=120 prompts)"
        )

    if "planner_confusion_majority" in kwargs:
        rows, fields = confusion_matrix_rows(kwargs["planner_confusion_majority"])
        written["planner_confusion_matrix_majority_vote"] = write_table_all_formats(
            rows, out_dir, "planner_confusion_matrix_majority_vote", fields, caption="Planner recommended_tool confusion matrix, prompt-level majority vote (n=120 prompts)"
        )

    if "router_consistency" in kwargs:
        written["router_consistency"] = write_table_all_formats(
            consistency_rows(kwargs["router_consistency"]), out_dir, "router_consistency", CONSISTENCY_FIELDS,
            caption="Router 3-run pairwise agreement and full (3/3) consistency",
        )

    if "planner_consistency" in kwargs:
        written["planner_consistency"] = write_table_all_formats(
            consistency_rows(kwargs["planner_consistency"]), out_dir, "planner_consistency", CONSISTENCY_FIELDS,
            caption="Planner recommended_tool 3-run pairwise agreement and full (3/3) consistency",
        )

    if "challenge_router_detail" in kwargs:
        written["challenge_router_detail"] = write_table_all_formats(
            kwargs["challenge_router_detail"], out_dir, "challenge_router_detail", CHALLENGE_DETAIL_ROUTER_FIELDS,
            caption="Router output on all 30 challenge prompts (descriptive only)",
        )

    if "challenge_planner_detail" in kwargs:
        written["challenge_planner_detail"] = write_table_all_formats(
            kwargs["challenge_planner_detail"], out_dir, "challenge_planner_detail", CHALLENGE_DETAIL_PLANNER_FIELDS,
            caption="Planner output on all 30 challenge prompts (descriptive only)",
        )

    if "challenge_router_dist" in kwargs and "challenge_planner_dist" in kwargs:
        written["challenge_prediction_distribution"] = write_table_all_formats(
            challenge_distribution_rows(kwargs["challenge_router_dist"], kwargs["challenge_planner_dist"]),
            out_dir, "challenge_prediction_distribution", CHALLENGE_DIST_FIELDS,
            caption="Challenge-set predicted-label distribution by challenge_type (descriptive only)",
        )

    if "live_run_summaries" in kwargs:
        written["live_run_summary"] = write_table_all_formats(
            live_run_summary_rows(kwargs["live_run_summaries"]), out_dir, "live_run_summary", LIVE_RUN_SUMMARY_FIELDS,
            caption="Live GPT-5.4 evaluation call outcomes: successes, failures, fallbacks, model identifiers",
        )

    return written
