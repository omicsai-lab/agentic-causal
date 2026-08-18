"""
Orchestrate the full analysis pipeline over the raw evaluation results
(produced by the Step 2 harness runs) and write processed JSON, Markdown/
CSV/LaTeX tables, and PNG/PDF figures (with underlying CSV data).

This script does NOT make any API calls -- it only reads
results/raw/*.jsonl and results/processed/deterministic_baseline.json
(both already produced, read-only here). It is a no-op (with a clear
message) if the raw files are absent.

Usage:
    python3 -m evaluation.semantic_orchestration.analysis.run_all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.io_utils import (  # noqa: E402
    RESULTS_PROCESSED_DIR,
    RESULTS_RAW_DIR,
    load_supported_prompts,
    read_jsonl,
)
from evaluation.semantic_orchestration.analysis.router_accuracy import (  # noqa: E402
    router_accuracy_report,
    confusion_pairs,
)
from evaluation.semantic_orchestration.analysis.planner_accuracy import planner_accuracy_report  # noqa: E402
from evaluation.semantic_orchestration.analysis.consistency import (  # noqa: E402
    router_consistency_report,
    planner_consistency_report,
)
from evaluation.semantic_orchestration.analysis.agreement import (  # noqa: E402
    planner_router_agreement,
    planner_vs_router_mcnemar,
)
from evaluation.semantic_orchestration.analysis.confusion import (  # noqa: E402
    router_confusion_matrix,
    challenge_prediction_distribution,
)
from evaluation.semantic_orchestration.analysis.stratified import (  # noqa: E402
    majority_vote_accuracy_stratified,
    consistency_stratified,
    majority_vote_confusion_matrix,
)
from evaluation.semantic_orchestration.analysis.baseline_comparison import (  # noqa: E402
    build_paired_items,
    planner_majority_vs_deterministic_report,
)
from evaluation.semantic_orchestration.analysis.challenge_descriptive import challenge_report  # noqa: E402
from evaluation.semantic_orchestration.analysis.tables import write_all_tables  # noqa: E402
from evaluation.semantic_orchestration.analysis.publication_tables import write_all_publication_tables  # noqa: E402
from evaluation.semantic_orchestration.analysis.figures import (  # noqa: E402
    plot_accuracy_by_category,
    plot_baseline_comparison,
    plot_confusion_heatmap,
    plot_repeat_consistency,
)

SEMANTIC_ORCH_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = SEMANTIC_ORCH_ROOT / "tables"
FIGURES_DIR = SEMANTIC_ORCH_ROOT / "figures"
DETERMINISTIC_BASELINE_PATH = RESULTS_PROCESSED_DIR / "deterministic_baseline.json"


def _live_call_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_success = sum(1 for r in rows if r.get("api_call_succeeded"))
    n_fail = n - n_success
    if rows and "fallback_detected" in rows[0]:
        n_fallback = sum(1 for r in rows if r.get("fallback_detected"))
    else:
        n_fallback = sum(1 for r in rows if r.get("likely_fallback_by_content_match"))
    n_subst = sum(1 for r in rows if r.get("model_substituted"))
    requested = {r.get("requested_model") for r in rows if r.get("requested_model")}
    returned = {m for r in rows for m in (r.get("returned_models") or [])}
    return {
        "n_calls": n,
        "n_api_success": n_success,
        "n_api_failure": n_fail,
        "n_fallback_or_likely_fallback": n_fallback,
        "n_model_substituted": n_subst,
        "requested_models": requested,
        "returned_models_observed": returned,
    }


def main() -> None:
    router_sup_path = RESULTS_RAW_DIR / "router_supported.jsonl"
    planner_sup_path = RESULTS_RAW_DIR / "planner_supported.jsonl"
    router_chal_path = RESULTS_RAW_DIR / "router_challenge.jsonl"
    planner_chal_path = RESULTS_RAW_DIR / "planner_challenge.jsonl"

    missing = [p for p in [router_sup_path, planner_sup_path] if not p.exists()]
    if missing:
        print("run_all: raw result files not found yet (expected before Step 2 live evaluation runs):")
        for p in missing:
            print(f"  - {p}")
        print("Nothing to analyze.")
        return

    router_rows = read_jsonl(router_sup_path)
    planner_rows = read_jsonl(planner_sup_path)
    supported_prompts = load_supported_prompts()

    gold_by_id = {p["id"]: p["gold_capability_id"] for p in supported_prompts}
    capability_by_id = {p["id"]: p["capability"] for p in supported_prompts}
    category_by_id = {p["id"]: p["category"] for p in supported_prompts}

    # ---- existing (unchanged) analyses ---------------------------------
    router_report = router_accuracy_report(router_rows)
    planner_report = planner_accuracy_report(planner_rows)
    router_consistency = router_consistency_report(router_rows)
    planner_consistency = planner_consistency_report(planner_rows)
    agreement = planner_router_agreement(router_rows, planner_rows)
    mcnemar_planner_vs_router = planner_vs_router_mcnemar(router_rows, planner_rows)
    router_cm_run_level = router_confusion_matrix(router_rows)
    router_pairs = confusion_pairs(router_rows)

    # ---- new: prompt-level (majority-vote) stratified analyses ---------
    router_majority = majority_vote_accuracy_stratified(
        router_rows, lambda r: r["predicted_capability_id"], gold_by_id, capability_by_id, category_by_id
    )
    planner_majority = majority_vote_accuracy_stratified(
        planner_rows, lambda r: r["plan"].get("recommended_tool"), gold_by_id, capability_by_id, category_by_id
    )
    router_consistency_strat = consistency_stratified(router_rows, lambda r: r["predicted_capability_id"], capability_by_id, category_by_id)
    planner_consistency_strat = consistency_stratified(
        planner_rows, lambda r: r["plan"].get("recommended_tool"), capability_by_id, category_by_id
    )
    router_cm_majority = majority_vote_confusion_matrix(router_rows, lambda r: r["predicted_capability_id"], gold_by_id)
    planner_cm_majority = majority_vote_confusion_matrix(planner_rows, lambda r: r["plan"].get("recommended_tool"), gold_by_id)

    # ---- PRIMARY: GPT-5.4 planner majority vote vs deterministic baseline
    if not DETERMINISTIC_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"{DETERMINISTIC_BASELINE_PATH} not found; run harness/run_deterministic_baseline.py first "
            "(it makes zero network calls)."
        )
    deterministic_baseline = json.loads(DETERMINISTIC_BASELINE_PATH.read_text(encoding="utf-8"))
    deterministic_planner_rows = deterministic_baseline["planner_supported"]["rows"]
    paired_items = build_paired_items(planner_rows, deterministic_planner_rows, supported_prompts)
    baseline_comparison = planner_majority_vs_deterministic_report(paired_items)

    # ---- challenge set (descriptive only) -------------------------------
    challenge_desc = None
    if router_chal_path.exists() and planner_chal_path.exists():
        router_chal_rows = read_jsonl(router_chal_path)
        planner_chal_rows = read_jsonl(planner_chal_path)
        challenge_desc = challenge_report(router_chal_rows, planner_chal_rows)

    # ---- live-call bookkeeping: failures / retries / fallbacks / models -
    live_run_summaries = {"router_supported": _live_call_summary(router_rows), "planner_supported": _live_call_summary(planner_rows)}
    if router_chal_path.exists():
        live_run_summaries["router_challenge"] = _live_call_summary(read_jsonl(router_chal_path))
    if planner_chal_path.exists():
        live_run_summaries["planner_challenge"] = _live_call_summary(read_jsonl(planner_chal_path))

    # No-retry check: the harness has no retry logic; a "retry" would show
    # up as a duplicate (prompt_id, repeat_index) key in a raw file. There
    # were none in this run (see EVALUATION_REPORT.md).
    def _n_duplicate_keys(rows: List[Dict[str, Any]]) -> int:
        keys = [(r["prompt_id"], r["repeat_index"]) for r in rows]
        return len(keys) - len(set(keys))

    for name, rows in [("router_supported", router_rows), ("planner_supported", planner_rows)]:
        live_run_summaries[name]["n_duplicate_prompt_repeat_keys"] = _n_duplicate_keys(rows)

    processed: Dict[str, Any] = {
        "router_accuracy": router_report,
        "planner_accuracy": planner_report,
        "router_consistency": router_consistency,
        "planner_consistency": planner_consistency,
        "planner_router_agreement": agreement,
        "planner_vs_router_mcnemar": mcnemar_planner_vs_router,
        "router_confusion_matrix": router_cm_run_level,
        "router_confusion_pairs": router_pairs,
        "router_accuracy_majority_vote": router_majority,
        "planner_accuracy_majority_vote": planner_majority,
        "router_consistency_stratified": router_consistency_strat,
        "planner_consistency_stratified": planner_consistency_strat,
        "router_confusion_matrix_majority_vote": router_cm_majority,
        "planner_confusion_matrix_majority_vote": planner_cm_majority,
        "planner_vs_deterministic_baseline": baseline_comparison,
        "live_run_summaries": {
            k: {**v, "requested_models": sorted(v["requested_models"]), "returned_models_observed": sorted(v["returned_models_observed"])}
            for k, v in live_run_summaries.items()
        },
    }
    if challenge_desc is not None:
        processed["challenge_descriptive"] = challenge_desc
        # Backward-compat keys (previously written directly by this script).
        processed["challenge_router_prediction_distribution"] = challenge_desc["router_predicted_capability_distribution_by_challenge_type"]
        processed["challenge_planner_recommended_tool_distribution"] = challenge_desc["planner_recommended_tool_distribution_by_challenge_type"]

    RESULTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_PROCESSED_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(processed, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")

    # ---- tables (Markdown-only legacy writer, kept for compatibility) --
    write_all_tables(TABLES_DIR, router_accuracy=router_report, planner_accuracy=planner_report, router_confusion=router_cm_run_level)

    # ---- publication tables (CSV + Markdown + LaTeX) --------------------
    pub_tables_kwargs: Dict[str, Any] = dict(
        router_accuracy=router_report,
        router_majority=router_majority,
        planner_accuracy=planner_report,
        planner_majority=planner_majority,
        baseline_comparison=baseline_comparison,
        router_confusion_run_level=router_cm_run_level,
        router_confusion_majority=router_cm_majority,
        planner_confusion_majority=planner_cm_majority,
        router_consistency=router_consistency_strat,
        planner_consistency=planner_consistency_strat,
        live_run_summaries=processed["live_run_summaries"],
    )
    if challenge_desc is not None:
        pub_tables_kwargs["challenge_router_detail"] = challenge_desc["router_detail"]
        pub_tables_kwargs["challenge_planner_detail"] = challenge_desc["planner_detail"]
        pub_tables_kwargs["challenge_router_dist"] = challenge_desc["router_predicted_capability_distribution_by_challenge_type"]
        pub_tables_kwargs["challenge_planner_dist"] = challenge_desc["planner_recommended_tool_distribution_by_challenge_type"]

    written_tables = write_all_publication_tables(TABLES_DIR, **pub_tables_kwargs)
    print(f"Wrote {len(written_tables)} publication table sets (csv+md+tex) to {TABLES_DIR}")

    # ---- figures (PNG + PDF + underlying CSV) ----------------------------
    plot_accuracy_by_category(router_report, FIGURES_DIR / "router_accuracy_by_category.png", "Router accuracy by prompt category (run level, n=360)")
    plot_confusion_heatmap(router_cm_run_level, FIGURES_DIR / "router_confusion_matrix.png", "Router confusion matrix, run level (gold vs predicted)")
    plot_confusion_heatmap(router_cm_majority, FIGURES_DIR / "router_confusion_matrix_majority_vote.png", "Router confusion matrix, prompt-level majority vote")
    plot_repeat_consistency(
        router_consistency.get("agreement", {}), FIGURES_DIR / "router_repeat_consistency.png", "Router 3-repeat pairwise agreement"
    )
    plot_baseline_comparison(
        baseline_comparison,
        FIGURES_DIR / "planner_vs_deterministic_baseline.png",
        "GPT-5.4 planner (majority vote) vs. deterministic _fallback_plan() baseline",
    )
    print(f"Wrote figures (png+pdf+csv) to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
