"""
Orchestrate the full analysis pipeline over the raw evaluation results
(produced by the Step 2 harness runs) and write processed JSON, Markdown
tables, and PNG figures.

This script does NOT make any API calls -- it only reads
results/raw/*.jsonl. It is safe to re-run at any time once those files
exist. It is a no-op (with a clear message) if the raw files are absent,
which is expected for Step 1.

Usage:
    python3 -m evaluation.semantic_orchestration.analysis.run_all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.semantic_orchestration.harness.io_utils import (  # noqa: E402
    RESULTS_PROCESSED_DIR,
    RESULTS_RAW_DIR,
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
from evaluation.semantic_orchestration.analysis.tables import write_all_tables  # noqa: E402
from evaluation.semantic_orchestration.analysis.figures import (  # noqa: E402
    plot_accuracy_by_category,
    plot_confusion_heatmap,
    plot_repeat_consistency,
)

SEMANTIC_ORCH_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = SEMANTIC_ORCH_ROOT / "tables"
FIGURES_DIR = SEMANTIC_ORCH_ROOT / "figures"


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
        print("Nothing to analyze. This is expected for Step 1.")
        return

    router_rows = read_jsonl(router_sup_path)
    planner_rows = read_jsonl(planner_sup_path)

    router_report = router_accuracy_report(router_rows)
    planner_report = planner_accuracy_report(planner_rows)
    router_consistency = router_consistency_report(router_rows)
    planner_consistency = planner_consistency_report(planner_rows)
    agreement = planner_router_agreement(router_rows, planner_rows)
    mcnemar = planner_vs_router_mcnemar(router_rows, planner_rows)
    router_cm = router_confusion_matrix(router_rows)
    router_pairs = confusion_pairs(router_rows)

    processed = {
        "router_accuracy": router_report,
        "planner_accuracy": planner_report,
        "router_consistency": router_consistency,
        "planner_consistency": planner_consistency,
        "planner_router_agreement": agreement,
        "planner_vs_router_mcnemar": mcnemar,
        "router_confusion_matrix": router_cm,
        "router_confusion_pairs": router_pairs,
    }

    if router_chal_path.exists():
        processed["challenge_router_prediction_distribution"] = challenge_prediction_distribution(read_jsonl(router_chal_path))
    if planner_chal_path.exists():
        chal_planner_rows = read_jsonl(planner_chal_path)
        dist: dict = {}
        for r in chal_planner_rows:
            ct = r["challenge_type"]
            tool = r["plan"].get("recommended_tool")
            dist.setdefault(ct, {})
            dist[ct][tool] = dist[ct].get(tool, 0) + 1
        processed["challenge_planner_recommended_tool_distribution"] = dist

    RESULTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_PROCESSED_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(processed, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")

    write_all_tables(
        TABLES_DIR,
        router_accuracy=router_report,
        planner_accuracy=planner_report,
        router_confusion=router_cm,
    )
    print(f"Wrote tables to {TABLES_DIR}")

    plot_accuracy_by_category(router_report, FIGURES_DIR / "router_accuracy_by_category.png", "Router accuracy by prompt category")
    plot_confusion_heatmap(router_cm, FIGURES_DIR / "router_confusion_matrix.png", "Router confusion matrix (gold vs predicted)")
    plot_repeat_consistency(router_consistency.get("agreement", {}), FIGURES_DIR / "router_repeat_consistency.png", "Router 3-repeat pairwise agreement")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
