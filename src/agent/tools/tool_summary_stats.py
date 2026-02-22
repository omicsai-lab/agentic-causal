# src/agent/tools/tool_summary_stats.py
from __future__ import annotations

from typing import Tuple, Dict, Any, List

import pandas as pd

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class SummaryStatsTool(BaseTool):
    @property
    def name(self) -> str:
        return "summary_stats"

    @property
    def capability_id(self) -> str:
        return "summary_stats"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.csv:
            return False, "summary_stats requires csv."
        return True, ""

    def run(self, req: RunRequest) -> ToolResult:
        try:
            df = pd.read_csv(req.csv)
        except Exception as e:
            return ToolResult(
                status="error",
                selected_tool=self.name,
                stdout="",
                stderr=f"Failed to read csv='{req.csv}': {e}",
                exit_code=2,
                artifacts={},
            )

        n_rows, n_cols = df.shape
        cols: List[str] = list(df.columns)

        # Missingness
        miss_by_col = df.isna().sum().sort_values(ascending=False)
        miss_rate_by_col = (miss_by_col / max(n_rows, 1)).round(6)

        # Decide which columns to summarize
        # - if user passed covariates: treat them as "columns of interest"
        # - else summarize all numeric columns
        if req.covariates:
            target_cols = [c for c in req.covariates if c in df.columns]
        else:
            target_cols = [c for c in df.select_dtypes(include="number").columns]

        # Numeric summary (deterministic ordering)
        numeric_summary: Dict[str, Dict[str, Any]] = {}
        if target_cols:
            desc = df[target_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
            # Keep a stable, small set of stats
            keep = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            desc = desc[[c for c in keep if c in desc.columns]]
            # Convert to plain python floats
            for col in desc.index.tolist():
                row = desc.loc[col].to_dict()
                numeric_summary[col] = {k: (float(v) if pd.notna(v) else None) for k, v in row.items()}

        # Pretty stdout (short, human-readable)
        topk = 12
        top_missing_lines = []
        for c in miss_by_col.head(topk).index.tolist():
            top_missing_lines.append(f"- {c}: missing={int(miss_by_col[c])}, rate={float(miss_rate_by_col[c])}")

        stdout = (
            "Dataset summary (MVP):\n"
            f"- path: {req.csv}\n"
            f"- shape: {n_rows} x {n_cols}\n"
            f"- n_columns: {len(cols)}\n"
            f"- n_numeric_summarized: {len(numeric_summary)}\n"
            "\nTop missing columns:\n"
            + ("\n".join(top_missing_lines) if top_missing_lines else "(none)\n")
        )

        artifacts = {
            "csv": req.csv,
            "shape": {"n_rows": n_rows, "n_cols": n_cols},
            "columns": cols,
            "missing": {
                "n_missing_total": int(df.isna().sum().sum()),
                "missing_by_col": {k: int(v) for k, v in miss_by_col.to_dict().items()},
                "missing_rate_by_col": {k: float(v) for k, v in miss_rate_by_col.to_dict().items()},
            },
            "numeric_summary": numeric_summary,
            "columns_summarized": target_cols,
        }

        return ToolResult(
            status="ok",
            selected_tool=self.name,
            stdout=stdout,
            stderr="",
            exit_code=0,
            artifacts=artifacts,
        )
