from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.agent.schemas_io import ToolResult
from src.agent.tools.base import BaseTool


class LinearRegressionTool(BaseTool):
    capability_id = "linear_regression"
    name = "linear_regression"

    def validate(self, req):
        missing: List[str] = []
        if not getattr(req, "csv", None):
            missing.append("csv")
        if not getattr(req, "outcome", None):
            missing.append("outcome")
        covs = getattr(req, "covariates", None)
        if not covs or not isinstance(covs, list) or len(covs) == 0:
            missing.append("covariates")
        if missing:
            return False, f"Missing required fields for linear_regression: {missing}"
        return True, ""

    def run(self, req):
        df = pd.read_csv(req.csv)

        y_col = req.outcome
        x_cols = req.covariates

        missing_cols = [c for c in [y_col] + x_cols if c not in df.columns]
        if missing_cols:
            return ToolResult(
                status="error",
                selected_tool=self.name,
                stdout="",
                stderr=f"Columns not found in CSV: {missing_cols}",
                exit_code=2,
                artifacts={"missing_columns": missing_cols},
                warnings=[f"Missing columns: {missing_cols}"],
            )

        sub = df[[y_col] + x_cols].dropna()
        if sub.shape[0] < 10:
            return ToolResult(
                status="error",
                selected_tool=self.name,
                stdout="",
                stderr=f"Not enough complete cases after dropping NA: n={sub.shape[0]}",
                exit_code=2,
                artifacts={"n_complete_cases": int(sub.shape[0])},
                warnings=["Too few complete cases."],
            )

        y = sub[y_col].values.astype(float)
        X = sub[x_cols].values.astype(float)

        model = LinearRegression()
        model.fit(X, y)

        r2 = float(model.score(X, y))
        coef = model.coef_.reshape(-1).tolist()
        intercept = float(model.intercept_)

        artifacts = {
            "n_complete_cases": int(sub.shape[0]),
            "outcome": y_col,
            "covariates": x_cols,
            "intercept": intercept,
            "coef": dict(zip(x_cols, coef)),
            "r2_in_sample": r2,
        }

        stdout_lines = [
            "Linear regression (MVP, in-sample):",
            f"- n complete cases: {sub.shape[0]}",
            f"- outcome: {y_col}",
            f"- covariates: {x_cols}",
            f"- R^2: {r2:.3f}",
        ]

        return ToolResult(
            status="ok",
            selected_tool=self.name,
            stdout="\n".join(stdout_lines),
            stderr="",
            exit_code=0,
            artifacts=artifacts,
            warnings=["Metrics are in-sample (no train/test split)."],
        )
