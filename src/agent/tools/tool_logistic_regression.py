from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.agent.schemas_io import ToolResult
from src.agent.tools.base import BaseTool


class LogisticRegressionTool(BaseTool):
    capability_id = "logistic_regression"
    name = "logistic_regression"

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
            return False, f"Missing required fields for logistic_regression: {missing}"
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

        y = sub[y_col].values
        # enforce binary 0/1
        uniq = np.unique(y)
        if len(uniq) != 2:
            return ToolResult(
                status="error",
                selected_tool=self.name,
                stdout="",
                stderr=f"Outcome must be binary for logistic regression. Unique values: {uniq.tolist()}",
                exit_code=2,
                artifacts={"unique_outcome_values": uniq.tolist()},
                warnings=["Non-binary outcome."],
            )

        X = sub[x_cols].values

        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        p = model.predict_proba(X)[:, 1]
        yhat = (p >= 0.5).astype(int)

        # metrics 
        acc = float(accuracy_score(y, yhat))
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = None

        coef = model.coef_.reshape(-1).tolist()
        intercept = float(model.intercept_[0])

        # odds ratios
        or_list = (np.exp(model.coef_.reshape(-1))).tolist()

        artifacts = {
            "n_complete_cases": int(sub.shape[0]),
            "outcome": y_col,
            "covariates": x_cols,
            "intercept": intercept,
            "coef": dict(zip(x_cols, coef)),
            "odds_ratio": dict(zip(x_cols, or_list)),
            "metrics_in_sample": {"accuracy": acc, "auc": auc},
        }

        stdout_lines = [
            "Logistic regression (MVP, in-sample):",
            f"- n complete cases: {sub.shape[0]}",
            f"- outcome: {y_col}",
            f"- covariates: {x_cols}",
            f"- accuracy: {acc:.3f}",
        ]
        if auc is not None:
            stdout_lines.append(f"- AUC: {auc:.3f}")

        return ToolResult(
            status="ok",
            selected_tool=self.name,
            stdout="\n".join(stdout_lines),
            stderr="",
            exit_code=0,
            artifacts=artifacts,
            warnings=["Metrics are in-sample (no train/test split)."],
        )
