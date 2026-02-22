from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.registry import register as register_tool



class DummyEchoTool:
    capability_id = "dummy_echo"
    name = "dummy_echo"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.request or not str(req.request).strip():
            return False, "dummy_echo requires a non-empty 'request'."
        return True, ""

    def run(self, req: RunRequest) -> ToolResult:
        shape = None
        columns = None

        if req.csv:
            try:
                df = pd.read_csv(req.csv)
                shape = [df.shape[0], df.shape[1]]
                columns = list(df.columns)
            except Exception as e:
                return ToolResult(
                    status="ok",
                    selected_tool=self.name,
                    stdout=f"Dummy echo (CSV read failed): {req.request}",
                    stderr=str(e),
                    exit_code=0,
                    artifacts={
                        "capability_id": "dummy_echo",
                        "echo": req.request,
                        "csv_error": str(e),
                    },
                )

        artifacts: Dict[str, Any] = {
            "capability_id": "dummy_echo",
            "echo": req.request,
            "csv": req.csv,
            "shape": shape,
            "columns": columns,
        }

        stdout = [
            "Dummy echo tool:",
            f"- echo: {req.request}",
        ]
        if shape:
            stdout.append(f"- dataset shape: {shape[0]} x {shape[1]}")
        if columns:
            stdout.append(f"- columns (first 10): {columns[:10]}")

        return ToolResult(
            status="ok",
            selected_tool=self.name,
            stdout="\n".join(stdout),
            stderr="",
            exit_code=0,
            artifacts=artifacts,
        )


register_tool(DummyEchoTool())

