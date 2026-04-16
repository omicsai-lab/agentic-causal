# src/agent/tools/tool_survival_adjusted_curves.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class AdjustedCurvesTool(BaseTool):
    @property
    def name(self) -> str:
        return "adjustedcurves"

    @property
    def capability_id(self) -> str:
        return "survival_adjusted_curves"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.csv:
            return False, "Survival requires csv."
        if not (req.time and req.event and req.group):
            return False, "Survival requires time, event, and group."
        return True, "ok"

    def run(self, req: RunRequest) -> ToolResult:
        out_dir = Path("out") / "adjustedcurves_runtime"
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts_file = out_dir / "artifacts.json"
        if artifacts_file.exists():
            artifacts_file.unlink()

        cmd = [
            sys.executable,
            "src/run_adjustedcurves_demo.py",
            "--csv", req.csv,
            "--time", req.time or "",
            "--event", req.event or "",
            "--group", req.group or "",
            "--covariates", ",".join(req.covariates or []),
            "--out_dir", str(out_dir),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = p.stdout, p.stderr

        artifacts = {}
        if artifacts_file.exists():
            try:
                artifacts = json.loads(artifacts_file.read_text(encoding="utf-8"))
            except Exception:
                artifacts = {}

        # 再做一层兜底：如果 artifacts.json 没读到，就尝试从 stdout 最后一行取 JSON
        if not artifacts:
            for line in reversed(stdout.splitlines()):
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        artifacts = json.loads(s)
                        break
                    except Exception:
                        pass

        status = "ok" if p.returncode == 0 else "error"
        return ToolResult(
            status=status,
            selected_tool=self.name,
            stdout=stdout,
            stderr=stderr,
            exit_code=p.returncode,
            artifacts=artifacts,
        )