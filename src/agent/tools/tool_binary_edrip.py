from __future__ import annotations

import json
import subprocess
import sys
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class BinaryEDRIPTool(BaseTool):
    @property
    def name(self) -> str:
        return "binary_edrip"

    @property
    def capability_id(self) -> str:
        return "binary_edrip"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.csv:
            return False, "binary_edrip requires csv."
        if not req.treatment or not req.outcome:
            return False, "binary_edrip requires treatment and outcome."
        covs = getattr(req, "covariates", None)
        if not covs or not isinstance(covs, list) or len(covs) == 0:
            return False, "binary_edrip requires covariates."
        return True, "ok"

    def run(self, req: RunRequest) -> ToolResult:
        cmd = [
            sys.executable,
            "src/run_binary_edrip.py",
            "--csv", req.csv,
            "--treatment", req.treatment or "",
            "--outcome", req.outcome or "",
            "--covariates", ",".join(req.covariates or []),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = p.stdout, p.stderr

        artifacts = {}
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
    
    