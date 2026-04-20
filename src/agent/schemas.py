from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# =========================
# Run Request (FastAPI input)
# =========================

class RunRequest(BaseModel):
    csv: str

    request: str = ""
    capability_id: Optional[str] = None
    use_llm_router: bool = True
    llm_model: Optional[str] = None

    # Explicit task shortcut
    task: Optional[str] = None

    # User-facing output controls
    generate_plots: bool = True

    # ATE fields
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    covariates: Optional[List[str]] = None

    # Survival fields
    time: Optional[str] = None
    event: Optional[str] = None
    group: Optional[str] = None


# =========================
# Run Result (FastAPI output)
# =========================

class RunResult(BaseModel):
    status: Literal["ok", "error"]

    selected_tool: Optional[str] = None

    stdout: str = ""
    stderr: str = ""

    artifacts: Dict[str, Any] = Field(default_factory=dict)

    user_summary: Optional[str] = None
    graph_paths: List[str] = Field(default_factory=list)
    report_pdf: Optional[str] = None

    error: Optional[str] = None