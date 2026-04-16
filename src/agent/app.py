from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import RunRequest, RunResult
from .graph import graph
from .report_utils import create_user_outputs
from .planner_llm import llm_generate_analysis_plan

# import the capability router (returns {"capability_id", "reason"})
try:
    from .router_llm import llm_choose_capability  # type: ignore
except Exception:
    llm_choose_capability = None  # type: ignore

# import capability loader (cap_*.json)
try:
    from .router_llm import load_capabilities  # type: ignore
except Exception:
    load_capabilities = None  # type: ignore


app = FastAPI(title="Causal Agent MVP")


def _repo_root() -> Path:
    # src/agent/app.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _capabilities_dir() -> Path:
    return _repo_root() / "src" / "agent" / "capabilities"


def _plots_dir() -> Path:
    return _repo_root() / "src" / "agent" / "plots"


def _load_capabilities_fallback() -> List[Dict[str, Any]]:
    """
    Fallback loader if router_llm.load_capabilities() is unavailable.
    Reads src/agent/capabilities/cap_*.json
    """
    caps_dir = _capabilities_dir()
    if not caps_dir.exists():
        return []

    out: List[Dict[str, Any]] = []
    for p in sorted(caps_dir.glob("cap_*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _load_capability_specs() -> List[Dict[str, Any]]:
    """
    Source of truth: src/agent/capabilities/cap_*.json
    Prefer router_llm.load_capabilities() if available; otherwise fallback.
    """
    if load_capabilities is not None:
        try:
            caps = load_capabilities()
            if isinstance(caps, list):
                return caps
        except Exception:
            pass
    return _load_capabilities_fallback()


def _capability_spec_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cap in _load_capability_specs():
        if not isinstance(cap, dict):
            continue
        cid = str(cap.get("capability_id") or cap.get("id") or "").strip()
        if cid:
            out[cid] = cap
    return out


def _allowed_capability_ids() -> List[str]:
    """
    Return allowed capability ids from cap specs.
    We accept either {"capability_id": ...} or {"id": ...}.
    """
    ids: List[str] = []
    for c in _load_capability_specs():
        if not isinstance(c, dict):
            continue
        cid = (c.get("capability_id") or c.get("id") or "").strip()
        if cid:
            ids.append(cid)

    seen = set()
    uniq: List[str] = []
    for x in ids:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _existing_plot_stems() -> set[str]:
    plots_dir = _plots_dir()
    if not plots_dir.exists():
        return set()
    return {p.stem for p in plots_dir.glob("*.py")}


def _normalize_module_name(x: Any) -> str:
    s = str(x or "").strip()
    if s.endswith(".py"):
        s = s[:-3]
    return s


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _candidate_plot_names_for_capability(cap: Dict[str, Any]) -> List[str]:
    """
    Build flexible plot candidates so newly added tools can work with minimal config.
    """
    capability_id = str(cap.get("capability_id") or cap.get("id") or "").strip()
    tool_name = str(cap.get("tool_name") or "").strip()

    explicit = [
        _normalize_module_name(cap.get("plot_module")),
        _normalize_module_name(cap.get("plot_name")),
        _normalize_module_name(cap.get("plot_file")),
    ]

    inferred: List[str] = []
    if capability_id:
        inferred.extend([f"plot_{capability_id}", capability_id])
    if tool_name and tool_name != capability_id:
        inferred.extend([f"plot_{tool_name}", tool_name])

    return _unique_preserve_order(explicit + inferred)


def _tool_rows_from_capabilities() -> List[Dict[str, str]]:
    """
    Build the Manage Tools table directly from the same capability source
    used by the backend.
    """
    rows: List[Dict[str, str]] = []
    plot_stems = _existing_plot_stems()

    for cap in _load_capability_specs():
        if not isinstance(cap, dict):
            continue

        capability_id = str(cap.get("capability_id") or cap.get("id") or "").strip()
        tool_name = str(cap.get("tool_name") or "").strip()
        display_name = tool_name or capability_id or "unknown_tool"

        description = str(cap.get("description") or "").strip()

        candidate_plot_names = _candidate_plot_names_for_capability(cap)
        matched_plot = next((name for name in candidate_plot_names if name in plot_stems), None)

        explicit_plot_declared = any(
            _normalize_module_name(cap.get(k))
            for k in ["plot_module", "plot_name", "plot_file"]
        )

        notes_parts: List[str] = []
        if description:
            notes_parts.append(description)
        if matched_plot:
            notes_parts.append(f"Matched plot: {matched_plot}.py")
        elif explicit_plot_declared:
            notes_parts.append(
                "Declared plot metadata found, but matching plot file was not found in src/agent/plots."
            )

        rows.append(
            {
                "tool": display_name,
                "status": "Registered",
                "notes": " ".join(notes_parts).strip(),
            }
        )

    rows.sort(key=lambda x: x["tool"].lower())
    return rows


def build_analysis_plan(req: RunRequest) -> Dict[str, Any]:
    """
    Build a user-facing analysis plan from the natural-language request.
    This plan is for explanation/display, not for final tool routing.
    """
    request_text = (req.request or "").strip()

    if not request_text:
        return {
            "analysis_goal": "",
            "task_type": "unknown",
            "target_estimand": "",
            "outcome_type": "unknown",
            "required_fields": [],
            "optional_fields": [],
            "assumptions": [],
            "reasoning": "No natural-language request was provided.",
        }

    try:
        plan = llm_generate_analysis_plan(
            request=request_text,
            model=req.llm_model,
        )
        if isinstance(plan, dict):
            return {
                "analysis_goal": str(plan.get("analysis_goal", "")).strip(),
                "task_type": str(plan.get("task_type", "unknown")).strip() or "unknown",
                "target_estimand": str(plan.get("target_estimand", "")).strip(),
                "outcome_type": str(plan.get("outcome_type", "unknown")).strip() or "unknown",
                "required_fields": [],
                "optional_fields": [],
                "assumptions": [
                    str(x).strip()
                    for x in (plan.get("assumptions", []) or [])
                    if str(x).strip()
                ],
                "reasoning": str(plan.get("reasoning", "")).strip(),
            }
    except Exception:
        pass

    return {
        "analysis_goal": "",
        "task_type": "unknown",
        "target_estimand": "",
        "outcome_type": "unknown",
        "required_fields": [],
        "optional_fields": [],
        "assumptions": [],
        "reasoning": "Planner failed; no structured plan was produced.",
    }


def finalize_analysis_plan(
    plan: Dict[str, Any],
    capability_id: str,
    router_reason: str,
) -> Dict[str, Any]:
    """
    Make the displayed analysis plan consistent with the actually selected tool.
    Capability JSON is the source of truth for required/optional fields.
    """
    cap = _capability_spec_map().get(capability_id, {})

    merged = dict(plan or {})
    merged["recommended_tool"] = capability_id

    required_fields = cap.get("required_fields", [])
    optional_fields = cap.get("optional_fields", [])

    if isinstance(required_fields, list):
        merged["required_fields"] = [
            str(x).strip()
            for x in required_fields
            if str(x).strip() and str(x).strip() != "csv"
        ]
    else:
        merged["required_fields"] = []

    if isinstance(optional_fields, list):
        merged["optional_fields"] = [
            str(x).strip()
            for x in optional_fields
            if str(x).strip()
        ]
    else:
        merged["optional_fields"] = []

    # Prefer router reason as the displayed justification for the selected tool.
    merged["reasoning"] = router_reason or str(merged.get("reasoning", "")).strip()

    return merged


def select_capability(req: RunRequest) -> Tuple[str, str, str]:
    """
    Tool routing only. Analysis plan is handled separately for display.

    Priority:
      1) req.capability_id (force)
      2) LLM router if use_llm_router + request
      3) req.task (explicit)
      4) rule-based auto

    Returns:
      (capability_id, selected_by, router_reason)
    """
    # 1) forced
    if req.capability_id:
        return req.capability_id, "capability_id", "Forced by capability_id."

    # 2) LLM router
    if req.use_llm_router and req.request and llm_choose_capability is not None:
        try:
            obj = llm_choose_capability(
                request=req.request,
                csv_columns=None,
                model=req.llm_model,
            )
            cap_id = str(obj.get("capability_id", "")).strip()
            reason = str(obj.get("reason", "")).strip() or "LLM selected capability."
            if cap_id:
                return cap_id, "llm", reason
        except Exception:
            pass

    # 3) explicit task
    if req.task == "ate":
        return "causal_ate", "task", "Selected by explicit task='ate'."
    if req.task == "survival":
        return "survival_adjusted_curves", "task", "Selected by explicit task='survival'."

    # 4) rule-based auto
    if req.time and req.event and req.group:
        return "survival_adjusted_curves", "auto", "Auto: time/event/group detected."

    return "causal_ate", "auto", "Auto: defaulting to causal_ate."


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _persist_api_outputs(
    *,
    payload: Dict[str, Any],
    status: str,
    selected_tool: Any,
    stdout: str,
    stderr: str,
    artifacts: Dict[str, Any],
    error: Any,
) -> Tuple[str, str]:
    """
    Persist request/result/stdout/stderr into:
      <repo_root>/out/api/<run_id>/
    Returns: (run_id, out_dir_str)
    """
    run_id = _new_run_id()
    out_base = _repo_root() / "out" / "api"
    out_dir = out_base / run_id

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "request.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        (out_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
        (out_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")

        (out_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "selected_tool": selected_tool,
                    "stdout": stdout,
                    "stderr": stderr,
                    "artifacts": artifacts,
                    "error": error,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    return run_id, str(out_dir)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def list_tools():
    rows = _tool_rows_from_capabilities()
    return {
        "status": "ok",
        "count": len(rows),
        "tools": rows,
    }


@app.post("/run", response_model=RunResult)
def run(req: RunRequest):
    # --- tool selection only ---
    cap_id, selected_by, router_reason = select_capability(req)

    # --- user-facing analysis plan ---
    raw_analysis_plan = build_analysis_plan(req)
    analysis_plan = finalize_analysis_plan(raw_analysis_plan, cap_id, router_reason)

    allowed = _allowed_capability_ids()
    allowed_set = set(allowed)

    payload = req.model_dump()
    generate_plots = req.generate_plots

    # Force graph to use the already-selected capability and avoid re-routing
    payload["capability_id"] = cap_id
    payload["use_llm_router"] = False
    payload["generate_plots"] = generate_plots

    # --- reject unknown capability_id ---
    if cap_id not in allowed_set:
        artifacts = {
            "capability_id": cap_id,
            "selected_by": selected_by,
            "router_reason": router_reason,
            "analysis_plan": analysis_plan,
            "allowed_capabilities": allowed,
            "generate_plots": generate_plots,
        }
        err_msg = (
            f"Unknown capability_id='{cap_id}'. "
            f"Check src/agent/capabilities/cap_*.json."
        )

        run_id, out_dir = _persist_api_outputs(
            payload=payload,
            status="error",
            selected_tool=None,
            stdout="",
            stderr="",
            artifacts={**artifacts, "run_id": "", "out_dir": ""},
            error=err_msg,
        )
        artifacts["run_id"] = run_id
        artifacts["out_dir"] = out_dir

        return JSONResponse(
            status_code=400,
            content=RunResult(
                status="error",
                selected_tool=None,
                stdout="",
                stderr="",
                artifacts=artifacts,
                user_summary=artifacts.get("user_summary"),
                graph_paths=artifacts.get("graph_paths", []),
                report_pdf=artifacts.get("report_pdf"),
                error=err_msg,
            ).model_dump(),
        )

    # --- execute ---
    out = graph.invoke({"req": payload})

    tool_result = out.get("tool_result", {}) if isinstance(out, dict) else {}
    selected_tool = out.get("selected_tool") if isinstance(out, dict) else None

    base_artifacts = tool_result.get("artifacts", {}) if isinstance(tool_result, dict) else {}
    if not isinstance(base_artifacts, dict):
        base_artifacts = {}

    artifacts: Dict[str, Any] = {
        **base_artifacts,
        "capability_id": cap_id,
        "selected_by": selected_by,
        "router_reason": router_reason,
        "analysis_plan": analysis_plan,
        "generate_plots": generate_plots,
    }

    stdout = str(tool_result.get("stdout", "")) if isinstance(tool_result, dict) else ""
    stderr = str(tool_result.get("stderr", "")) if isinstance(tool_result, dict) else ""

    code = tool_result.get("exit_code", 1) if isinstance(tool_result, dict) else 1
    status = out.get("status") if isinstance(out, dict) else "error"

    if status != "ok" or code != 0:
        api_status = "error"
        api_error: Any = f"Tool failed (status={status}, exit_code={code})"
        http_code = 500
    else:
        api_status = "ok"
        api_error = None
        http_code = 200

    run_id, out_dir = _persist_api_outputs(
        payload=payload,
        status=api_status,
        selected_tool=selected_tool,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        error=api_error,
    )
    artifacts["run_id"] = run_id
    artifacts["out_dir"] = out_dir

    try:
        user_outputs = create_user_outputs(
            out_dir=Path(out_dir),
            status=api_status,
            selected_tool=selected_tool,
            capability_id=cap_id,
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
            generate_plots=generate_plots,
        )
        artifacts.update(user_outputs)
    except Exception as e:
        artifacts["user_output_error"] = str(e)

    if api_status != "ok":
        return JSONResponse(
            status_code=http_code,
            content=RunResult(
                status="error",
                selected_tool=selected_tool,
                stdout=stdout,
                stderr=stderr,
                artifacts=artifacts,
                user_summary=artifacts.get("user_summary"),
                graph_paths=artifacts.get("graph_paths", []),
                report_pdf=artifacts.get("report_pdf"),
                error=api_error,
            ).model_dump(),
        )

    return RunResult(
        status="ok",
        selected_tool=selected_tool,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        user_summary=artifacts.get("user_summary"),
        graph_paths=artifacts.get("graph_paths", []),
        report_pdf=artifacts.get("report_pdf"),
        error=None,
    )