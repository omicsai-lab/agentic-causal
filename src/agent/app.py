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

# import the *capability* router (returns {"capability_id", "reason"})
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

    # de-dup while preserving order
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

    Priority:
      1) explicit capability fields
      2) capability_id-based defaults
      3) tool_name-based defaults

    Supported examples:
      - plot_module = "plot_binary_edrip"
      - plot_name = "plot_binary_edrip"
      - plot_file = "plot_binary_edrip.py"
      - capability_id = "binary_edrip" -> try plot_binary_edrip, binary_edrip
      - tool_name = "binary_edrip" -> try plot_binary_edrip, binary_edrip
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

    Design goals:
    - Never second-guess backend capability existence using guessed tool filenames.
    - Newly added tools should appear automatically as long as a valid cap_*.json exists.
    - Plot support should work for both:
        1) explicit capability metadata
        2) conventional files in src/agent/plots/
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

        # Consider plot support true if:
        # - a plot file is explicitly declared in capability metadata, or
        # - a matching plot file exists in the plots folder.
        explicit_plot_declared = any(
            _normalize_module_name(cap.get(k))
            for k in ["plot_module", "plot_name", "plot_file"]
        )
        plot_supported = explicit_plot_declared or (matched_plot is not None)

        notes_parts: List[str] = []
        if description:
            notes_parts.append(description)
        if matched_plot:
            notes_parts.append(f"Matched plot: {matched_plot}.py")
        elif explicit_plot_declared:
            notes_parts.append("Declared plot metadata found, but matching plot file was not found in src/agent/plots.")

        rows.append(
            {
                "tool": display_name,
                "status": "Registered",
                "plot": "Yes" if plot_supported else "No",
                "notes": " ".join(notes_parts).strip(),
            }
        )

    rows.sort(key=lambda x: x["tool"].lower())
    return rows


def select_capability(req: RunRequest) -> Tuple[str, str, str]:
    """
    Minimal routing (LLM-first):

      Priority:
        1) req.capability_id (force)
        2) LLM router if use_llm_router + request
        3) req.task (explicit)
        4) rule-based auto

    Returns: (capability_id, selected_by, router_reason)
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
            cap_id = (obj.get("capability_id") or "").strip()
            reason = (obj.get("reason") or "").strip() or "LLM selected capability."
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
        # Never break the API response because of file I/O.
        pass

    return run_id, str(out_dir)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def list_tools():
    """
    Return the tools that the backend currently knows about.

    This is the source of truth for the Manage Tools UI.
    It is intentionally capability-driven so that newly added tools
    appear automatically after the backend reloads capability files.
    """
    rows = _tool_rows_from_capabilities()
    return {
        "status": "ok",
        "count": len(rows),
        "tools": rows,
    }


@app.post("/run", response_model=RunResult)
def run(req: RunRequest):
    # --- capability selection ---
    cap_id, selected_by, router_reason = select_capability(req)

    allowed = _allowed_capability_ids()
    allowed_set = set(allowed)

    # Build payload early so even errors can be persisted.
    payload = req.model_dump()

    # Keep user-facing output controls in payload for downstream use / persistence
    generate_plots = req.generate_plots

    # Force graph to use the already-selected capability and avoid re-routing
    payload["capability_id"] = cap_id
    payload["use_llm_router"] = False

    # Preserve generate_plots explicitly
    payload["generate_plots"] = generate_plots

    # --- reject unknown capability_id ---
    if cap_id not in allowed_set:
        artifacts = {
            "capability_id": cap_id,
            "selected_by": selected_by,
            "router_reason": router_reason,
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

    # Always attach router metadata (app-level router)
    artifacts: Dict[str, Any] = {
        **base_artifacts,
        "capability_id": cap_id,
        "selected_by": selected_by,
        "router_reason": router_reason,
        "generate_plots": generate_plots,
    }

    stdout = str(tool_result.get("stdout", "")) if isinstance(tool_result, dict) else ""
    stderr = str(tool_result.get("stderr", "")) if isinstance(tool_result, dict) else ""

    # Tool exit handling
    code = tool_result.get("exit_code", 1) if isinstance(tool_result, dict) else 1
    status = out.get("status") if isinstance(out, dict) else "error"

    # Determine final API status/error
    if status != "ok" or code != 0:
        api_status = "error"
        api_error: Any = f"Tool failed (status={status}, exit_code={code})"
        http_code = 500
    else:
        api_status = "ok"
        api_error = None
        http_code = 200

    # --- persist to out/api/<run_id>/ ---
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

    # --- create user-friendly outputs ---
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

    # --- return ---
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