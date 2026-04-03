import gradio as gr
import requests
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import html

BACKEND_URL = "http://127.0.0.1:8000/run"
TOOLS_URL = "http://127.0.0.1:8000/tools"


def REPO_ROOT() -> Path:
    return Path(__file__).parent.resolve()


def CAP_DIR() -> Path:
    return REPO_ROOT() / "src" / "agent" / "capabilities"


def TOOL_DIR() -> Path:
    return REPO_ROOT() / "src" / "agent" / "tools"


def PLOT_DIR() -> Path:
    return REPO_ROOT() / "src" / "agent" / "plots"


def ensure_plugin_dirs():
    CAP_DIR().mkdir(parents=True, exist_ok=True)
    TOOL_DIR().mkdir(parents=True, exist_ok=True)
    PLOT_DIR().mkdir(parents=True, exist_ok=True)


def load_capability_json(capability_id: str) -> Dict[str, Any]:
    """Load cap_*.json by capability_id."""
    for p in CAP_DIR().glob("cap_*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                cap = json.load(f)
            if cap.get("capability_id") == capability_id:
                return cap
        except Exception:
            continue
    return {}


def missing_required_fields(cap: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    required = cap.get("required_fields", [])
    missing = []
    for field in required:
        if field == "csv":
            continue
        if field not in payload or payload[field] in [None, "", []]:
            missing.append(field)
    return missing


KNOWN_FIELDS = {"treatment", "outcome", "covariates", "time", "event", "group"}


def updates_for_capability_fields(missing: List[str], optional: List[str]):
    """
    Show required missing fields under Required section,
    and optional fields under Optional section.

    Return order:
    required_header,
    req_treatment, req_outcome, req_covariates, req_time, req_event, req_group,
    optional_header,
    opt_treatment, opt_outcome, opt_covariates, opt_time, opt_event, opt_group,
    extra_json
    """
    missing_set = set(missing)
    optional_set = set(optional)

    required_known = missing_set & KNOWN_FIELDS
    optional_known = optional_set & KNOWN_FIELDS

    unknown_fields = (missing_set | optional_set) - KNOWN_FIELDS
    show_extra = len(unknown_fields) > 0

    show_required_header = len(required_known) > 0
    show_optional_header = len(optional_known) > 0

    return (
        gr.update(visible=show_required_header),
        gr.update(visible=("treatment" in required_known)),
        gr.update(visible=("outcome" in required_known)),
        gr.update(visible=("covariates" in required_known)),
        gr.update(visible=("time" in required_known)),
        gr.update(visible=("event" in required_known)),
        gr.update(visible=("group" in required_known)),
        gr.update(visible=show_optional_header),
        gr.update(visible=("treatment" in optional_known)),
        gr.update(visible=("outcome" in optional_known)),
        gr.update(visible=("covariates" in optional_known)),
        gr.update(visible=("time" in optional_known)),
        gr.update(visible=("event" in optional_known)),
        gr.update(visible=("group" in optional_known)),
        gr.update(visible=show_extra),
    )


def hide_all_param_boxes():
    return (
        gr.update(visible=False),  # required_header
        gr.update(visible=False),  # req_treatment
        gr.update(visible=False),  # req_outcome
        gr.update(visible=False),  # req_covariates
        gr.update(visible=False),  # req_time
        gr.update(visible=False),  # req_event
        gr.update(visible=False),  # req_group
        gr.update(visible=False),  # optional_header
        gr.update(visible=False),  # opt_treatment
        gr.update(visible=False),  # opt_outcome
        gr.update(visible=False),  # opt_covariates
        gr.update(visible=False),  # opt_time
        gr.update(visible=False),  # opt_event
        gr.update(visible=False),  # opt_group
        gr.update(visible=False),  # extra_json
    )


def pick_value(required_value, optional_value):
    if required_value not in [None, ""]:
        return required_value
    return optional_value


def _find_raw_json_file(result: Dict[str, Any]) -> Optional[str]:
    artifacts = result.get("artifacts", {}) or {}
    out_dir = artifacts.get("out_dir")
    if not out_dir:
        return None

    p = Path(out_dir) / "result.json"
    if p.exists():
        return str(p)
    return None


def run_backend(
    csv_file,
    request_text,
    req_treatment,
    req_outcome,
    req_covariates,
    req_time,
    req_event,
    req_group,
    opt_treatment,
    opt_outcome,
    opt_covariates,
    opt_time,
    opt_event,
    opt_group,
    extra_json_text,
):
    hidden_updates = hide_all_param_boxes()

    if csv_file is None:
        return (
            "Please upload a CSV file.",
            "",
            None,
            None,
            None,
            *hidden_updates,
        )

    treatment = pick_value(req_treatment, opt_treatment)
    outcome = pick_value(req_outcome, opt_outcome)
    covariates = pick_value(req_covariates, opt_covariates)
    time = pick_value(req_time, opt_time)
    event = pick_value(req_event, opt_event)
    group = pick_value(req_group, opt_group)

    payload: Dict[str, Any] = {
        "csv": str(csv_file.name),
        "request": request_text or "",
        "use_llm_router": True,
    }

    if treatment:
        payload["treatment"] = treatment

    if outcome:
        payload["outcome"] = outcome

    if covariates:
        payload["covariates"] = [x.strip() for x in covariates.split(",") if x.strip()]

    if time:
        payload["time"] = time

    if event:
        payload["event"] = event

    if group:
        payload["group"] = group

    if extra_json_text:
        try:
            extra = json.loads(extra_json_text)
            if isinstance(extra, dict):
                payload.update(extra)
            else:
                return (
                    "Invalid JSON: Extra Parameters must be a JSON object.",
                    "",
                    None,
                    None,
                    None,
                    *hidden_updates[:-1],
                    gr.update(visible=True),
                )
        except Exception:
            return (
                "Invalid JSON in Extra Parameters.",
                "",
                None,
                None,
                None,
                *hidden_updates[:-1],
                gr.update(visible=True),
            )

    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=180)
        result = r.json()
    except Exception as e:
        return (
            f"Backend error: {type(e).__name__}: {e}",
            "",
            None,
            None,
            None,
            *hidden_updates,
        )

    summary = result.get("user_summary", "")
    graph_paths = result.get("graph_paths", []) or []
    pdf_path = result.get("report_pdf")
    plot_path = graph_paths[0] if graph_paths else None
    raw_json_path = _find_raw_json_file(result)

    if result.get("status") == "ok":
        return (
            "Success ✅",
            summary,
            plot_path,
            pdf_path,
            raw_json_path,
            *hidden_updates,
        )

    cap_id = result.get("artifacts", {}).get("capability_id")
    if not cap_id:
        return (
            "Error (no capability detected)",
            summary,
            plot_path,
            pdf_path,
            raw_json_path,
            *hidden_updates,
        )

    cap = load_capability_json(cap_id)
    missing = missing_required_fields(cap, payload)
    optional = cap.get("optional_fields", []) if cap else []

    if not cap:
        missing = ["(unknown required fields)"]
        optional = []

    field_updates = updates_for_capability_fields(missing, optional)

    if missing:
        msg = f"Need required parameters: {', '.join(missing)}"
    else:
        msg = "Optional parameters available."

    return (
        msg,
        summary,
        plot_path,
        pdf_path,
        raw_json_path,
        *field_updates,
    )


def _extract_name_and_bytes(file_obj):
    if isinstance(file_obj, str):
        p = Path(str(file_obj))
        if p.exists():
            return p.name, p.read_bytes()
        raise FileNotFoundError(f"Uploaded file path does not exist: {p}")

    if isinstance(file_obj, dict):
        if "path" in file_obj:
            p = Path(file_obj["path"])
            return p.name, p.read_bytes()
        if "name" in file_obj:
            p = Path(file_obj["name"])
            if p.exists():
                return p.name, p.read_bytes()
        raise ValueError(f"Unsupported dict file object: keys={list(file_obj.keys())}")

    name = Path(getattr(file_obj, "name", "uploaded_file")).name
    data = file_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return name, data


def fetch_tools_from_backend() -> List[Dict[str, str]]:
    """
    Get the real registered tool list from the backend.
    """
    try:
        r = requests.get(TOOLS_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        tools = data.get("tools", [])
        if isinstance(tools, list):
            out: List[Dict[str, str]] = []
            for item in tools:
                if isinstance(item, dict):
                    out.append(
                        {
                            "tool": str(item.get("tool", "") or ""),
                            "status": str(item.get("status", "") or ""),
                            "plot": str(item.get("plot", "") or ""),
                            "notes": str(item.get("notes", "") or ""),
                        }
                    )
            return out
    except Exception:
        pass
    return []


def get_tool_summary_table_html() -> str:
    rows = fetch_tools_from_backend()

    if not rows:
        return """
        <div class="tool-table-scroll">
          <div class="tool-empty">
            Could not load tool list from backend. Make sure the FastAPI server is running and includes the /tools endpoint.
          </div>
        </div>
        """

    parts = [
        '<div class="tool-table-scroll">',
        '<table class="tool-table">',
        "<thead>",
        "<tr>",
        "<th>Tool</th>",
        "<th>Status</th>",
        "<th>Plot</th>",
        "<th>Notes</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for row in rows:
        tool = html.escape(row["tool"])
        status = html.escape(row["status"])
        plot = html.escape(row["plot"])
        notes = html.escape(row["notes"])

        parts.extend(
            [
                "<tr>",
                f"<td>{tool}</td>",
                f"<td>{status}</td>",
                f"<td>{plot}</td>",
                f"<td>{notes}</td>",
                "</tr>",
            ]
        )

    parts.extend(
        [
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )

    return "\n".join(parts)


def refresh_tool_table_html():
    return get_tool_summary_table_html()


def add_tool(cap_file, tool_file, plot_file=None):
    try:
        if cap_file is None or tool_file is None:
            return "Please upload both cap_*.json and tool_*.py", get_tool_summary_table_html()

        ensure_plugin_dirs()

        cap_name, cap_bytes = _extract_name_and_bytes(cap_file)
        tool_name, tool_bytes = _extract_name_and_bytes(tool_file)

        if not cap_name.startswith("cap_") or not cap_name.endswith(".json"):
            return (
                f"Invalid capability file name: {cap_name} (expected cap_*.json)",
                get_tool_summary_table_html(),
            )

        if not tool_name.endswith(".py"):
            return (
                f"Invalid tool file name: {tool_name} (expected a .py file)",
                get_tool_summary_table_html(),
            )

        cap_path = CAP_DIR() / cap_name
        tool_path = TOOL_DIR() / tool_name

        cap_path.write_bytes(cap_bytes)
        tool_path.write_bytes(tool_bytes)

        status_lines = [
            "Added successfully.",
            f"- capability: {cap_path}",
            f"- tool: {tool_path}",
        ]

        if plot_file is not None:
            plot_name, plot_bytes = _extract_name_and_bytes(plot_file)

            if not plot_name.endswith(".py"):
                return (
                    f"Invalid plot file name: {plot_name} (expected a .py file)",
                    get_tool_summary_table_html(),
                )

            plot_path = PLOT_DIR() / plot_name
            plot_path.write_bytes(plot_bytes)
            status_lines.append(f"- plot: {plot_path}")
        else:
            status_lines.append("- plot: not uploaded (optional)")

        status_lines.append("")
        status_lines.append("Restart backend to activate newly added tools.")
        status_lines.append("After restart, click Refresh Tool List to re-read the backend registry.")

        return "\n".join(status_lines), get_tool_summary_table_html()

    except Exception as e:
        return f"Error while adding tool: {type(e).__name__}: {e}", get_tool_summary_table_html()


CUSTOM_CSS = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

.gradio-container {
    max-width: 100vw !important;
    width: 100vw !important;
    padding: 16px !important;
    box-sizing: border-box !important;
}

.card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 12px;
    background: white;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.section-label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #374151 !important;
    margin-top: 6px !important;
    margin-bottom: 4px !important;
}

.gr-file-upload {
    font-size: 11px !important;
}

.compact-upload {
    min-height: 120px !important;
}

.compact-upload .wrap,
.compact-upload > div {
    min-height: 120px !important;
}

.status-output textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}

button {
    min-height: 44px !important;
}

.param-box textarea,
.param-box input {
    font-size: 13px !important;
}

.top-row {
    align-items: stretch !important;
}

.left-panel,
.right-panel {
    align-self: stretch !important;
}

.tool-table-scroll {
    width: 100%;
    height: 560px;
    max-height: 560px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}

.tool-table {
    width: 100%;
    min-width: 900px;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 16px;
    background: white;
}

.tool-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: white;
}

.tool-table th,
.tool-table td {
    text-align: left;
    padding: 14px 14px;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}

.tool-table th {
    font-weight: 700;
    font-size: 16px;
}

.tool-table th:nth-child(1),
.tool-table td:nth-child(1) {
    width: 24%;
}

.tool-table th:nth-child(2),
.tool-table td:nth-child(2) {
    width: 16%;
}

.tool-table th:nth-child(3),
.tool-table td:nth-child(3) {
    width: 10%;
}

.tool-table th:nth-child(4),
.tool-table td:nth-child(4) {
    width: 50%;
}

.tool-empty {
    padding: 12px 14px;
    color: #6b7280;
}
"""


with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
# Causal Agent
Upload a dataset, submit an analysis request, and receive user-friendly results.
"""
    )

    with gr.Tabs():
        with gr.Tab("Run Analysis"):
            with gr.Row(elem_classes="top-row"):
                with gr.Column(scale=5, elem_classes="left-panel"):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Input")

                        csv_input = gr.File(
                            label="Dataset (CSV)",
                            height=120,
                            elem_classes="compact-upload",
                        )

                        request_input = gr.Textbox(
                            label="Analysis Request",
                            lines=4,
                            placeholder="e.g. Estimate the average treatment effect of treatment on outcome adjusting for age and bmi.",
                        )

                        run_button = gr.Button("Run Analysis", variant="primary")

                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Advanced Parameters")

                        required_header = gr.Markdown(
                            "**Required**",
                            visible=False,
                            elem_classes="section-label",
                        )

                        req_treatment_input = gr.Textbox(
                            label="treatment",
                            visible=False,
                            elem_classes="param-box",
                        )

                        req_outcome_input = gr.Textbox(
                            label="outcome",
                            visible=False,
                            elem_classes="param-box",
                        )

                        req_covariates_input = gr.Textbox(
                            label="covariates (comma-separated)",
                            visible=False,
                            elem_classes="param-box",
                        )

                        req_time_input = gr.Textbox(
                            label="time",
                            visible=False,
                            elem_classes="param-box",
                        )

                        req_event_input = gr.Textbox(
                            label="event",
                            visible=False,
                            elem_classes="param-box",
                        )

                        req_group_input = gr.Textbox(
                            label="group",
                            visible=False,
                            elem_classes="param-box",
                        )

                        optional_header = gr.Markdown(
                            "**Optional**",
                            visible=False,
                            elem_classes="section-label",
                        )

                        opt_treatment_input = gr.Textbox(
                            label="treatment",
                            visible=False,
                            elem_classes="param-box",
                        )

                        opt_outcome_input = gr.Textbox(
                            label="outcome",
                            visible=False,
                            elem_classes="param-box",
                        )

                        opt_covariates_input = gr.Textbox(
                            label="covariates (comma-separated)",
                            visible=False,
                            elem_classes="param-box",
                        )

                        opt_time_input = gr.Textbox(
                            label="time",
                            visible=False,
                            elem_classes="param-box",
                        )

                        opt_event_input = gr.Textbox(
                            label="event",
                            visible=False,
                            elem_classes="param-box",
                        )

                        opt_group_input = gr.Textbox(
                            label="group",
                            visible=False,
                            elem_classes="param-box",
                        )

                        extra_json_input = gr.Textbox(
                            label="Extra Parameters (JSON)",
                            placeholder='{"alpha": 0.05}',
                            visible=False,
                            lines=4,
                            elem_classes="param-box",
                        )

                with gr.Column(scale=7, elem_classes="right-panel"):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Results")

                        status_output = gr.Textbox(
                            label="Status",
                            lines=2,
                            interactive=False,
                            elem_classes="status-output",
                        )

                        summary_output = gr.Textbox(
                            label="Summary",
                            lines=6,
                            interactive=False,
                        )

                        plot_output = gr.Image(
                            label="Figure",
                            type="filepath",
                        )

                        pdf_output = gr.File(
                            label="Download Project Report (PDF)",
                        )

                        raw_json_output = gr.File(
                            label="Download Raw JSON Output",
                        )

            run_button.click(
                run_backend,
                inputs=[
                    csv_input,
                    request_input,
                    req_treatment_input,
                    req_outcome_input,
                    req_covariates_input,
                    req_time_input,
                    req_event_input,
                    req_group_input,
                    opt_treatment_input,
                    opt_outcome_input,
                    opt_covariates_input,
                    opt_time_input,
                    opt_event_input,
                    opt_group_input,
                    extra_json_input,
                ],
                outputs=[
                    status_output,
                    summary_output,
                    plot_output,
                    pdf_output,
                    raw_json_output,
                    required_header,
                    req_treatment_input,
                    req_outcome_input,
                    req_covariates_input,
                    req_time_input,
                    req_event_input,
                    req_group_input,
                    optional_header,
                    opt_treatment_input,
                    opt_outcome_input,
                    opt_covariates_input,
                    opt_time_input,
                    opt_event_input,
                    opt_group_input,
                    extra_json_input,
                ],
            )

        with gr.Tab("Manage Tools"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Add New Tool")

                        cap_upload = gr.File(
                            label="Upload cap_*.json",
                            height=120,
                            elem_classes="compact-upload",
                        )

                        tool_upload = gr.File(
                            label="Upload tool_*.py",
                            height=120,
                            elem_classes="compact-upload",
                        )

                        plot_upload = gr.File(
                            label="Upload plot_*.py (optional)",
                            height=120,
                            elem_classes="compact-upload",
                        )

                        with gr.Row():
                            add_button = gr.Button("Add Tool", variant="primary")
                            refresh_button = gr.Button("Refresh Tool List")

                with gr.Column(scale=7):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Existing Tools")
                        tool_table_html = gr.HTML(value=get_tool_summary_table_html())

                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### Tool Status")

                        add_status = gr.Textbox(
                            label="Status",
                            lines=14,
                            interactive=False,
                        )

            add_button.click(
                add_tool,
                inputs=[cap_upload, tool_upload, plot_upload],
                outputs=[add_status, tool_table_html],
            )

            refresh_button.click(
                refresh_tool_table_html,
                inputs=[],
                outputs=[tool_table_html],
            )

demo.launch()