import gradio as gr
import requests
import json
from pathlib import Path
from typing import Dict, Any, List

# =========================
# Backend Config
# =========================
BACKEND_URL = "http://127.0.0.1:8000/run"

# =========================
# Repo Path Utilities
# =========================
def REPO_ROOT() -> Path:
    return Path(__file__).parent.resolve()

def CAP_DIR() -> Path:
    # 你的真实结构：src/agent/capabilities
    return REPO_ROOT() / "src" / "agent" / "capabilities"

def TOOL_DIR() -> Path:
    # 你的真实结构：src/agent/tools
    return REPO_ROOT() / "src" / "agent" / "tools"

# =========================
# Capability Helpers
# =========================
def load_capability_json(capability_id: str) -> Dict[str, Any]:
    """Load cap_*.json by capability_id."""
    for p in CAP_DIR().glob("cap_*.json"):
        try:
            with open(p, "r") as f:
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

def updates_for_missing(missing: List[str]):
    """
    Return gr.update(visible=...) for each parameter component + extra_json.
    Order: treatment, outcome, covariates, time, event, group, extra_json
    """
    miss = set(missing)
    show_extra = any(f not in KNOWN_FIELDS for f in miss) and len(miss) > 0

    return (
        gr.update(visible=("treatment" in miss)),
        gr.update(visible=("outcome" in miss)),
        gr.update(visible=("covariates" in miss)),
        gr.update(visible=("time" in miss)),
        gr.update(visible=("event" in miss)),
        gr.update(visible=("group" in miss)),
        gr.update(visible=show_extra),
    )

def hide_all_param_boxes():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

# =========================
# Core Run Logic
# =========================
def run_backend(
    csv_file,
    request_text,
    treatment,
    outcome,
    covariates,
    time,
    event,
    group,
    extra_json_text,
):
    # 1) CSV missing -> ask upload
    if csv_file is None:
        u_t, u_o, u_c, u_time, u_e, u_g, u_x = hide_all_param_boxes()
        return (
            "Please upload a CSV file.",
            "",
            u_t, u_o, u_c, u_time, u_e, u_g, u_x,
        )

    # 2) Build payload (only include what user filled)
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

    # Extra JSON (universal fallback for new tools)
    if extra_json_text:
        try:
            extra = json.loads(extra_json_text)
            if isinstance(extra, dict):
                payload.update(extra)
            else:
                u_t, u_o, u_c, u_time, u_e, u_g, _ = hide_all_param_boxes()
                return (
                    "Invalid JSON: must be a JSON object (dict).",
                    "",
                    u_t, u_o, u_c, u_time, u_e, u_g, gr.update(visible=True),
                )
        except Exception:
            u_t, u_o, u_c, u_time, u_e, u_g, _ = hide_all_param_boxes()
            return (
                "Invalid JSON in Extra Parameters.",
                "",
                u_t, u_o, u_c, u_time, u_e, u_g, gr.update(visible=True),
            )

    # 3) Call backend
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=180)
        result = r.json()
    except Exception as e:
        u_t, u_o, u_c, u_time, u_e, u_g, u_x = hide_all_param_boxes()
        return (
            f"Backend error: {type(e).__name__}: {e}",
            "",
            u_t, u_o, u_c, u_time, u_e, u_g, u_x,
        )

    pretty = json.dumps(result, indent=2, ensure_ascii=False)

    # 4) Success -> hide all parameter boxes
    if result.get("status") == "ok":
        u_t, u_o, u_c, u_time, u_e, u_g, u_x = hide_all_param_boxes()
        return (
            "Success ✅",
            pretty,
            u_t, u_o, u_c, u_time, u_e, u_g, u_x,
        )

    # 5) Error -> show only missing required params for selected capability
    cap_id = result.get("artifacts", {}).get("capability_id")
    if not cap_id:
        u_t, u_o, u_c, u_time, u_e, u_g, u_x = hide_all_param_boxes()
        return (
            "Error (no capability detected)",
            pretty,
            u_t, u_o, u_c, u_time, u_e, u_g, u_x,
        )

    cap = load_capability_json(cap_id)
    missing = missing_required_fields(cap, payload)

    # If cap json isn't found or required_fields missing:
    if not cap or not cap.get("required_fields"):
        # still allow user to pass parameters via Extra JSON
        missing = ["(unknown required fields)"]

    u_t, u_o, u_c, u_time, u_e, u_g, u_x = updates_for_missing(missing)
    msg = f"Need parameters: {', '.join(missing)}"
    return (
        msg,
        pretty,
        u_t, u_o, u_c, u_time, u_e, u_g, u_x,
    )

# =========================
# Add Tool Logic (Gradio 6 compatible)
# =========================
def _extract_name_and_bytes(file_obj):
    """
    Gradio 6 File component may return:
    - NamedString (behaves like str)
    - str path
    - dict with 'path'
    - file-like with .name and .read()
    """
    # NamedString behaves like str
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

    # file-like fallback
    name = Path(getattr(file_obj, "name", "uploaded_file")).name
    data = file_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return name, data

def add_tool(cap_file, tool_file):
    try:
        if cap_file is None or tool_file is None:
            return "Please upload both cap_*.json and tool_*.py"

        CAP_DIR().mkdir(parents=True, exist_ok=True)
        TOOL_DIR().mkdir(parents=True, exist_ok=True)

        cap_name, cap_bytes = _extract_name_and_bytes(cap_file)
        tool_name, tool_bytes = _extract_name_and_bytes(tool_file)

        # Basic validation
        if not cap_name.startswith("cap_") or not cap_name.endswith(".json"):
            return f"Invalid capability file name: {cap_name} (expected cap_*.json)"
        if not tool_name.startswith("tool_") or not tool_name.endswith(".py"):
            return f"Invalid tool file name: {tool_name} (expected tool_*.py)"

        cap_path = CAP_DIR() / cap_name
        tool_path = TOOL_DIR() / tool_name

        cap_path.write_bytes(cap_bytes)
        tool_path.write_bytes(tool_bytes)

        return (
            "Added successfully.\n"
            f"- {cap_path}\n"
            f"- {tool_path}\n\n"
            "Restart backend to activate."
        )
    except Exception as e:
        return f"Error while adding tool: {type(e).__name__}: {e}"

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# Causal Agent UI (Minimal Version)")

    with gr.Tab("Run"):
        csv_input = gr.File(label="Upload CSV")
        request_input = gr.Textbox(label="Natural Language Request", lines=2)

        run_button = gr.Button("Run", variant="primary")

        status_output = gr.Textbox(label="Status")
        result_output = gr.Textbox(label="Full JSON Output", lines=18)

        gr.Markdown("### Parameters (only shown if required)")
        treatment_input = gr.Textbox(label="treatment", visible=False)
        outcome_input = gr.Textbox(label="outcome", visible=False)
        covariates_input = gr.Textbox(label="covariates (comma-separated)", visible=False)
        time_input = gr.Textbox(label="time", visible=False)
        event_input = gr.Textbox(label="event", visible=False)
        group_input = gr.Textbox(label="group", visible=False)

        extra_json_input = gr.Textbox(
            label="Extra Parameters (JSON)",
            placeholder='{"alpha": 0.05}',
            visible=False,
        )

        run_button.click(
            run_backend,
            inputs=[
                csv_input,
                request_input,
                treatment_input,
                outcome_input,
                covariates_input,
                time_input,
                event_input,
                group_input,
                extra_json_input,
            ],
            outputs=[
                status_output,
                result_output,
                treatment_input,
                outcome_input,
                covariates_input,
                time_input,
                event_input,
                group_input,
                extra_json_input,
            ],
        )

    with gr.Tab("Add Tool"):
        cap_upload = gr.File(label="Upload cap_*.json")
        tool_upload = gr.File(label="Upload tool_*.py")
        add_button = gr.Button("Add Tool", variant="primary")
        add_status = gr.Textbox(label="Status", lines=6)

        add_button.click(
            add_tool,
            inputs=[cap_upload, tool_upload],
            outputs=[add_status],
        )

demo.launch()