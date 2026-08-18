"""
Instrumented, non-invasive wrapper around the frozen production router and
planner functions (src/agent/router_llm.py, src/agent/planner_llm.py).

Design constraint: this file MUST NOT edit those production modules. To get
explicit API-success/fallback/model-substitution telemetry that the
production functions do not themselves return, we temporarily monkeypatch
the `OpenAI` class *reference* those two modules hold in their own module
namespace (`router_llm.OpenAI`, `planner_llm.OpenAI`) with a thin recording
proxy, call the real production function unchanged, then restore the
original reference. The production code path, prompts, and exception
handling all execute exactly as they do in the running application --
only the raw request/response/exception is additionally logged.

This is the same technique as `unittest.mock.patch.object` and is a
standard, safe way to observe a module's behavior without modifying it on
disk.
"""

from __future__ import annotations

import importlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.semantic_orchestration.harness.config import (  # noqa: E402
    REQUESTED_MODEL,
    TEMPERATURE,
    ROUTER_FALLBACK_REASON_MARKERS,
)


@dataclass
class CallTelemetry:
    attempted: bool = False
    api_call_count: int = 0
    api_exception: Optional[str] = None
    api_exception_type: Optional[str] = None
    requested_model: Optional[str] = None
    returned_models: List[str] = field(default_factory=list)
    latency_seconds: Optional[float] = None
    raw_finish_reasons: List[str] = field(default_factory=list)

    @property
    def api_call_succeeded(self) -> bool:
        return self.attempted and self.api_exception is None and self.api_call_count > 0

    @property
    def model_substituted(self) -> bool:
        if not self.returned_models:
            return False
        return any(m != self.requested_model for m in self.returned_models if m)


class _RecordingCompletions:
    def __init__(self, real_completions: Any, telemetry: CallTelemetry):
        self._real = real_completions
        self._telemetry = telemetry

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t = self._telemetry
        t.attempted = True
        t.requested_model = kwargs.get("model")
        start = time.monotonic()
        try:
            resp = self._real.create(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - must observe any exception type
            t.api_exception = str(e)
            t.api_exception_type = type(e).__name__
            raise
        finally:
            t.latency_seconds = (t.latency_seconds or 0.0) + (time.monotonic() - start)

        t.api_call_count += 1
        model = getattr(resp, "model", None)
        if model:
            t.returned_models.append(model)
        try:
            for choice in getattr(resp, "choices", []) or []:
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    t.raw_finish_reasons.append(fr)
        except Exception:  # noqa: BLE001 - telemetry must never break the real call
            pass
        return resp


class _RecordingChat:
    def __init__(self, real_chat: Any, telemetry: CallTelemetry):
        self.completions = _RecordingCompletions(real_chat.completions, telemetry)


class _RecordingOpenAIInstance:
    """Wraps one real OpenAI() client instance, recording into `telemetry`."""

    def __init__(self, real_client: Any, telemetry: CallTelemetry):
        self._real_client = real_client
        self.chat = _RecordingChat(real_client.chat, telemetry)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._real_client, item)


def _make_recording_openai_factory(real_openai_cls: Any, telemetry: CallTelemetry):
    def factory(*args: Any, **kwargs: Any) -> _RecordingOpenAIInstance:
        real_client = real_openai_cls(*args, **kwargs)
        return _RecordingOpenAIInstance(real_client, telemetry)

    return factory


@contextmanager
def _instrument_module_openai(module: Any, telemetry: CallTelemetry):
    """
    Temporarily replace `module.OpenAI` with a recording factory, then
    restore it. If `module.OpenAI` is None (openai package unavailable at
    import time in that module), this is a no-op and the module's own
    "openai package not available" fallback path fires exactly as in
    production.
    """
    original = getattr(module, "OpenAI", None)
    if original is None:
        yield
        return

    module.OpenAI = _make_recording_openai_factory(original, telemetry)
    try:
        yield
    finally:
        module.OpenAI = original


@dataclass
class RouterCallResult:
    prompt_id: str
    repeat_index: int
    capability_id: str
    reason: str
    fallback_detected: bool
    fallback_marker: Optional[str]
    telemetry: CallTelemetry
    harness_error: Optional[str] = None


@dataclass
class PlannerCallResult:
    prompt_id: str
    repeat_index: int
    plan: Dict[str, Any]
    likely_fallback_by_content_match: bool
    telemetry: CallTelemetry
    harness_error: Optional[str] = None


def _reload_module(mod_name: str):
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


def call_router(*, prompt_id: str, request_text: str, repeat_index: int, model: str = REQUESTED_MODEL) -> RouterCallResult:
    from src.agent import router_llm

    telemetry = CallTelemetry()
    harness_error: Optional[str] = None
    capability_id = ""
    reason = ""

    try:
        with _instrument_module_openai(router_llm, telemetry):
            obj = router_llm.llm_choose_capability(request=request_text, csv_columns=None, model=model)
        capability_id = str(obj.get("capability_id", ""))
        reason = str(obj.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        # Uncaught exception from llm_choose_capability itself (both internal
        # attempts failed with an exception it does not swallow). Distinct
        # from the function's own internal "reason"-marked fallback.
        harness_error = f"{type(e).__name__}: {e}"

    fallback_marker = None
    for marker in ROUTER_FALLBACK_REASON_MARKERS:
        if marker in reason:
            fallback_marker = marker
            break
    fallback_detected = fallback_marker is not None or harness_error is not None

    return RouterCallResult(
        prompt_id=prompt_id,
        repeat_index=repeat_index,
        capability_id=capability_id,
        reason=reason,
        fallback_detected=fallback_detected,
        fallback_marker=fallback_marker,
        telemetry=telemetry,
        harness_error=harness_error,
    )


def call_planner(*, prompt_id: str, request_text: str, repeat_index: int, model: str = REQUESTED_MODEL) -> PlannerCallResult:
    from src.agent import planner_llm

    telemetry = CallTelemetry()
    harness_error: Optional[str] = None
    plan: Dict[str, Any] = {}

    try:
        with _instrument_module_openai(planner_llm, telemetry):
            plan = planner_llm.llm_generate_analysis_plan(request=request_text, model=model)
    except Exception as e:  # noqa: BLE001
        harness_error = f"{type(e).__name__}: {e}"

    # llm_generate_analysis_plan swallows all internal exceptions and
    # returns _fallback_plan(request) with no marker distinguishing it from
    # a genuine model response (see AUDIT.md, Finding 4). Cross-check by
    # direct content comparison against the pure deterministic function.
    likely_fallback = False
    try:
        fb = planner_llm._fallback_plan(request_text)  # noqa: SLF001 - intentional, documented cross-check
        likely_fallback = plan == fb
    except Exception:  # noqa: BLE001
        pass

    return PlannerCallResult(
        prompt_id=prompt_id,
        repeat_index=repeat_index,
        plan=plan,
        likely_fallback_by_content_match=likely_fallback,
        telemetry=telemetry,
        harness_error=harness_error,
    )
