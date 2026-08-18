# Audit of the frozen semantic-orchestration implementation

Scope: `src/agent/router_llm.py`, `src/agent/planner_llm.py`,
`src/agent/capabilities/cap_*.json`, `src/agent/app.py`, `src/agent/graph.py`,
`src/agent/schemas.py`, `src/agent/schemas_io.py`, `src/agent/tools/*`.

Audited at commit `150f50ed8b555a896378f25b13d9cf40bdec6e1d` (branch
`paper_evaluation`, clean working tree at audit time). This audit is the
basis for the benchmark design in `benchmark/prompts_supported.py` /
`benchmark/prompts_challenge.py` and for the fallback-detection logic in
`harness/config.py` / `harness/model_client.py`. If the audited source
files change, this document and the harness config must be re-checked
together (see `tests/test_capability_registry_snapshot.py`, which pins
several of the findings below as executable assertions).

## 1. Registered capabilities (the router's full choice set)

The router (`router_llm.load_capabilities()`) and the FastAPI app
(`app.py::_load_capability_specs()`) both derive the allowed capability
set from `src/agent/capabilities/cap_*.json`, sorted alphabetically by
filename. There are exactly **7** registered capabilities:

| capability_id | title | required_fields | optional_fields |
|---|---|---|---|
| `binary_edrip` | Binary outcome causal effect (EDRIP) | csv, treatment, outcome | covariates |
| `causal_ate` | Average Treatment Effect (ATE) | csv, treatment, outcome | covariates |
| `hello_world` | (demo) | — | — |
| `linear_regression` | Linear regression | csv, outcome, covariates | — |
| `logistic_regression` | Logistic regression | csv, outcome, covariates | — |
| `summary_stats` | Dataset summary statistics (EDA) | csv | covariates |
| `survival_adjusted_curves` | Confounder-adjusted survival curves | csv, time, event, group | covariates |

Two additional tools exist in `src/agent/tools/` (`tool_dummy.py` ->
`dummy_capability`, `tool_dummy_echo.py` -> `dummy_echo`) but have **no**
corresponding `cap_*.json` file, so they are invisible to both the router
and the planner's guidance text. `tool_dummy_echo.py`'s `DummyEchoTool`
additionally does not subclass `BaseTool`, so it would not even
auto-register via `src/agent/tools/__init__.py`'s
`issubclass(obj, BaseTool)` check. Neither dummy tool is part of the
router's decision space; they are not treated as distractors in the
benchmark because a well-behaved router literally cannot select them.

**Benchmark decision:** the full 7-capability registry stays live as router
distractors for every supported and challenge prompt (per task
instructions). Only `causal_ate` and `survival_adjusted_curves` are ever a
correct gold label.

## 2. `causal_ate` / `binary_edrip` keyword overlap (the main ambiguity risk)

`cap_causal_ate.json` keywords: `ate, average treatment effect, average
causal effect, treatment effect, treatment vs control, causal effect, ipw,
propensity score, doubly robust, standard causal model`.

`cap_binary_edrip.json` keywords: `binary outcome, binary response, causal
effect, treatment effect, edrip, doubly robust, robust estimator, flexible
model, semiparametric, advanced causal method`.

**`doubly robust`, `causal effect`, and `treatment effect` are claimed by
both capabilities.** The descriptions resolve the overlap qualitatively,
not by keyword: `causal_ate` is "general-purpose... Applicable when the
goal is a single overall causal effect estimate"; `binary_edrip` is
"Specialized... for binary-outcome settings where the request emphasizes
robustness, flexibility, reduced parametric assumptions, or a specialized
estimator." The genuine ambiguity zone is therefore **a binary outcome
described with robustness/flexibility/semiparametric framing**, not the
presence of "doubly robust" alone.

**Benchmark decision:** no supported prompt (either capability) combines a
binary-outcome framing with any of `edrip`, `semiparametric`, `reduced
parametric assumptions`, `specialized estimator`, `flexible
model/estimator`, or `robust estimator`. `causal_ate` prompts freely use
"doubly robust" / "propensity score" / "IPW" language, but only paired
with continuous or plainly-unspecified outcomes (with the exception of the
`near_boundary` cell, discussed below). This rule is enforced
programmatically by `tests/test_benchmark_integrity.py::
test_no_causal_ate_prompt_uses_binary_edrip_overlap_language` against the
literal `BINARY_EDRIP_OVERLAP_TERMS` list in `prompts_supported.py`.

`causal_ate`'s `near_boundary` cell includes some binary-outcome prompts
(e.g. 30-day readmission yes/no) specifically to test that boundary, but
phrased with plain ATE language and no robustness/flexibility framing --
per the capability description's own criterion, a binary outcome alone
does not make `binary_edrip` correct; the specialized-estimator framing
does.

## 3. Two independent, non-identical fallback layers

There are **two** separate deterministic fallback mechanisms in the
routing path, and they disagree on the default capability:

**(a) `router_llm.llm_choose_capability`'s own internal fallback**
(`src/agent/router_llm.py`, lines ~71-154): if `OPENAI_API_KEY` is unset,
the `openai` package is unavailable, the model returns non-JSON, the JSON
is invalid, or the model names a `capability_id` not in the allowed list,
the function returns `{"capability_id": fallback_id, "reason": "<marker
string>"}` where `fallback_id = allowed[0]` -- **the alphabetically-first
`cap_*.json` file, i.e. `binary_edrip`**. This path is reached for the
overwhelming majority of real-world failure modes (missing key, malformed
output, hallucinated capability id) and never raises.

**(b) `graph.py`'s separate rule-based fallback** (`_choose_capability` /
`_router_fallback` in `src/agent/graph.py`): defaults to
`survival_adjusted_curves` if `time`, `event`, and `group` request fields
are all present, else **`causal_ate`**. This path is reached only if
`llm_choose_capability` (a) raises an uncaught exception -- which requires
*both* of its internal try/except attempts to fail, since the first
attempt's exception is itself caught and retried without
`response_format` -- (b) returns a non-dict, or (c) returns an empty
`capability_id` string; or if `use_llm_router=False`; or if the request
text is empty/whitespace. In practice, because (a)'s own internal fallback
absorbs nearly every failure mode into a valid non-empty dict, path (b) is
close to unreachable through the LLM-router branch in normal operation.
`app.py::select_capability` has a third, similar rule-based fallback
(defaulting to `causal_ate` if not `task="survival"`), reached under the
same narrow conditions.

**Consequence for the benchmark and analysis:** "the deterministic
fallback" is not a single well-defined baseline. The benchmark's
deterministic baseline (`harness/run_deterministic_baseline.py`) evaluates
path (a) only -- calling the real `router_llm.llm_choose_capability` and
`planner_llm.llm_generate_analysis_plan` functions directly, the same
components the live GPT-5.4 evaluation targets -- and does not exercise
`graph.py`'s separate rule-based layer, since that layer is normally
unreachable given (a)'s absorbing behavior. This is recorded here
descriptively per the task instructions rather than "fixed," since
`graph.py` is frozen application code, out of scope to modify.

`tests/test_capability_registry_snapshot.py::
test_router_alphabetical_fallback_id_is_binary_edrip` pins fallback_id to
`binary_edrip`, and pins the literal fallback-reason marker strings
(`harness/config.py::ROUTER_FALLBACK_REASON_MARKERS`) against the actual
`router_llm.py` source text, so drift in either is caught by the static
test suite rather than silently invalidating the deterministic-baseline
numbers below.

## 4. Planner's fallback has no observable marker

`planner_llm.llm_generate_analysis_plan` wraps its entire OpenAI call in a
single broad `try/except Exception: return _fallback_plan(request)`. Unlike
the router, **the returned plan dict carries no field indicating whether
it came from GPT-5.4 or from the pure rule-based `_fallback_plan()`**
(`tests/test_capability_registry_snapshot.py::
test_planner_fallback_plan_has_no_explicit_fallback_marker` pins the exact
key set). Black-box detection by inspecting only the returned plan is
therefore impossible.

**Harness decision:** `harness/model_client.py::call_planner` does not
rely on the returned plan's content to detect fallback. Instead it
temporarily replaces the `OpenAI` class reference inside the
`planner_llm` module's own namespace with a recording proxy (never editing
`planner_llm.py` on disk) so that whether an API call was attempted, its
exception (if any), and the actual `response.model` value are all
captured directly -- giving explicit, ground-truth success/fallback/
model-substitution telemetry independent of plan content. As a secondary,
descriptive cross-check only, `call_planner` also compares the returned
plan against a direct call to `_fallback_plan(request)` and flags
`likely_fallback_by_content_match` when they are byte-identical.

## 5. `binary_edrip` registry/tool mismatch

`cap_binary_edrip.json` lists `covariates` under `optional_fields`, but
`BinaryEDRIPTool.validate()` (`src/agent/tools/tool_binary_edrip.py`)
rejects any request with an empty or missing `covariates` list ("
`binary_edrip requires covariates`"). This is a pre-existing inconsistency
between the capability registry (what the router/planner see) and the
tool's actual validation (what would happen if the tool were invoked). It
does not affect the router/planner evaluation (neither component executes
the tool), so it is recorded here and pinned by
`tests/test_capability_registry_snapshot.py::
test_binary_edrip_capability_json_lists_covariates_optional_but_tool_requires_it`
rather than fixed, since `tool_binary_edrip.py` and
`cap_binary_edrip.json` are both frozen application files out of scope for
this evaluation task.

## 6. Two parallel `RunRequest` schemas; environment/dependency drift

`src/agent/schemas.py` defines a Pydantic `RunRequest` used by
`app.py` (FastAPI layer); `src/agent/schemas_io.py` defines a separate
dataclass `RunRequest` used by `graph.py` and all `tool_*.py` files. They
overlap in most fields but are not the same type. This is pre-existing
architecture, not touched by this evaluation (the router/planner
evaluation targets `router_llm.py` / `planner_llm.py` directly and does
not go through either `RunRequest` type).

Separately, this evaluation environment's installed package versions
differ from `requirements.txt`'s pins (notably `openai` 3.3.0 installed vs.
1.52.2 pinned; see `provenance/ENVIRONMENT.md` for the full diff, captured
at freeze time). `openai>=1.0`'s `client.chat.completions.create(...)`
interface, which both `router_llm.py` and `planner_llm.py` use, is stable
across this range, so this is not expected to change router/planner
behavior, but is recorded for reproducibility.

## 7. Model routed to GPT-5.4

Both `router_llm.llm_choose_capability(..., model=...)` and
`planner_llm.llm_generate_analysis_plan(..., model=...)` accept an
explicit `model` override, which the harness always passes as `"gpt-5.4"`
(`harness/config.py::REQUESTED_MODEL`), matching the task's "requested
model gpt-5.4, no model substitution" requirement.
`router_llm.py`'s own internal default (used only if `model=None` is
passed by a caller) is also `"gpt-5.4"`; `planner_llm.py`'s own internal
default is `"gpt-4o-mini"` -- irrelevant here since the harness always
passes `model="gpt-5.4"` explicitly, but worth noting so a future reader
does not assume the planner's production default matches what this
evaluation measures.

## Summary of benchmark design decisions driven by this audit

1. Only `causal_ate` and `survival_adjusted_curves` are ever a gold label;
   all 7 capabilities remain live router options.
2. No supported prompt combines binary-outcome framing with
   robustness/flexibility/semiparametric/EDRIP language (Finding 2).
3. The deterministic baseline evaluates `router_llm.llm_choose_capability`
   and `planner_llm.llm_generate_analysis_plan` directly (Finding 3), not
   `graph.py`'s separate, largely-unreachable rule-based layer.
4. Fallback detection for the router uses reason-string markers pinned
   against the actual source (Finding 3); fallback detection for the
   planner uses call-level API telemetry via non-invasive monkeypatching,
   not plan content (Finding 4).
5. Challenge prompts (`underspecified`, `unsupported_method`,
   `out_of_scope_mixed`) are deliberately given no single
   `gold_capability_id`, since forcing one would itself be an ambiguous
   gold label; `unsupported_method` prompts are checked not to
   accidentally describe `binary_edrip`, which *is* implemented.
