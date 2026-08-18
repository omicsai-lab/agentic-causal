"""
Fixed evaluation configuration. Nothing in this module may vary across a
run of the benchmark -- see EVALUATION_PROTOCOL.md, "Fixed run
parameters".
"""

from __future__ import annotations

REQUESTED_MODEL = "gpt-5.4"
TEMPERATURE = 0

SUPPORTED_REPEATS = 3
CHALLENGE_REPEATS = 1

BOOTSTRAP_SEED = 20260818

# Router fallback markers: literal prefixes emitted by
# src/agent/router_llm.py::llm_choose_capability's `reason` field when it
# falls back internally rather than acting on a genuine model response.
# See AUDIT.md, Finding 3. If router_llm.py's wording ever changes,
# tests/test_capability_registry_snapshot.py will fail and this list must
# be updated together with the audit.
ROUTER_FALLBACK_REASON_MARKERS = [
    "OPENAI_API_KEY not set; defaulting to first capability.",
    "openai package not available; defaulting to first capability.",
    "LLM returned non-JSON; defaulting to first capability.",
    "LLM returned invalid JSON; defaulting to first capability.",
    "LLM chose invalid capability_id=",  # prefix; full message includes the bad id
]

# The capability_id router_llm.py falls back to: alphabetically-first
# cap_*.json by filename glob, which is "binary_edrip".
ROUTER_FALLBACK_CAPABILITY_ID = "binary_edrip"

# graph.py's separate rule-based fallback (reached only if
# llm_choose_capability raises, returns a non-dict, or returns an empty
# capability_id -- see AUDIT.md, Finding 3). Not exercised by the harness
# directly (the harness evaluates router_llm.llm_choose_capability and
# planner_llm.llm_generate_analysis_plan in isolation, matching how the
# paper evaluates "the planner" and "the router" as components), but
# recorded here for completeness / cross-reference from AUDIT.md.
GRAPH_FALLBACK_DEFAULT_NO_TIME_FIELDS = "causal_ate"
GRAPH_FALLBACK_DEFAULT_WITH_TIME_FIELDS = "survival_adjusted_curves"

RESULTS_RAW_DIR = "results/raw"
RESULTS_PROCESSED_DIR = "results/processed"
