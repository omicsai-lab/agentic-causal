# Evaluation report: semantic orchestration benchmark

**Status: Step 1 complete (benchmark frozen + deterministic baseline).
Step 2 (780 live GPT-5.4 calls) has not been run.** Sections marked
`[PENDING STEP 2]` will be filled in by re-running
`analysis/run_all.py` after `harness/run_router_eval.py`,
`harness/run_planner_eval.py`, and `harness/run_challenge_eval.py` have
produced `results/raw/*.jsonl`. This report must not be hand-edited with
invented numbers; it is regenerated from `results/processed/*.json`.

See `AUDIT.md` for the frozen-implementation audit, `EVALUATION_PROTOCOL.md`
for the full scoring methodology, `BENCHMARK_REVIEW.md` for the benchmark
self-review, and `BENCHMARK_MANIFEST.json` for the frozen benchmark's
hashes and counts.

## Benchmark summary

- 120 supported prompts: 60 `causal_ate`, 60 `survival_adjusted_curves`,
  10 per (capability x category) across 6 categories (`explicit_method`,
  `formal_estimand`, `biomedical_domain`, `indirect_colloquial`,
  `noisy_verbose`, `near_boundary`).
- 30 challenge prompts: 10 `underspecified`, 10 `unsupported_method`, 10
  `out_of_scope_mixed`. No challenge prompt has a single correct gold
  capability by design (see `EVALUATION_PROTOCOL.md`, "Challenge set
  scoring").
- Frozen at commit `150f50ed8b555a896378f25b13d9cf40bdec6e1d`, benchmark
  version `1.0.0` (`BENCHMARK_MANIFEST.json`).

## Deterministic baseline (Step 1 result -- final, not pending)

Computed by `harness/run_deterministic_baseline.py`, which calls the real
`router_llm.llm_choose_capability` / `planner_llm.llm_generate_analysis_plan`
functions with `OPENAI_API_KEY` removed from the environment, deterministically
triggering each function's own internal fallback path (AUDIT.md, Findings
3-4). No network calls were made. Full per-prompt output:
`results/processed/deterministic_baseline.json`.

| Component | Baseline behavior | Accuracy on 120 supported prompts |
|---|---|---|
| Router (`llm_choose_capability` internal fallback) | Always predicts `binary_edrip` (alphabetically-first registered capability; AUDIT.md Finding 3) | **0 / 120 = 0.0%** |
| Planner (`_fallback_plan`, pure keyword rules) | Rule-based keyword matching on `recommended_tool` | **89 / 120 = 74.2%** |

Planner deterministic-baseline accuracy by (capability, category) cell:

| Capability | Category | Accuracy |
|---|---|---|
| causal_ate | explicit_method | 10/10 |
| causal_ate | formal_estimand | 9/10 |
| causal_ate | biomedical_domain | 2/10 |
| causal_ate | indirect_colloquial | 0/10 |
| causal_ate | noisy_verbose | 10/10 |
| causal_ate | near_boundary | 7/10 |
| survival_adjusted_curves | explicit_method | 10/10 |
| survival_adjusted_curves | formal_estimand | 10/10 |
| survival_adjusted_curves | biomedical_domain | 6/10 |
| survival_adjusted_curves | indirect_colloquial | 5/10 |
| survival_adjusted_curves | noisy_verbose | 10/10 |
| survival_adjusted_curves | near_boundary | 10/10 |

Interpretation: `_fallback_plan()`'s keyword rules
(`"survival"|"kaplan"|"time-to-event"|"time to event"|"hazard"` ->
survival; `"causal effect"|"treatment effect"|"average treatment
effect"|"ate"` -> causal_ate) fire reliably whenever a prompt contains
those literal substrings (`explicit_method`, `formal_estimand` mostly do;
`noisy_verbose` prompts were deliberately written to still contain the
trigger phrase somewhere in the rambling text), and fail almost entirely
on `causal_ate`'s `indirect_colloquial` cell, which was deliberately
written to avoid all of those literal phrases. This establishes the floor
the GPT-5.4 router/planner are expected to clear in Step 2, and gives a
concrete, data-backed illustration of why a keyword-only baseline is
insufficient for colloquial phrasing.

## Router accuracy `[PENDING STEP 2]`

## Planner structured-field accuracy `[PENDING STEP 2]`

## Three-run consistency `[PENDING STEP 2]`

## Pairwise agreement `[PENDING STEP 2]`

## Majority voting `[PENDING STEP 2]`

## Planner-router agreement `[PENDING STEP 2]`

## Exact McNemar (planner vs. router) `[PENDING STEP 2]`

## Paired bootstrap CI (seed 20260818) `[PENDING STEP 2]`

## Confusion matrices `[PENDING STEP 2]`

## Challenge-set results (descriptive only) `[PENDING STEP 2]`

## Static test suite (Step 1 result -- final)

`python3 -m pytest evaluation/semantic_orchestration/tests/` — **45
passed**, 0 failed. All 32 `.py` files under
`evaluation/semantic_orchestration/` compile cleanly under
`python3 -m py_compile`.

## Preflight `[PENDING STEP 2 -- SKIPPED IN STEP 1]`

`harness/preflight.py` was run once in Step 1 and reported `SKIPPED`:
no `OPENAI_API_KEY` is configured in this environment, so zero API calls
were made. Preflight must be re-run and must report `OK` before Step 2's
780-call evaluation begins.
