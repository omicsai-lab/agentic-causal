# Evaluation protocol: semantic orchestration benchmark

This document specifies exactly how the frozen benchmark
(`BENCHMARK_MANIFEST.json`) is to be evaluated. It is written before any
live GPT-5.4 call is made (Step 1). Step 2 must follow this protocol
without amendment; any change to the protocol after Step 2 begins
invalidates comparability between the two runs and must be recorded as a
new, dated addendum rather than a silent edit.

## Fixed run parameters

| Parameter | Value | Where enforced |
|---|---|---|
| Requested model | `gpt-5.4` | `harness/config.py::REQUESTED_MODEL`, passed explicitly to every call |
| Temperature | `0` | Set inside `router_llm.llm_choose_capability` / `planner_llm.llm_generate_analysis_plan` (frozen application code; harness does not override) |
| Model substitution | Never. If the requested model errors, the error is recorded verbatim; the harness never retries with a different model id. | `harness/model_client.py` -- no retry/fallback-to-different-model logic anywhere in the harness |
| Supported-prompt repeats | 3 per prompt, per component (router, planner) | `harness/config.py::SUPPORTED_REPEATS` |
| Challenge-prompt repeats | 1 per prompt, per component | `harness/config.py::CHALLENGE_REPEATS` |
| Total live API calls (Step 2) | 120×3 (router) + 120×3 (planner) + 30×1 (router) + 30×1 (planner) = **780** | -- |
| Bootstrap seed | `20260818` | `analysis/stats.py::BOOTSTRAP_SEED`, `harness/config.py::BOOTSTRAP_SEED` |

## Components under evaluation

1. **Router**: `src/agent/router_llm.py::llm_choose_capability`. Scored
   against `gold_capability_id` for supported prompts. Not scored
   (distribution reported only) for challenge prompts.
2. **Planner**: `src/agent/planner_llm.py::llm_generate_analysis_plan`.
   Scored on structured-field accuracy: `recommended_tool` (must equal
   `gold_capability_id`), `task_type` (`causal_ate` -> `"causal_effect"`,
   `survival_adjusted_curves` -> `"survival"`), and `outcome_type` (must
   equal the prompt's `gold_outcome_type`).
3. **Deterministic baseline**: the same two functions, called with
   `OPENAI_API_KEY` removed from the process environment for the duration
   of the call (restored immediately after), which deterministically
   triggers each function's own internal fallback path (see AUDIT.md,
   Findings 3-4). No network calls. Run in Step 1
   (`harness/run_deterministic_baseline.py`) and re-run unchanged in Step
   2 as a sanity check that the frozen fallback behavior has not drifted.

## API-success / fallback / model-substitution detection

Implemented in `harness/model_client.py` (see AUDIT.md Findings 3-4 for
the design rationale):

- **Router**: `fallback_detected = True` if the returned `reason` string
  contains one of `harness/config.py::ROUTER_FALLBACK_REASON_MARKERS`
  (pinned against the live `router_llm.py` source by
  `tests/test_capability_registry_snapshot.py`), OR if calling
  `llm_choose_capability` raised an uncaught exception
  (`harness_error` is set in that case).
- **Planner**: success/failure is read from call-level telemetry
  (`CallTelemetry.api_call_succeeded`, `api_exception_type`,
  `api_exception`) captured by a non-invasive monkeypatch of the
  `OpenAI` class reference inside the `planner_llm` module's namespace,
  restored immediately after each call. `likely_fallback_by_content_match`
  is a secondary, descriptive-only cross-check against
  `planner_llm._fallback_plan(request)`.
- **Model substitution**: `CallTelemetry.model_substituted` is `True` if
  any captured `response.model` differs from the requested
  `"gpt-5.4"`. This is ground truth from the raw API response object, not
  inferred.

A call counts as a genuine model success only if `api_call_succeeded` is
`True` AND `fallback_detected` (router) / `likely_fallback_by_content_match`
(planner, descriptive) is `False` AND `model_substituted` is `False`.

## Supported-set scoring

- **Router accuracy**: exact match of `predicted_capability_id` against
  `gold_capability_id`, per repeat, pooled across repeats, and per
  (capability, category) cell (12 cells, n=10 per repeat, n=30 pooled
  across 3 repeats). Wilson score 95% CIs throughout
  (`analysis/stats.py::wilson_ci`) -- chosen over the normal approximation
  because several cells have n=10 or n=30, where the normal approximation
  is unreliable near 0%/100%.
- **Planner structured-field accuracy**: per-field (recommended_tool,
  task_type, outcome_type) and all-fields-correct, same CI treatment.
- **Three-run consistency**: for prompts with all 3 repeats present,
  pairwise agreement across the 3 repeat-pairs (`analysis/stats.py::
  pairwise_agreement`) and the fraction of prompts where all 3 repeats
  agree exactly ("full consistency").
- **Majority voting**: for each prompt, the majority label across 3
  repeats (ties broken deterministically by first-occurrence order,
  flagged as non-clean) is scored against gold, separately from
  per-repeat accuracy, to show whether repeat-and-vote improves over a
  single call.
- **Planner-router agreement**: for matched (prompt_id, repeat_index)
  pairs, whether the planner's `recommended_tool` equals the router's
  `predicted_capability_id`. This is agreement between the two
  components, not correctness against gold.
- **Planner vs. router comparison**: exact (binomial-based) McNemar's test
  on matched-item correctness (`analysis/stats.py::exact_mcnemar`, exact
  for any discordant-pair count rather than the chi-square
  approximation), plus a paired bootstrap 95% CI (seed `20260818`,
  10,000 resamples) on the accuracy difference.
- **Confusion matrices**: gold capability_id x predicted capability_id
  across the full 7-capability registry (not just the 2 supported ones),
  since distractor capabilities remain live predictions.

## Challenge-set scoring

By design (see AUDIT.md / `prompts_challenge.py` docstring), no challenge
prompt has a single correct `gold_capability_id`:

- `underspecified`: more than one registered capability is a defensible
  pick given the missing information.
- `unsupported_method`: no registered capability correctly implements the
  requested method.
- `out_of_scope_mixed`: the request is unrelated to any registered
  capability, or mixes an out-of-scope ask with an in-scope one.

Challenge-set results are therefore reported **descriptively**: the
distribution of predicted `capability_id` (router) and `recommended_tool`
(planner) per `challenge_type`, and the fallback/API-success rate per
`challenge_type`. No accuracy percentage is computed or claimed for the
challenge set. `EVALUATION_REPORT.md` must not present a single
"challenge accuracy" number; if a future analysis wants one, it must
first define, in writing, what "correct" means for a forced 7-way choice
against a prompt with no correct answer -- this protocol deliberately does
not do that.

## Benchmark immutability

Once `BENCHMARK_MANIFEST.json` is written (Step 1), the SHA-256 hashes of
`benchmark/prompts_supported.jsonl` and `benchmark/prompts_challenge.jsonl`
are fixed. No prompt text, gold label, or category assignment may change
after any model output has been observed. If a defect is later found in a
prompt or gold label:

1. Do not edit the prompt in place.
2. Record the defect and its resolution in `BENCHMARK_REVIEW.md`.
3. If a fix is required, produce a new frozen benchmark version (new
   manifest, new hash, explicit version note) rather than mutating the
   version already used for any live evaluation, so prior results remain
   attributable to the exact benchmark version that produced them.

## What Step 1 does and does not run

Step 1 runs: `benchmark/build_benchmark.py` (JSONL generation),
`harness/run_deterministic_baseline.py` (no network calls),
`harness/preflight.py` (at most one API call, to check model
reachability only), the full static test suite, and
`provenance/capture_environment.py`.

Step 1 does **not** run `harness/run_router_eval.py`,
`harness/run_planner_eval.py`, or `harness/run_challenge_eval.py` against
the full benchmark -- those perform the 780 live GPT-5.4 calls and are
reserved for Step 2. `analysis/run_all.py` is a no-op until the raw
result files those scripts produce exist.
