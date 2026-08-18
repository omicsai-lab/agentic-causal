# Manuscript facts sheet

**Status: FINAL.** Citable, source-linked facts for the JAMIA Open
manuscript's semantic-orchestration evaluation section. Every number here
is computed by a script in this directory from `results/raw/*.jsonl`
(untouched live model outputs) and is reproducible by re-running
`analysis/run_all.py`. Do not hand-write a number into the manuscript
that is not traceable to a line in this file or
`results/processed/analysis_summary.json`.

## Benchmark construction

- Benchmark covers 2 of 7 registered agent capabilities (`causal_ate`,
  `survival_adjusted_curves`); all 7 remain active as router distractors
  during evaluation. Source: `AUDIT.md` Section 1, `BENCHMARK_MANIFEST.json`.
- 120 supported prompts (60 per capability), 10 per (capability × category)
  across 6 categories. 30 out-of-distribution challenge prompts (10 each:
  underspecified, unsupported-method, out-of-scope/mixed-intent), none
  with a single forced gold capability label by design. Source:
  `BENCHMARK_MANIFEST.json::counts`.
- Benchmark frozen (SHA-256 manifest) before any live model call; version
  `1.0.0`, git commit `150f50ed8b555a896378f25b13d9cf40bdec6e1d`. Zero
  near-duplicate prompt pairs (token-Jaccard > 0.70). Source:
  `BENCHMARK_MANIFEST.json`, `BENCHMARK_REVIEW.md`.

## Live evaluation execution

- 780 total GPT-5.4 API calls (120×3 router-supported, 120×3
  planner-supported, 30×1 router-challenge, 30×1 planner-challenge), all
  temperature 0, requested model `gpt-5.4` fixed with no substitution.
  **0 API failures, 0 fallback events, 0 model substitutions, 0 retries
  across all 780 calls.** Every single call returned the identical model
  snapshot **`gpt-5.4-2026-03-05`**. Source:
  `provenance/live_evaluation_summary.json`,
  `results/processed/analysis_summary.json::live_run_summaries`.

## PRIMARY comparison: GPT-5.4 planner (majority vote) vs. deterministic baseline

The manuscript's central planner result. Prompt-level majority vote of
the GPT-5.4 planner's `recommended_tool` across 3 repeats vs. the
repository's existing deterministic, zero-network-call `_fallback_plan()`
baseline, on the identical 120 supported prompts (exact McNemar,
binomial-based; paired bootstrap 95% CI, seed 20260818, 10,000 resamples).

- **GPT-5.4 planner majority-vote accuracy: 98.3% (118/120), 95% Wilson
  CI [94.1%, 99.5%].**
- **Deterministic `_fallback_plan()` baseline accuracy: 74.2% (89/120),
  95% Wilson CI [65.7%, 81.2%].**
- **Absolute difference: +24.2 percentage points.**
- **Exact McNemar: b=29, c=0, p=3.7×10⁻⁹** (the deterministic baseline
  never corrected a GPT-5.4 planner error, in either direction, anywhere
  in the benchmark).
- **Paired bootstrap 95% CI on the accuracy difference: [+16.7pp, +31.7pp]**
  (seed 20260818).
- By gold capability: causal_ate +33.3pp (96.7% vs 63.3%, p=2.0×10⁻⁶);
  survival_adjusted_curves +15.0pp (100.0% vs 85.0%, p=0.0039).
- By linguistic category: the deterministic baseline is statistically
  indistinguishable from GPT-5.4 only on `explicit_method` (both 100%,
  p=1.0), `noisy_verbose` (both 100%, p=1.0), and `near_boundary` (100%
  vs 85%, p=0.25) -- the categories where the baseline's literal keyword
  triggers appear in the prompt text by construction. The largest gaps are
  `indirect_colloquial` (+70.0pp, 95% vs 25%, p=0.00012) and
  `biomedical_domain` (+55.0pp, 95% vs 40%, p=0.00098) -- categories
  deliberately written to avoid those literal trigger words.
- Source: `results/processed/analysis_summary.json::
  planner_vs_deterministic_baseline`, `tables/planner_vs_deterministic_
  baseline.{csv,md,tex}`, `figures/planner_vs_deterministic_baseline.
  {png,pdf,csv}`. NOT the same comparison as planner-vs-router agreement
  (see below) -- see `analysis/baseline_comparison.py` docstring.

## Router results

- Run-level accuracy (n=360, 3 repeats × 120 prompts): **98.3% (354/360),
  95% Wilson CI [96.4%, 99.2%].** Fallback rate 0.0% (0/360).
- Prompt-level majority-vote accuracy (n=120): **98.3% (118/120), 95%
  Wilson CI [94.1%, 99.5%]** -- unchanged from run-level because all 120
  prompts had a unanimous (3/3) vote.
- ATE (`causal_ate`) accuracy: 96.7% (174/180). Survival
  (`survival_adjusted_curves`) accuracy: **100.0% (180/180).**
- By category: explicit_method 100.0%, formal_estimand 100.0%,
  biomedical_domain 95.0%, indirect_colloquial 95.0%, noisy_verbose
  100.0%, near_boundary 100.0%.
- 3-run consistency: **100% full (3/3) agreement across all 120 prompts**,
  overall, by capability, and by every category (mean pairwise agreement
  = 1.0 in every stratum).
- All 6 run-level router errors (2 prompts × 3 repeats, both unanimous)
  are `causal_ate → linear_regression` misroutes; zero confusion with
  `binary_edrip` (the capability the benchmark specifically targets for
  disambiguation) anywhere in the 360 supported-prompt calls.
- Source: `results/processed/analysis_summary.json::router_accuracy`,
  `::router_accuracy_majority_vote`, `::router_consistency_stratified`,
  `::router_confusion_pairs`.

## Planner structured-field results

- Run-level (n=360): recommended_tool 98.3% [96.4%, 99.2%], task_type
  98.3% [96.4%, 99.2%], outcome_type 99.2% [97.6%, 99.7%],
  all-three-fields-correct 97.5% [95.3%, 98.7%].
- Prompt-level majority vote on recommended_tool (n=120): 98.3%
  [94.1%, 99.5%] (same accuracy as router majority vote, but a different,
  only partially overlapping, pair of missed prompts -- see
  `EVALUATION_REPORT.md`).
- 100% full (3/3) consistency across all 120 prompts, overall, by
  capability, and by category.
- 0/360 planner calls showed the signature of a silent fallback
  (`likely_fallback_by_content_match`).
- Source: `results/processed/analysis_summary.json::planner_accuracy`,
  `::planner_accuracy_majority_vote`, `::planner_consistency_stratified`.

## Planner-router agreement (separate from the PRIMARY comparison above)

- Matched-call agreement (n=360): 98.3% (354/360) [96.4%, 99.2%].
- Exact McNemar comparing router vs. planner correctness on the same
  matched calls: b=3, c=3, **p=1.0** (no detectable accuracy difference
  between the two components). Paired bootstrap mean diff 0.0 [-1.4pp,
  +1.4pp] (seed 20260818).
- Source: `results/processed/analysis_summary.json::
  planner_router_agreement`, `::planner_vs_router_mcnemar`.

## Challenge set (descriptive only -- explicitly NOT an accuracy or abstention/safety benchmark)

No accuracy, pass-rate, or abstention-rate figure exists for the
challenge set anywhere in this evaluation, by design (see
`EVALUATION_PROTOCOL.md`, "Challenge set scoring"). Observed
descriptively: 0 API failures / fallback events / model substitutions
across all 60 challenge calls. For `unsupported_method` prompts (10
causal-inference methods not implemented by any registered capability),
both router (8/10) and planner (10/10) default overwhelmingly to
`causal_ate` rather than spreading across the registry. For
`out_of_scope_mixed` prompts, the two components diverge: router's modal
pick is `hello_world` (7/10), planner's is `summary_stats` (8/10). Full
distributions and per-prompt detail: `tables/challenge_prediction_
distribution.{csv,md,tex}`, `tables/challenge_router_detail.{csv,md,tex}`,
`tables/challenge_planner_detail.{csv,md,tex}`.

## Methodology

- Model: GPT-5.4, temperature 0, fixed and never substituted; verified
  against each raw API response's `response.model` field. Supported
  prompts: 3 independent repeats per prompt per component. Challenge
  prompts: 1 repeat. Source: `EVALUATION_PROTOCOL.md`.
- Statistics: Wilson score 95% CIs for all proportions; exact
  (binomial-based) McNemar's test for paired comparisons; percentile
  paired bootstrap 95% CI (seed `20260818`, 10,000 resamples). Source:
  `analysis/stats.py` (unmodified from Step 1 -- no statistical formula
  was altered during finalization).
