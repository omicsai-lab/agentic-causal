# Manuscript facts sheet

Citable, source-linked facts for the JAMIA Open manuscript's semantic-
orchestration evaluation section. Every number here is either (a) computed
by a script in this directory and reproducible by re-running it, or (b)
explicitly marked `[PENDING STEP 2]`. Do not hand-write a number into the
manuscript that is not traceable to a line in this file.

## Benchmark construction (final, citable now)

- Benchmark covers 2 of 7 registered agent capabilities
  (`causal_ate`, `survival_adjusted_curves`); all 7 remain active as
  router distractors during evaluation. Source: `AUDIT.md` Section 1,
  `BENCHMARK_MANIFEST.json`.
- 120 supported prompts (60 per capability), stratified across 6
  linguistic/framing categories (explicit_method, formal_estimand,
  biomedical_domain, indirect_colloquial, noisy_verbose, near_boundary),
  10 prompts per capability x category cell (12 cells). Source:
  `BENCHMARK_MANIFEST.json::counts`.
- 30 out-of-distribution challenge prompts (10 underspecified, 10
  unsupported-method, 10 out-of-scope/mixed-intent), each without a
  single forced gold capability label by design. Source:
  `BENCHMARK_MANIFEST.json::counts`, `EVALUATION_PROTOCOL.md` "Challenge
  set scoring".
- Benchmark frozen (SHA-256 manifest) before any live model call; version
  `1.0.0`, git commit `150f50ed8b555a896378f25b13d9cf40bdec6e1d`. Source:
  `BENCHMARK_MANIFEST.json`.
- Zero near-duplicate prompt pairs (token-Jaccard > 0.70) across all 8,925
  supported-prompt pairs and 435 challenge-prompt pairs. Source:
  `BENCHMARK_REVIEW.md`, `tests/test_benchmark_integrity.py`.

## Deterministic (non-LLM) baseline (final, citable now)

- The production router's own internal deterministic fallback path
  (triggered whenever no usable API key/response is available) always
  selects the alphabetically-first registered capability (`binary_edrip`),
  yielding **0.0% accuracy (0/120)** against the supported-prompt gold
  labels -- i.e., the deterministic fallback is not a competitive router
  baseline by construction, since it never selects either in-scope
  capability. Source: `results/processed/deterministic_baseline.json`,
  `EVALUATION_REPORT.md`.
- The production planner's own internal deterministic fallback
  (`_fallback_plan()`, a pure keyword-matching rule set) achieves **74.2%
  accuracy (89/120)** on `recommended_tool` against gold, with a sharp
  category split: 100% on prompts that name the method or estimand
  explicitly, 0-20% on colloquially-phrased `causal_ate` prompts that
  avoid the literal trigger words ("treatment effect", "causal effect",
  "ATE"). Source: `results/processed/deterministic_baseline.json`,
  `EVALUATION_REPORT.md`.

## Evaluation methodology (final, citable now)

- Model: GPT-5.4, temperature 0, fixed and never substituted; requested
  model id verified against each raw API response's `response.model`
  field (ground truth, not inferred). Source: `EVALUATION_PROTOCOL.md`
  "Fixed run parameters", `harness/model_client.py`.
- Supported prompts evaluated with 3 independent repeats per prompt per
  component (router, planner); challenge prompts with 1 repeat. Total
  780 API calls planned for Step 2. Source: `EVALUATION_PROTOCOL.md`.
- Statistics: Wilson score 95% CIs for all proportions (chosen over the
  normal approximation given n=10-30 cell sizes); exact (binomial-based)
  McNemar's test for paired planner-vs-router comparison; percentile
  paired bootstrap 95% CI (seed `20260818`, 10,000 resamples) on the
  accuracy difference. Source: `analysis/stats.py`,
  `EVALUATION_PROTOCOL.md`.

## Results `[PENDING STEP 2]`

The following are not yet available and must not be cited until Step 2's
live evaluation has run and `analysis/run_all.py` has produced
`results/processed/analysis_summary.json`:

- Router accuracy (overall / per-capability / per-category / per-repeat)
- Planner structured-field accuracy
- Three-run consistency and pairwise agreement
- Majority-vote accuracy
- Planner-router agreement
- Exact McNemar p-value and paired bootstrap CI for the planner-vs-router
  accuracy difference
- Confusion matrices
- Challenge-set descriptive results
