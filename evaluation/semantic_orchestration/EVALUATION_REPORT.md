# Evaluation report: semantic orchestration benchmark

**Status: FINAL.** The live GPT-5.4 evaluation (780 calls: 120×3 router +
120×3 planner supported, 30×1 router + 30×1 planner challenge) has
completed successfully with zero API failures, zero fallback events, zero
model substitutions, and zero retries. This report is generated from
`results/processed/analysis_summary.json`, itself produced by
`analysis/run_all.py` reading only `results/raw/*.jsonl` (untouched, raw
model outputs) and `results/processed/deterministic_baseline.json`
(zero-network-call deterministic baseline). No number below was hand-typed
independent of those files.

See `AUDIT.md` for the frozen-implementation audit, `EVALUATION_PROTOCOL.md`
for the scoring methodology, `BENCHMARK_REVIEW.md` for the benchmark
self-review, `BENCHMARK_MANIFEST.json` for the frozen benchmark's hashes,
and `MANUSCRIPT_FACTS.md` for the citable facts sheet.

## Benchmark summary

- 120 supported prompts: 60 `causal_ate`, 60 `survival_adjusted_curves`,
  10 per (capability × category) across 6 categories.
- 30 challenge prompts: 10 `underspecified`, 10 `unsupported_method`, 10
  `out_of_scope_mixed`. **The challenge set is descriptive only and is
  not an abstention/safety benchmark** -- no challenge prompt has a
  single correct gold capability by design, so no accuracy figure is
  computed or reported for it anywhere in this document (see
  `EVALUATION_PROTOCOL.md`, "Challenge set scoring").
- Frozen at commit `150f50ed8b555a896378f25b13d9cf40bdec6e1d`, benchmark
  version `1.0.0`.

## Live evaluation call outcomes

| Component | Calls | API success | API failure | Fallback / likely-fallback | Model substituted | Duplicate keys (retries) |
|---|---|---|---|---|---|---|
| Router, supported | 360 | 360 | 0 | 0 | 0 | 0 |
| Planner, supported | 360 | 360 | 0 | 0 | 0 | 0 |
| Router, challenge | 30 | 30 | 0 | 0 | 0 | -- |
| Planner, challenge | 30 | 30 | 0 | 0 | 0 | -- |
| **Total** | **780** | **780** | **0** | **0** | **0** | **0** |

Requested model: `gpt-5.4` (all 780 calls). Actual returned model
identifier, every call: **`gpt-5.4-2026-03-05`** (a single dated snapshot
of the requested model, correctly recognized as non-substituted; see
`provenance/live_evaluation_summary.json`). Full detail:
`tables/live_run_summary.{csv,md,tex}`.

## PRIMARY comparison: GPT-5.4 planner (majority vote) vs. deterministic `_fallback_plan()` baseline

Prompt-level majority vote of the planner's `recommended_tool` across the
3 repeats, scored against gold, compared against the repository's
existing deterministic `_fallback_plan()` baseline on the identical 120
supported prompts. This is **not** the planner-vs-router comparison
below -- see `analysis/baseline_comparison.py`.

| Cell | n | GPT-5.4 majority-vote accuracy [95% Wilson CI] | Deterministic baseline accuracy [95% Wilson CI] | Absolute diff (pp) | Exact McNemar (b, c, p) | Paired bootstrap 95% CI (diff, seed 20260818) |
|---|---|---|---|---|---|---|
| **Overall** | 120 | **98.3% [94.1%, 99.5%]** | **74.2% [65.7%, 81.2%]** | **+24.2** | b=29, c=0, **p=3.7e-09** | [+16.7pp, +31.7pp] |
| capability=causal_ate | 60 | 96.7% [88.6%, 99.1%] | 63.3% [50.7%, 74.4%] | +33.3 | b=20, c=0, p=2.0e-06 | [+21.7pp, +45.0pp] |
| capability=survival_adjusted_curves | 60 | 100.0% [93.98%, 100.0%] | 85.0% [73.9%, 91.9%] | +15.0 | b=9, c=0, p=0.0039 | [+6.7pp, +25.0pp] |
| category=explicit_method | 20 | 100.0% | 100.0% | 0.0 | b=0, c=0, p=1.0 | [0, 0] |
| category=formal_estimand | 20 | 100.0% | 95.0% | +5.0 | b=1, c=0, p=1.0 | [0, +15.0pp] |
| category=biomedical_domain | 20 | 95.0% | 40.0% | +55.0 | b=11, c=0, p=0.00098 | [+35.0pp, +75.0pp] |
| category=indirect_colloquial | 20 | 95.0% | 25.0% | +70.0 | b=14, c=0, p=0.00012 | [+50.0pp, +90.0pp] |
| category=noisy_verbose | 20 | 100.0% | 100.0% | 0.0 | b=0, c=0, p=1.0 | [0, 0] |
| category=near_boundary | 20 | 100.0% | 85.0% | +15.0 | b=3, c=0, p=0.25 | [0, +30.0pp] |

Full precision: `tables/planner_vs_deterministic_baseline.{csv,md,tex}`,
`results/processed/analysis_summary.json::planner_vs_deterministic_baseline`.
Figure: `figures/planner_vs_deterministic_baseline.{png,pdf}` (+
underlying `figures/planner_vs_deterministic_baseline.csv`).

**Interpretation.** In every one of the 9 cells above, `c = 0`: the
deterministic baseline never corrected a planner error. The two cells
where the deterministic baseline is statistically indistinguishable from
GPT-5.4 (`explicit_method`, `noisy_verbose`, both p=1.0, and
`near_boundary`, p=0.25) are exactly the cells where `_fallback_plan()`'s
keyword rules were designed to fire reliably (the literal trigger phrases
appear in the prompt text -- see `BENCHMARK_REVIEW.md`). The largest gaps
(`indirect_colloquial`, +70pp; `biomedical_domain`, +55pp) are exactly the
cells deliberately written to avoid those literal trigger words. This
gives a concrete, statistically supported account of what an LLM planner
adds over keyword matching: not average-case lift so much as robustness
to paraphrase.

## Router accuracy

**Run level** (n=360, 3 repeats × 120 prompts): **98.3% [96.4%, 99.2%]**
(354/360 correct). Fallback rate: **0.0%** (0/360). Unexplained API
failures: 0.

- ATE (`causal_ate`): 96.7% [92.9%, 98.5%] (174/180).
- Survival (`survival_adjusted_curves`): **100.0%** [97.9%, 100.0%] (180/180).
- By category: `explicit_method` 100.0%, `formal_estimand` 100.0%,
  `biomedical_domain` 95.0%, `indirect_colloquial` 95.0%,
  `noisy_verbose` 100.0%, `near_boundary` 100.0%.
- All 6 misrouted run-level calls follow one confusion pattern:
  **`causal_ate → linear_regression`** (6/6 errors; see
  `results/processed/analysis_summary.json::router_confusion_pairs`).
  No causal_ate prompt was ever misrouted to `binary_edrip`, the
  capability the benchmark was specifically designed to disambiguate
  against (see AUDIT.md Finding 2, BENCHMARK_REVIEW.md).

**Prompt-level majority vote** (n=120): **98.3% [94.1%, 99.5%]** (118/120),
identical to the run-level rate because every one of the 120 prompts had
a unanimous (3/3) router vote (`clean_majority_rate = 1.0`) -- majority
voting neither gained nor lost accuracy here, it only confirms the
run-level rate held under a stricter, prompt-level aggregation.

Full breakdown (all 12 capability×category cells, all 3 repeats
individually, majority-vote): `tables/router_accuracy_run_level.{csv,md,tex}`,
`tables/router_accuracy_majority_vote.{csv,md,tex}`. Figure:
`figures/router_accuracy_by_category.{png,pdf,csv}`.

## Router 3-run consistency and pairwise agreement

**100% full (3/3) consistency across all 120 prompts, overall, by
capability, and by every one of the 6 categories** -- every individual
repeat pair (1↔2, 1↔3, 2↔3) agreed on 100% of prompts in every stratum.
Full table: `tables/router_consistency.{csv,md,tex}`. Figure:
`figures/router_repeat_consistency.{png,pdf,csv}`.

## Planner structured-field accuracy

Run level (n=360): `recommended_tool` 98.3% [96.4%, 99.2%], `task_type`
98.3% [96.4%, 99.2%], `outcome_type` 99.2% [97.6%, 99.7%],
`all_fields_correct` (all three at once) 97.5% [95.3%, 98.7%].

By capability: causal_ate `recommended_tool`/`task_type` 96.7%,
`outcome_type` 98.3%; survival_adjusted_curves all three fields 100.0%.

Prompt-level majority vote on `recommended_tool` (n=120): 98.3% [94.1%,
99.5%] (118/120) -- the same accuracy as router majority vote, but not
the identical 2 prompts: the planner unanimously misroutes
`sup_causal_ate_biomedical_domain_05` (shared with the router's 2 misses)
and `sup_causal_ate_indirect_colloquial_03` (planner-only miss), while the
router unanimously misroutes `sup_causal_ate_biomedical_domain_05` and
`sup_causal_ate_indirect_colloquial_05` (router-only miss) -- both always
to `linear_regression`. This partial non-overlap is exactly what produces
the b=3, c=3 discordant pairs in the planner-router McNemar comparison
below (same accuracy, different per-item mistakes). Full breakdown:
`tables/planner_structured_field_accuracy.{csv,md,tex}`,
`tables/planner_accuracy_majority_vote.{csv,md,tex}`.

`likely_fallback_by_content_match`: 0/360 -- the live planner output was
never byte-identical to the pure-rule `_fallback_plan()` output, i.e. no
call shows the signature of a silent API-level fallback (see AUDIT.md
Finding 4).

## Planner-router agreement (NOT the primary baseline comparison above)

Matched-call agreement between the planner's `recommended_tool` and the
router's `predicted_capability_id` (n=360, same prompt+repeat): **98.3%
[96.4%, 99.2%]** (354/360 agree). Exact McNemar comparing router
correctness vs. planner correctness on the same matched calls: b=3, c=3,
**p=1.0** (no detectable difference between the two components' accuracy).
Paired bootstrap (seed 20260818) mean accuracy difference: 0.0 [-1.4pp,
+1.4pp]. Full detail: `results/processed/analysis_summary.json::
planner_router_agreement`, `::planner_vs_router_mcnemar`.

## Confusion matrices

- Run level (n=360): `figures/router_confusion_matrix.{png,pdf,csv}`,
  `tables/router_confusion_matrix_run_level.{csv,md,tex}`.
- Prompt-level majority vote, router (n=120):
  `figures/router_confusion_matrix_majority_vote.{png,pdf,csv}`,
  `tables/router_confusion_matrix_majority_vote.{csv,md,tex}`.
- Prompt-level majority vote, planner `recommended_tool` (n=120):
  `tables/planner_confusion_matrix_majority_vote.{csv,md,tex}`.

All errors, for both components, are `causal_ate` prompts unanimously
misrouted to `linear_regression` on all 3 repeats -- never a split
vote, and never any other confused capability (`binary_edrip`,
`hello_world`, `logistic_regression`, `summary_stats` each have zero
errors anywhere in the 360 supported-prompt calls for either component).
The router's 2 unanimous misses are `sup_causal_ate_biomedical_domain_05`
and `sup_causal_ate_indirect_colloquial_05`; the planner's 2 unanimous
misses are `sup_causal_ate_biomedical_domain_05` (shared with the router)
and `sup_causal_ate_indirect_colloquial_03` (planner-only) -- verified
directly against `results/raw/router_supported.jsonl` and
`results/raw/planner_supported.jsonl`. Run-level error count is exactly 6
per component (2 prompts × 3 repeats); majority-vote error count is
exactly 2 per component.

## Challenge-set results (descriptive only -- not an accuracy benchmark)

**Reiterating explicitly: the challenge set is descriptive only. It is
not an abstention/safety benchmark, and no accuracy, pass rate, or
abstention rate is computed for it anywhere in this evaluation** (see
`EVALUATION_PROTOCOL.md`, "Challenge set scoring", and AUDIT.md /
`benchmark/prompts_challenge.py`, which document why no challenge prompt
has a single correct gold capability).

Observed predicted-capability distributions (1 repeat per prompt, n=10 per
challenge_type per component):

**Router** (`predicted_capability_id`):
- `underspecified`: `causal_ate` 9/10, `summary_stats` 1/10.
- `unsupported_method`: `causal_ate` 8/10, `linear_regression` 1/10,
  `logistic_regression` 1/10.
- `out_of_scope_mixed`: `hello_world` 7/10, `linear_regression` 1/10,
  `summary_stats` 1/10, `causal_ate` 1/10.

**Planner** (`recommended_tool`):
- `underspecified`: `causal_ate` 9/10, `survival_adjusted_curves` 1/10.
- `unsupported_method`: `causal_ate` 10/10.
- `out_of_scope_mixed`: `summary_stats` 8/10, `linear_regression` 1/10,
  `causal_ate` 1/10.

Observations (descriptive, not scored): for `underspecified` prompts,
both components concentrate heavily on `causal_ate` -- consistent with
`causal_ate` being the more generically-described capability in the
registry (see AUDIT.md Section 1). For `unsupported_method` prompts
(instrumental variables, regression discontinuity, difference-in-
differences, synthetic control, mediation analysis, causal forests,
g-computation, front-door adjustment, matching, time-varying MSMs -- none
implemented by any registered capability), both components default almost
exclusively to `causal_ate` rather than any other capability, i.e. the
forced 7-way choice resolves toward the nearest generic causal-inference
capability rather than spreading across the registry or defaulting to
`binary_edrip`. For `out_of_scope_mixed` prompts, the router and planner
diverge: the router's most common pick is `hello_world` (7/10, plausibly
the shortest/most generic capability description), while the planner's
most common pick is `summary_stats` (8/10) -- the two components resolve
genuinely out-of-scope requests differently. Live-call outcomes for the
challenge set: 0 API failures, 0 fallback/likely-fallback events, 0 model
substitutions across all 60 challenge calls (30 router + 30 planner).

Full per-prompt detail (all 30 challenge prompts, router prediction,
planner recommendation, reasoning text): `tables/challenge_router_detail.
{csv,md,tex}`, `tables/challenge_planner_detail.{csv,md,tex}`.
Distribution table: `tables/challenge_prediction_distribution.{csv,md,tex}`.

## Static test suite and compile checks (final)

`python3 -m pytest evaluation/semantic_orchestration/tests/` — **74
passed**, 0 failed. All `.py` files under `evaluation/semantic_orchestration/`
compile cleanly under `python3 -m py_compile`.

## Generated artifact index

- `results/processed/analysis_summary.json` -- full machine-readable results.
- `results/processed/deterministic_baseline.json` -- deterministic baseline (unchanged from Step 1).
- `provenance/live_evaluation_summary.json` -- live-call outcome summary.
- `provenance/ENVIRONMENT.md` -- environment snapshots (Step 1 freeze + this finalization).
- `tables/*.{csv,md,tex}` -- 14 publication table sets (42 files) + 3 legacy Markdown-only tables.
- `figures/*.{png,pdf,csv}` -- 5 figures × 3 formats each (15 files).
