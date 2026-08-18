# Benchmark review

Self-review of `benchmark/prompts_supported.py` and
`benchmark/prompts_challenge.py`, performed before freezing
(`BENCHMARK_MANIFEST.json`). This is the record required by task step 1
("create static benchmark validators and near-duplicate checks... freeze
the benchmark... before any live evaluation").

## Checks performed and results

| Check | Method | Result |
|---|---|---|
| Exactly 120 supported prompts, 60/60 by capability | `tests/test_benchmark_integrity.py::test_supported_prompt_counts` | PASS |
| Exactly 10 prompts per (capability, category) cell, 12 cells | `tests/test_benchmark_integrity.py::test_supported_category_cell_counts_are_exactly_ten` | PASS |
| Exactly 30 challenge prompts, 10/10/10 by type | `tests/test_benchmark_integrity.py::test_challenge_prompt_counts` | PASS |
| Every supported prompt's `gold_capability_id == capability` | `test_supported_gold_capability_id_matches_capability` | PASS |
| No challenge prompt forces a single gold capability | `test_challenge_prompts_have_no_forced_gold_capability` | PASS |
| No causal_ate prompt combines binary-outcome framing with binary_edrip's specialized/robust/flexible/semiparametric language | `test_no_causal_ate_prompt_uses_binary_edrip_overlap_language` (checked against all 120 supported prompts, both capabilities) | PASS, 0 violations |
| No supported prompt uses reserved terms "edrip" / "semiparametric" | `test_no_supported_prompt_uses_reserved_binary_edrip_terms` | PASS |
| No unsupported_method challenge prompt accidentally describes binary_edrip (which IS implemented) | `test_unsupported_method_prompts_do_not_describe_binary_edrip` | PASS |
| Near-duplicate prompts (token-Jaccard > 0.70) | `test_no_near_duplicate_supported_prompts`, `test_no_near_duplicate_challenge_prompts`, all 8,925 supported pairs + 435 challenge pairs checked | PASS, 0 pairs above 0.70 |
| All prompt IDs unique | `test_supported_prompt_ids_are_unique`, `test_challenge_prompt_ids_are_unique` | PASS |
| Frozen JSONL byte-identical to rebuilding from source | `test_frozen_jsonl_matches_rebuilt_output` | PASS |
| Manifest hashes match files on disk | `test_manifest_integrity.py` | PASS |

## Manual review notes

- **Near-duplicate advisory (not a failure):** two pairs of prompts sit at
  Jaccard 0.58-0.61 (below the 0.70 failure threshold):
  `sup_survival_explicit_method_10` / `sup_survival_formal_estimand_10`,
  and `sup_survival_biomedical_domain_10` / `sup_survival_indirect_colloquial_10`.
  Both pairs share the hematology (leukemia induction regimen) clinical
  scenario, which is intentional: each of the 10 domains used per category
  is reused once per category to balance domain coverage, so some
  vocabulary overlap (`leukemia_trial.csv`, `cytogenetic_risk`, `induction
  regimen`) is expected between the same domain's entries across
  categories. The actual ask and phrasing register differ substantially
  between the two members of each pair (formal potential-outcomes notation
  vs. explicit-method phrasing; clinical narration vs. casual tone), so
  these are not treated as true near-duplicates.
- **`causal_ate` near_boundary cell** intentionally includes 5 binary-outcome
  prompts (30-day readmission, viral suppression, remission, hospitalization,
  acute rejection -- all "(yes/no)") to test the boundary against
  `binary_edrip` without tripping the overlap-language guard (Finding 2 in
  AUDIT.md): each explicitly asks for "the average/overall treatment
  effect," "a single number," or equivalent, and none uses
  robust/flexible/semiparametric framing.
- **`survival_adjusted_curves` near_boundary cell** intentionally uses
  "treatment effect" language in every one of its 10 prompts (the only
  survival category cell where that phrase appears at all -- enforced by
  `test_survival_prompts_do_not_request_single_summary_effect_except_near_boundary`)
  paired with an explicit request for the curve/trajectory rather than a
  single number, to test the boundary against `causal_ate` from the other
  direction.
- **Challenge `unsupported_method` prompts** were checked individually
  against all 7 capability descriptions (not just `binary_edrip`) to
  confirm none is answerable by `causal_ate`, `linear_regression`,
  `logistic_regression`, or `summary_stats` either: instrumental
  variables, regression discontinuity, difference-in-differences,
  synthetic control, mediation analysis, causal forests / heterogeneous
  treatment effects, time-varying-treatment g-computation / target trial
  emulation, front-door adjustment, matching (vs. weighting/adjustment),
  and time-varying marginal structural models with censoring weights are
  all absent from the registry.
- No automated spell-check or grammar-check was run; prompts were
  hand-authored and manually re-read once during this review pass.

## Outcome

No defects found. Benchmark frozen as version `1.0.0`
(`BENCHMARK_MANIFEST.json`, `frozen_at_utc` timestamp recorded there) with
no post-freeze edits required.
