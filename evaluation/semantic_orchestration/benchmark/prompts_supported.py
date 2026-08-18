"""
Literal source-of-truth data for the 120 supported benchmark prompts.

Frozen content — do not edit after `build_benchmark.py` has produced the
JSONL files and BENCHMARK_MANIFEST.json hashes have been recorded. Any
post-freeze edit must go through a new freeze cycle (see
EVALUATION_PROTOCOL.md, "Benchmark immutability").

Design rules enforced by `evaluation/semantic_orchestration/tests/
test_benchmark_integrity.py`:

1. Exactly 120 entries: 60 capability="causal_ate", 60
   capability="survival_adjusted_curves".
2. Exactly 10 entries per (capability, category) cell, categories =
   {explicit_method, formal_estimand, biomedical_domain,
   indirect_colloquial, noisy_verbose, near_boundary}.
3. `gold_capability_id` == `capability` for every supported entry (no
   ambiguous gold labels are permitted in the supported set).
4. No causal_ate prompt combines an explicitly-binary outcome with
   specialized/robust/flexible/semiparametric framing language, since
   that combination is the genuine overlap zone with the `binary_edrip`
   capability (see AUDIT.md, Finding 2). Enforced by
   `BINARY_EDRIP_OVERLAP_TERMS` / `BINARY_OUTCOME_TERMS` scan.
5. No supported prompt contains the literal strings "edrip" or
   "semiparametric" (those are reserved for the binary_edrip capability
   and for the unsupported-method challenge prompts).
6. No two prompts (within or across capabilities) are near-duplicates
   under a token-Jaccard threshold (see test_benchmark_integrity.py).
"""

from __future__ import annotations

CATEGORIES = [
    "explicit_method",
    "formal_estimand",
    "biomedical_domain",
    "indirect_colloquial",
    "noisy_verbose",
    "near_boundary",
]

# Terms that, in combination with BINARY_OUTCOME_TERMS, would push a
# causal_ate prompt into the binary_edrip overlap zone per cap_binary_edrip.json
# ("emphasizes robustness, flexibility, reduced parametric assumptions, or a
# specialized estimator"). causal_ate prompts must never combine the two lists.
BINARY_EDRIP_OVERLAP_TERMS = [
    "edrip",
    "semiparametric",
    "reduced parametric assumptions",
    "specialized estimator",
    "flexible model",
    "flexible estimator",
    "robust estimator",
    "doubly robust and flexible",
]
BINARY_OUTCOME_TERMS = [
    "binary outcome",
    "yes/no outcome",
    "yes or no outcome",
    "responded (yes/no)",
    "responder vs non-responder",
    "binary response",
]

# =========================================================================
# causal_ate — 60 prompts, 10 per category
# =========================================================================

CAUSAL_ATE_PROMPTS = [
    # --- explicit_method (10) --------------------------------------------
    {
        "id": "sup_causal_ate_explicit_method_01",
        "category": "explicit_method",
        "domain": "cardiology",
        "text": (
            "Using inverse probability weighting, estimate the average treatment "
            "effect of the new anticoagulant on 90-day systolic blood pressure "
            "change in cardiac_trial.csv, adjusting for age, renal_function, and "
            "baseline_bp."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_02",
        "category": "explicit_method",
        "domain": "diabetes",
        "text": (
            "Please fit a propensity-score-adjusted average treatment effect model "
            "comparing metformin to placebo on HbA1c change in diabetes_cohort.csv, "
            "controlling for bmi and baseline_hba1c."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_03",
        "category": "explicit_method",
        "domain": "orthopedics",
        "text": (
            "Run a doubly robust ATE estimator on rehab_outcomes.csv to quantify "
            "the causal effect of early physical therapy versus standard care on "
            "six-week mobility score, covariates age and baseline_mobility."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_04",
        "category": "explicit_method",
        "domain": "oncology",
        "text": (
            "Compute the average treatment effect of chemo_regimen on tumor_size_"
            "reduction in oncology_trial.csv using covariate-adjusted causal "
            "estimation with covariates stage and prior_treatment."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_05",
        "category": "explicit_method",
        "domain": "mental_health",
        "text": (
            "Apply an IPW-based treatment effect estimator to depression_trial.csv "
            "to get the ATE of the new antidepressant on change in phq9_score, "
            "adjusting for baseline_phq9 and age."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_06",
        "category": "explicit_method",
        "domain": "public_health",
        "text": (
            "Estimate the causal average treatment effect using propensity score "
            "matching-style adjustment for the smoking cessation program on "
            "cigarettes_per_day at 6 months in cessation_study.csv, covariates "
            "baseline_cigs and years_smoking."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_07",
        "category": "explicit_method",
        "domain": "obesity",
        "text": (
            "Using a standard causal inference workflow with covariate adjustment, "
            "compute the average treatment effect of bariatric_surgery on "
            "weight_loss_kg in obesity_cohort.csv, covariates baseline_weight and "
            "diabetes_status."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_08",
        "category": "explicit_method",
        "domain": "hepatology",
        "text": (
            "Run the average-treatment-effect tool on PBC_ate5y.csv to estimate "
            "the treatment vs control causal effect of D-penicillamine on "
            "bilirubin_level, adjusting for age and edema."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_09",
        "category": "explicit_method",
        "domain": "critical_care",
        "text": (
            "Estimate the ATE of early goal-directed fluid therapy on icu_length_"
            "of_stay_days in icu_fluids.csv using propensity-score adjustment for "
            "apache_score and admission_source."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_explicit_method_10",
        "category": "explicit_method",
        "domain": "rheumatology",
        "text": (
            "Use covariate-adjusted average treatment effect estimation to compare "
            "biologic_therapy to standard_dmard on das28_score_change in "
            "ra_registry.csv, covariates disease_duration and baseline_das28."
        ),
        "gold_outcome_type": "continuous",
    },
    # --- formal_estimand (10) ---------------------------------------------
    {
        "id": "sup_causal_ate_formal_estimand_01",
        "category": "formal_estimand",
        "domain": "cardiology",
        "text": (
            "Identify and estimate the estimand E[Y(1) - Y(0)] where Y is 90-day "
            "LDL cholesterol level and A is statin assignment, using "
            "statin_trial.csv adjusted for baseline_ldl and age under "
            "conditional ignorability."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_02",
        "category": "formal_estimand",
        "domain": "diabetes",
        "text": (
            "Under the potential outcomes framework, compute tau = E[Y(a=1)] - "
            "E[Y(a=0)] for the effect of insulin_intensification on fasting_"
            "glucose in glucose_control.csv, adjusting for confounders bmi and "
            "duration_diabetes to satisfy no-unmeasured-confounding."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_03",
        "category": "formal_estimand",
        "domain": "oncology",
        "text": (
            "Estimate the population average causal effect ATE = E[Y_i(1) - "
            "Y_i(0)] of adjuvant_therapy on progression_free_days in "
            "oncology_registry.csv, with propensity model conditioned on "
            "stage and biomarker_level."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_04",
        "category": "formal_estimand",
        "domain": "nephrology",
        "text": (
            "Assuming exchangeability given covariates, estimate the marginal "
            "treatment effect delta = E[Y|do(A=1)] - E[Y|do(A=0)] of ace_"
            "inhibitor on egfr_change_12mo in ckd_cohort.csv, covariates "
            "baseline_egfr and proteinuria."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_05",
        "category": "formal_estimand",
        "domain": "maternal_health",
        "text": (
            "Estimate the counterfactual contrast E[birth_weight | do(A=1)] - "
            "E[birth_weight | do(A=0)] for prenatal_supplement exposure in "
            "maternal_cohort.csv, adjusting for maternal_age and gestational_age "
            "at enrollment for identifiability."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_06",
        "category": "formal_estimand",
        "domain": "psychiatry",
        "text": (
            "Under conditional exchangeability Y(a) independent of A given X, "
            "estimate the average causal effect of cbt_intervention on anxiety_"
            "score_change in anxiety_trial.csv, X = {baseline_anxiety, age}."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_07",
        "category": "formal_estimand",
        "domain": "respiratory",
        "text": (
            "Compute the g-formula-consistent average treatment effect estimand "
            "for bronchodilator_dose on fev1_change in copd_trial.csv, "
            "conditioning the propensity model on baseline_fev1 and smoking_"
            "pack_years to satisfy positivity and ignorability."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_08",
        "category": "formal_estimand",
        "domain": "infectious_disease",
        "text": (
            "Estimate the marginal structural contrast E[Y(1)] - E[Y(0)] of "
            "antiviral_regimen on viral_load_reduction in hiv_cohort.csv, "
            "adjusting for cd4_baseline and adherence_score under the standard "
            "identifiability assumptions."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_09",
        "category": "formal_estimand",
        "domain": "pain_management",
        "text": (
            "Given the potential outcomes Y(0), Y(1) for pain_score at 4 weeks, "
            "estimate the population-level ATE of nerve_block_procedure in "
            "pain_clinic.csv, adjusting for baseline_pain and procedure_history "
            "to identify the causal contrast."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_formal_estimand_10",
        "category": "formal_estimand",
        "domain": "transplant",
        "text": (
            "Estimate tau_ATE = E[creatinine(a=1)] - E[creatinine(a=0)] for "
            "induction_protocol in transplant_registry.csv, with the propensity "
            "score conditioned on donor_type and hla_mismatch to satisfy "
            "conditional ignorability."
        ),
        "gold_outcome_type": "continuous",
    },
    # --- biomedical_domain (10) --------------------------------------------
    {
        "id": "sup_causal_ate_biomedical_domain_01",
        "category": "biomedical_domain",
        "domain": "oncology",
        "text": (
            "Does the new adjuvant chemotherapy actually lower average tumor size "
            "at 3 months compared to the standard regimen, once we account for "
            "baseline stage and prior surgery, in oncology_trial.csv?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_02",
        "category": "biomedical_domain",
        "domain": "cardiology",
        "text": (
            "In heart_failure.csv, how much does starting the new beta-blocker "
            "reduce average NT-proBNP levels at 6 months versus not starting it, "
            "after adjusting for baseline ejection fraction and age?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_03",
        "category": "biomedical_domain",
        "domain": "diabetes",
        "text": (
            "For patients in insulin_pump_study.csv, what is the average "
            "difference in time-in-range glucose percentage attributable to the "
            "insulin pump versus multiple daily injections, controlling for "
            "baseline HbA1c and diabetes duration?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_04",
        "category": "biomedical_domain",
        "domain": "orthopedics",
        "text": (
            "Compare, on average, how much extra knee flexion at 12 weeks the "
            "accelerated rehab protocol produces relative to standard rehab in "
            "knee_surgery.csv, holding baseline flexion and age constant."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_05",
        "category": "biomedical_domain",
        "domain": "hepatology",
        "text": (
            "Using PBC_agent.csv, quantify how the experimental drug shifts "
            "average serum bilirubin relative to placebo once age and edema "
            "status are accounted for."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_06",
        "category": "biomedical_domain",
        "domain": "nephrology",
        "text": (
            "In dialysis_outcomes.csv, what's the causal impact on average "
            "monthly hospitalization days of switching patients to home dialysis "
            "versus in-center dialysis, adjusting for comorbidity_index?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_07",
        "category": "biomedical_domain",
        "domain": "pediatrics",
        "text": (
            "Does early nutritional supplementation genuinely raise average "
            "weight-for-age z-scores in preterm_infants.csv relative to standard "
            "feeding, after adjusting for gestational_age and birth_weight?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_08",
        "category": "biomedical_domain",
        "domain": "geriatrics",
        "text": (
            "In falls_prevention.csv, how much does the balance-training program "
            "change the average number of falls per year for older adults, once "
            "baseline mobility score and medication_count are controlled for?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_09",
        "category": "biomedical_domain",
        "domain": "endocrinology",
        "text": (
            "For thyroid_study.csv, what is the average causal change in TSH "
            "level from switching patients to the extended-release levothyroxine "
            "formulation, controlling for baseline TSH and weight?"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_biomedical_domain_10",
        "category": "biomedical_domain",
        "domain": "sports_medicine",
        "text": (
            "Using return_to_play.csv, estimate how much faster, on average, "
            "athletes return to play under the new concussion protocol compared "
            "to the old one, adjusting for baseline_symptom_score and age."
        ),
        "gold_outcome_type": "continuous",
    },
    # --- indirect_colloquial (10) -------------------------------------------
    {
        "id": "sup_causal_ate_indirect_colloquial_01",
        "category": "indirect_colloquial",
        "domain": "oncology",
        "text": (
            "so did the new chemo actually help shrink tumors more than the old "
            "one on average, once you control for stage and age? data's in "
            "oncology_trial.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_02",
        "category": "indirect_colloquial",
        "domain": "diabetes",
        "text": (
            "can you tell me if the new drug really moved the needle on blood "
            "sugar compared to placebo, accounting for people's starting bmi? "
            "file is diabetes_cohort.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_03",
        "category": "indirect_colloquial",
        "domain": "mental_health",
        "text": (
            "does the therapy group end up better off on average than the "
            "control group in terms of anxiety scores, after we adjust for how "
            "bad they were at baseline? using anxiety_trial.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_04",
        "category": "indirect_colloquial",
        "domain": "orthopedics",
        "text": (
            "compare the two rehab programs and tell me who actually did better "
            "on knee flexion once we account for age and starting flexibility, "
            "knee_surgery.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_05",
        "category": "indirect_colloquial",
        "domain": "cardiology",
        "text": (
            "i want to know if the new beta blocker is really lowering NT-proBNP "
            "compared to not taking it, adjusting for ejection fraction. use "
            "heart_failure.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_06",
        "category": "indirect_colloquial",
        "domain": "public_health",
        "text": (
            "what's the honest effect of the smoking program on how much people "
            "smoke afterward, controlling for how much they smoked before? "
            "cessation_study.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_07",
        "category": "indirect_colloquial",
        "domain": "obesity",
        "text": (
            "how much weight did people really lose because of the surgery, not "
            "just because of who chose to have it, controlling for starting "
            "weight and diabetes status? obesity_cohort.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_08",
        "category": "indirect_colloquial",
        "domain": "pediatrics",
        "text": (
            "did the feeding change actually help preemie weight gain or is that "
            "just because healthier babies got it? adjust for gestational age, "
            "preterm_infants.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_09",
        "category": "indirect_colloquial",
        "domain": "geriatrics",
        "text": (
            "is the balance program actually cutting down falls or is it just "
            "the healthier patients doing it? control for baseline mobility, "
            "falls_prevention.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_indirect_colloquial_10",
        "category": "indirect_colloquial",
        "domain": "nephrology",
        "text": (
            "does switching people to home dialysis actually cut hospital days, "
            "or is that just because the healthier ones switch? adjust for "
            "comorbidity_index, dialysis_outcomes.csv"
        ),
        "gold_outcome_type": "continuous",
    },
    # --- noisy_verbose (10) -------------------------------------------------
    {
        "id": "sup_causal_ate_noisy_verbose_01",
        "category": "noisy_verbose",
        "domain": "cardiology",
        "text": (
            "ok so this is a bit long but basically our team ran a trial and "
            "honestly the enrollment was messier than we wanted, some sites "
            "were slow, but anyway what I really need right now, before the "
            "Friday meeting, is just the average treatment effect of the new "
            "anticoagulant on blood pressure change, comparing treated vs "
            "untreated, and please adjust for age and kidney function since "
            "those clearly differ between groups, the file is cardiac_trial.csv, "
            "thanks in advance and sorry for the rambling."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_02",
        "category": "noisy_verbose",
        "domain": "diabetes",
        "text": (
            "hey, quick context dump before the actual ask: we've got about "
            "18 months of follow up, a few dropouts, IRB approved everything "
            "twice because of an amendment, whatever, the point is I need the "
            "causal effect, not just a correlation, of metformin vs placebo on "
            "HbA1c, definitely adjust for baseline HbA1c and bmi because those "
            "were unbalanced at randomization apparently, dataset is "
            "diabetes_cohort.csv, let me know if you need anything else from me."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_03",
        "category": "noisy_verbose",
        "domain": "oncology",
        "text": (
            "sorry for the wall of text, long week, but here goes: we have "
            "oncology_trial.csv, patients got either the new chemo regimen or "
            "the old one, and I keep getting asked by the PI whether the new "
            "one is actually better on average for shrinking tumors, on "
            "average across everyone not just responders, please adjust for "
            "stage and prior treatment since sicker patients tended to get the "
            "old regimen, appreciate it."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_04",
        "category": "noisy_verbose",
        "domain": "mental_health",
        "text": (
            "not sure if this is the right format but I'll try, we ran a small "
            "study on the new antidepressant, some patients also had comorbid "
            "anxiety which complicates things a bit but let's set that aside "
            "for now, what I actually need is the average treatment effect of "
            "drug vs placebo on PHQ-9 score change, adjusted for baseline PHQ-9 "
            "and age, file is depression_trial.csv, apologies if I'm missing "
            "context you need."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_05",
        "category": "noisy_verbose",
        "domain": "obesity",
        "text": (
            "long story short, and I promise I'll get to the point, the surgery "
            "cohort was collected across three hospitals with slightly "
            "different protocols which is annoying but shouldn't matter much "
            "here, what matters is I need the average causal effect of "
            "bariatric surgery on weight loss in kg, please control for "
            "baseline weight and diabetes status, obesity_cohort.csv, thanks "
            "for bearing with the rambling explanation."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_06",
        "category": "noisy_verbose",
        "domain": "nephrology",
        "text": (
            "apologies in advance for the long message, our data team had some "
            "back and forth about column naming so hopefully it's clean now, "
            "anyway the ask: home dialysis vs in-center dialysis, I need the "
            "average treatment effect on hospitalization days per month, "
            "adjust for comorbidity index because sicker patients often stay "
            "in-center, dialysis_outcomes.csv, let me know if the columns look "
            "off."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_07",
        "category": "noisy_verbose",
        "domain": "respiratory",
        "text": (
            "this got long, sorry, but basically after the site audit we "
            "trimmed a few rows for data quality reasons so the n is slightly "
            "smaller than before, that shouldn't change the ask though: I need "
            "the causal average effect of the new bronchodilator dose on FEV1 "
            "change, adjusted for baseline FEV1 and smoking pack years, file is "
            "copd_trial.csv, appreciate the help as always."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_08",
        "category": "noisy_verbose",
        "domain": "pain_management",
        "text": (
            "quick heads up this message is longer than usual because I'm "
            "copy-pasting from an email thread with the clinic, but the actual "
            "request is simple: average treatment effect of the nerve block "
            "procedure on pain score at 4 weeks, adjust for baseline pain and "
            "procedure history since repeat patients respond differently, "
            "pain_clinic.csv, sorry again for the noise in this message."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_09",
        "category": "noisy_verbose",
        "domain": "transplant",
        "text": (
            "ok bear with me, this is a bit of a brain dump, the registry data "
            "spans several transplant centers and there was a data cleaning "
            "pass last month, but anyway what I need, cutting through all "
            "that, is the average causal effect of the induction protocol on "
            "creatinine level, adjusted for donor type and HLA mismatch, "
            "transplant_registry.csv, thank you for reading through all this."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_noisy_verbose_10",
        "category": "noisy_verbose",
        "domain": "sports_medicine",
        "text": (
            "sorry, going to ramble a bit here because I want to give full "
            "context, the concussion protocol changed midway through the "
            "season and some athletes got the new one and some the old one, "
            "not randomized unfortunately, so I need the average treatment "
            "effect on days to return to play, adjusted for baseline symptom "
            "score and age, return_to_play.csv, appreciate you working through "
            "my long message."
        ),
        "gold_outcome_type": "continuous",
    },
    # --- near_boundary (10) --------------------------------------------------
    # Boundary vs survival_adjusted_curves: mentions "over time" / longitudinal
    # framing but explicitly asks for a single overall effect number, not a
    # curve, and does not request Kaplan-Meier/survival-probability output.
    # Boundary vs binary_edrip: some entries have a binary outcome, but use
    # plain ATE language with no robustness/flexibility/semiparametric framing
    # (see BINARY_EDRIP_OVERLAP_TERMS guard in test_benchmark_integrity.py).
    {
        "id": "sup_causal_ate_near_boundary_01",
        "category": "near_boundary",
        "domain": "oncology",
        "text": (
            "Across the whole follow-up window, what is the single average "
            "treatment effect of the new chemo regimen on tumor size, not a "
            "curve over time, just the one overall number, adjusting for stage "
            "and prior treatment, oncology_trial.csv."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_near_boundary_02",
        "category": "near_boundary",
        "domain": "cardiology",
        "text": (
            "I don't need survival curves here, just give me the overall "
            "average treatment effect of the anticoagulant on 90-day blood "
            "pressure change, adjusting for age and renal function, "
            "cardiac_trial.csv."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_near_boundary_03",
        "category": "near_boundary",
        "domain": "orthopedics",
        "text": (
            "Was there a readmission for any reason within 30 days for the "
            "accelerated rehab group versus standard rehab, on average? I want "
            "the average treatment effect on that binary outcome, not a "
            "time-to-event curve, adjust for age and baseline mobility, "
            "knee_surgery.csv."
        ),
        "gold_outcome_type": "binary",
    },
    {
        "id": "sup_causal_ate_near_boundary_04",
        "category": "near_boundary",
        "domain": "infectious_disease",
        "text": (
            "For hiv_cohort.csv, did patients achieve viral suppression "
            "(yes/no) by month 6 more often under the new antiviral regimen on "
            "average, adjusting for cd4_baseline and adherence_score? I just "
            "want the overall average effect, not a probability curve over "
            "time."
        ),
        "gold_outcome_type": "binary",
    },
    {
        "id": "sup_causal_ate_near_boundary_05",
        "category": "near_boundary",
        "domain": "obesity",
        "text": (
            "Even though weight was tracked monthly, I only need the average "
            "treatment effect of the surgery on 12-month weight loss as a "
            "single summary number, adjusting for baseline weight and diabetes "
            "status, not a longitudinal trajectory, obesity_cohort.csv."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_near_boundary_06",
        "category": "near_boundary",
        "domain": "psychiatry",
        "text": (
            "Did patients reach remission (yes/no) by week 12 more on the new "
            "antidepressant than placebo on average, adjusting for baseline "
            "PHQ-9 and age? A single average effect estimate is fine, I don't "
            "need a time-to-remission curve, depression_trial.csv."
        ),
        "gold_outcome_type": "binary",
    },
    {
        "id": "sup_causal_ate_near_boundary_07",
        "category": "near_boundary",
        "domain": "nephrology",
        "text": (
            "Give me the overall average causal effect of home dialysis on "
            "whether a patient was hospitalized at least once (yes/no) in the "
            "first year, adjusted for comorbidity_index, dialysis_outcomes.csv "
            "-- one summary effect, not a curve across the year."
        ),
        "gold_outcome_type": "binary",
    },
    {
        "id": "sup_causal_ate_near_boundary_08",
        "category": "near_boundary",
        "domain": "respiratory",
        "text": (
            "Although FEV1 was measured repeatedly, I just want the single "
            "average treatment effect of the new bronchodilator dose on FEV1 "
            "change from baseline to final visit, adjusting for baseline FEV1 "
            "and smoking pack years, copd_trial.csv."
        ),
        "gold_outcome_type": "continuous",
    },
    {
        "id": "sup_causal_ate_near_boundary_09",
        "category": "near_boundary",
        "domain": "transplant",
        "text": (
            "Did the induction protocol reduce whether a patient experienced "
            "acute rejection (yes/no) within the first 90 days, on average, "
            "adjusting for donor type and HLA mismatch? A single overall "
            "treatment effect number is what I need, not a rejection-free "
            "probability curve, transplant_registry.csv."
        ),
        "gold_outcome_type": "binary",
    },
    {
        "id": "sup_causal_ate_near_boundary_10",
        "category": "near_boundary",
        "domain": "geriatrics",
        "text": (
            "Even though falls were logged monthly across the year, just give "
            "me one average treatment effect number for the balance program on "
            "total falls over the year, adjusted for baseline mobility and "
            "medication_count, not a monthly trend, falls_prevention.csv."
        ),
        "gold_outcome_type": "continuous",
    },
]

assert len(CAUSAL_ATE_PROMPTS) == 60

# =========================================================================
# survival_adjusted_curves — 60 prompts, 10 per category
# =========================================================================

SURVIVAL_PROMPTS = [
    # --- explicit_method (10) --------------------------------------------
    {
        "id": "sup_survival_explicit_method_01",
        "category": "explicit_method",
        "domain": "hepatology",
        "text": (
            "Plot IPTW-adjusted Kaplan-Meier survival curves for PBC_agent.csv "
            "comparing D-penicillamine vs placebo, with time-to-death, event "
            "indicator, and treatment group, adjusting for age and edema."
        ),
    },
    {
        "id": "sup_survival_explicit_method_02",
        "category": "explicit_method",
        "domain": "oncology",
        "text": (
            "Generate confounder-adjusted survival curves using inverse "
            "probability of treatment weighting for oncology_survival.csv, "
            "time_to_event, event_indicator, and chemo_group, covariates stage "
            "and age."
        ),
    },
    {
        "id": "sup_survival_explicit_method_03",
        "category": "explicit_method",
        "domain": "nephrology",
        "text": (
            "Produce adjusted Kaplan-Meier-style survival curves comparing home "
            "dialysis vs in-center dialysis groups in dialysis_survival.csv, "
            "using time_months, death_event, and dialysis_group, adjusted for "
            "comorbidity_index."
        ),
    },
    {
        "id": "sup_survival_explicit_method_04",
        "category": "explicit_method",
        "domain": "cardiology",
        "text": (
            "Estimate and plot IPTW survival curves for cardiac_outcomes.csv "
            "with columns time_to_mace, mace_event, and treatment_arm, "
            "adjusting for baseline_ejection_fraction and age."
        ),
    },
    {
        "id": "sup_survival_explicit_method_05",
        "category": "explicit_method",
        "domain": "transplant",
        "text": (
            "Please compute confounder-adjusted event-free probability curves "
            "over time for graft_survival.csv, group=induction_protocol, "
            "time=months_to_failure, event=graft_failure, covariates donor_type "
            "and hla_mismatch."
        ),
    },
    {
        "id": "sup_survival_explicit_method_06",
        "category": "explicit_method",
        "domain": "infectious_disease",
        "text": (
            "Visualize adjusted survival curves via IPTW for hiv_progression.csv "
            "comparing antiviral_regimen groups, time_to_aids, aids_event, "
            "adjusting for cd4_baseline."
        ),
    },
    {
        "id": "sup_survival_explicit_method_07",
        "category": "explicit_method",
        "domain": "respiratory",
        "text": (
            "Build IPTW-weighted Kaplan-Meier curves for copd_exacerbation.csv, "
            "group=inhaler_type, time=days_to_exacerbation, "
            "event=exacerbation_event, covariates baseline_fev1 and "
            "smoking_pack_years."
        ),
    },
    {
        "id": "sup_survival_explicit_method_08",
        "category": "explicit_method",
        "domain": "oncology",
        "text": (
            "Estimate adjusted survival curves comparing adjuvant_therapy arms "
            "in relapse_registry.csv using time_to_relapse and relapse_event, "
            "weighted by inverse probability of treatment based on stage and "
            "biomarker_level."
        ),
    },
    {
        "id": "sup_survival_explicit_method_09",
        "category": "explicit_method",
        "domain": "geriatrics",
        "text": (
            "Plot Kaplan-Meier adjusted survival curves for nursing_home_"
            "study.csv, group=care_model, time=days_to_readmission, "
            "event=readmission_event, adjusting for baseline_frailty_score."
        ),
    },
    {
        "id": "sup_survival_explicit_method_10",
        "category": "explicit_method",
        "domain": "hematology",
        "text": (
            "Compute IPTW-adjusted survival curves for leukemia_trial.csv "
            "comparing induction_regimen groups, time_to_relapse_or_death, "
            "event_indicator, covariates cytogenetic_risk and age."
        ),
    },
    # --- formal_estimand (10) -----------------------------------------------
    {
        "id": "sup_survival_formal_estimand_01",
        "category": "formal_estimand",
        "domain": "hepatology",
        "text": (
            "Estimate the adjusted survival functions S(t | do(A=1)) and "
            "S(t | do(A=0)) for D-penicillamine assignment in PBC_agent.csv, "
            "time_to_death, death_event, treatment_group, conditioning the "
            "weights on age and edema for exchangeability."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_02",
        "category": "formal_estimand",
        "domain": "oncology",
        "text": (
            "Compute the confounder-adjusted counterfactual survival curves "
            "S_a(t) for a in {0,1} corresponding to chemo_group in "
            "oncology_survival.csv, using time_to_event and event_indicator, "
            "with weights conditioned on stage and age to satisfy conditional "
            "ignorability."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_03",
        "category": "formal_estimand",
        "domain": "nephrology",
        "text": (
            "Under the assumption of no unmeasured confounding given X = "
            "{comorbidity_index}, estimate the IPTW-adjusted survival curves "
            "S(t|A=a,do) for dialysis_group in dialysis_survival.csv using "
            "time_months and death_event."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_04",
        "category": "formal_estimand",
        "domain": "cardiology",
        "text": (
            "Estimate the treatment-specific survival functions Pr(T > t | "
            "do(A=a)) for treatment_arm in cardiac_outcomes.csv, time_to_mace "
            "and mace_event, adjusting the weights on baseline_ejection_"
            "fraction and age."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_05",
        "category": "formal_estimand",
        "domain": "transplant",
        "text": (
            "Compute the counterfactual event-free probability curves S_a(t) = "
            "Pr(T(a) > t) for induction_protocol in graft_survival.csv, "
            "months_to_failure and graft_failure, weights conditioned on "
            "donor_type and hla_mismatch."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_06",
        "category": "formal_estimand",
        "domain": "infectious_disease",
        "text": (
            "Assuming conditional exchangeability given cd4_baseline, estimate "
            "the adjusted survival curves S(t|do(A=1)) vs S(t|do(A=0)) for "
            "antiviral_regimen in hiv_progression.csv, time_to_aids and "
            "aids_event."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_07",
        "category": "formal_estimand",
        "domain": "respiratory",
        "text": (
            "Estimate the marginal structural survival curves for inhaler_type "
            "in copd_exacerbation.csv, S_a(t) for a=0,1, using days_to_"
            "exacerbation and exacerbation_event, IPTW weights on baseline_"
            "fev1 and smoking_pack_years to satisfy positivity."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_08",
        "category": "formal_estimand",
        "domain": "oncology",
        "text": (
            "Identify and estimate the treatment-specific counterfactual "
            "survival curves for adjuvant_therapy in relapse_registry.csv "
            "under do(A=a), using time_to_relapse and relapse_event, weights "
            "conditioned on stage and biomarker_level."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_09",
        "category": "formal_estimand",
        "domain": "geriatrics",
        "text": (
            "Under exchangeability given baseline_frailty_score, estimate the "
            "adjusted survival curves S(t|A=a) for care_model in nursing_"
            "home_study.csv using days_to_readmission and readmission_event."
        ),
    },
    {
        "id": "sup_survival_formal_estimand_10",
        "category": "formal_estimand",
        "domain": "hematology",
        "text": (
            "Estimate the counterfactual survival functions Pr(T(a) > t) for "
            "induction_regimen in leukemia_trial.csv, time_to_relapse_or_death "
            "and event_indicator, IPTW-adjusted for cytogenetic_risk and age."
        ),
    },
    # --- biomedical_domain (10) ---------------------------------------------
    {
        "id": "sup_survival_biomedical_domain_01",
        "category": "biomedical_domain",
        "domain": "hepatology",
        "text": (
            "For PBC_agent.csv, show how survival differs over time between "
            "patients on D-penicillamine and those on placebo, adjusting for "
            "age and edema, since sicker patients may have been assigned "
            "differently."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_02",
        "category": "biomedical_domain",
        "domain": "oncology",
        "text": (
            "In oncology_survival.csv, I want to see how overall survival "
            "unfolds over time for patients on the new chemo regimen compared "
            "to the old one, controlling for stage and age imbalances between "
            "the groups."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_03",
        "category": "biomedical_domain",
        "domain": "nephrology",
        "text": (
            "Show me survival over time for home dialysis versus in-center "
            "dialysis patients in dialysis_survival.csv, adjusted for how sick "
            "patients were at baseline (comorbidity_index)."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_04",
        "category": "biomedical_domain",
        "domain": "cardiology",
        "text": (
            "How does the probability of staying free of a major cardiac event "
            "change over time for each treatment arm in cardiac_outcomes.csv, "
            "after accounting for baseline ejection fraction and age?"
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_05",
        "category": "biomedical_domain",
        "domain": "transplant",
        "text": (
            "For graft_survival.csv, I need to see how graft survival unfolds "
            "over time across induction protocols, adjusting for donor type "
            "and HLA mismatch since those weren't balanced at transplant."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_06",
        "category": "biomedical_domain",
        "domain": "infectious_disease",
        "text": (
            "Show the AIDS-free probability over time for each antiviral "
            "regimen group in hiv_progression.csv, adjusted for baseline CD4 "
            "count differences between groups."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_07",
        "category": "biomedical_domain",
        "domain": "respiratory",
        "text": (
            "In copd_exacerbation.csv, how does the chance of staying "
            "exacerbation-free change over time between the two inhaler types, "
            "after adjusting for baseline lung function and smoking history?"
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_08",
        "category": "biomedical_domain",
        "domain": "oncology",
        "text": (
            "Using relapse_registry.csv, show relapse-free survival over time "
            "for each adjuvant therapy arm, adjusting for tumor stage and "
            "biomarker level since assignment wasn't random."
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_09",
        "category": "biomedical_domain",
        "domain": "geriatrics",
        "text": (
            "For nursing_home_study.csv, how does readmission-free time unfold "
            "for each care model, once we adjust for how frail patients were "
            "at baseline?"
        ),
    },
    {
        "id": "sup_survival_biomedical_domain_10",
        "category": "biomedical_domain",
        "domain": "hematology",
        "text": (
            "Show relapse-and-death-free survival over time for each induction "
            "regimen in leukemia_trial.csv, adjusted for cytogenetic risk "
            "group and age at diagnosis."
        ),
    },
    # --- indirect_colloquial (10) --------------------------------------------
    {
        "id": "sup_survival_indirect_colloquial_01",
        "category": "indirect_colloquial",
        "domain": "hepatology",
        "text": (
            "can you show me how the two drug groups compare over time in "
            "PBC_agent.csv, like a survival curve thing, adjusted for age and "
            "edema? want to see the whole trajectory not just one number"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_02",
        "category": "indirect_colloquial",
        "domain": "oncology",
        "text": (
            "can we get a picture of how survival trends over time for the two "
            "chemo groups in oncology_survival.csv, adjusting for stage since "
            "the sicker patients probably got the old regimen more"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_03",
        "category": "indirect_colloquial",
        "domain": "nephrology",
        "text": (
            "I basically want to watch how the two dialysis groups drift apart "
            "over time in terms of who's still alive, adjusting for how sick "
            "they were to start, dialysis_survival.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_04",
        "category": "indirect_colloquial",
        "domain": "cardiology",
        "text": (
            "can you draw me the curve showing how each treatment group holds "
            "up event-free over the follow-up period, adjusted for their "
            "starting heart function, cardiac_outcomes.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_05",
        "category": "indirect_colloquial",
        "domain": "transplant",
        "text": (
            "show me the trajectory of graft survival for each induction group "
            "over time, not a single number, adjusted for donor type since "
            "that wasn't balanced, graft_survival.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_06",
        "category": "indirect_colloquial",
        "domain": "infectious_disease",
        "text": (
            "give me the picture of how each antiviral group's AIDS-free "
            "chances change as time goes on, adjusted for their starting CD4 "
            "counts, hiv_progression.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_07",
        "category": "indirect_colloquial",
        "domain": "respiratory",
        "text": (
            "let's see the curves for how long each inhaler group stays "
            "exacerbation-free over time, adjusted for lung function at the "
            "start, copd_exacerbation.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_08",
        "category": "indirect_colloquial",
        "domain": "oncology",
        "text": (
            "I want the visual trend of relapse-free survival over time for "
            "each therapy arm, not a summary number, adjusted for tumor stage, "
            "relapse_registry.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_09",
        "category": "indirect_colloquial",
        "domain": "geriatrics",
        "text": (
            "show me how readmission-free time trends for each care model over "
            "the follow-up, adjusted for baseline frailty, nursing_home_"
            "study.csv"
        ),
    },
    {
        "id": "sup_survival_indirect_colloquial_10",
        "category": "indirect_colloquial",
        "domain": "hematology",
        "text": (
            "can you plot how each induction regimen group trends over time in "
            "terms of staying relapse and death free, adjusted for "
            "cytogenetic risk, leukemia_trial.csv"
        ),
    },
    # --- noisy_verbose (10) ---------------------------------------------------
    {
        "id": "sup_survival_noisy_verbose_01",
        "category": "noisy_verbose",
        "domain": "hepatology",
        "text": (
            "sorry for the long message, we had some data entry issues early "
            "on but the registry team fixed most of it, anyway what I need is "
            "to see, over the whole follow-up period, how survival differs "
            "between the D-penicillamine group and placebo group as a curve, "
            "not a single number, adjusting for age and edema because those "
            "weren't balanced, file is PBC_agent.csv, thanks for your "
            "patience."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_02",
        "category": "noisy_verbose",
        "domain": "oncology",
        "text": (
            "long week, sorry in advance for the rambling, but the tumor board "
            "keeps asking for a visual of how the two chemo regimens compare "
            "over time in terms of overall survival, not just one endpoint "
            "number, so please plot the adjusted survival curves for chemo_"
            "group using time_to_event and event_indicator, adjusted for stage "
            "and age, oncology_survival.csv, appreciate it."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_03",
        "category": "noisy_verbose",
        "domain": "nephrology",
        "text": (
            "quick brain dump before the ask: dialysis_survival.csv came from "
            "three centers, slightly different follow-up windows which is "
            "annoying, but what I actually need is the adjusted survival curve "
            "comparing home vs in-center dialysis over time, adjusted for "
            "comorbidity_index, not a single hospitalization count, sorry for "
            "the long setup."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_04",
        "category": "noisy_verbose",
        "domain": "cardiology",
        "text": (
            "apologies for the wall of text, cardiology fellows keep changing "
            "the outcome definitions on me, but the current ask, cutting "
            "through the noise, is to plot adjusted event-free survival curves "
            "over time for each treatment arm in cardiac_outcomes.csv using "
            "time_to_mace and mace_event, adjusted for baseline ejection "
            "fraction and age, thanks for bearing with me."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_05",
        "category": "noisy_verbose",
        "domain": "transplant",
        "text": (
            "sorry, this is going to be long, the transplant registry had a "
            "data migration last quarter so some columns got renamed, "
            "hopefully graft_survival.csv is clean now, what I need is the "
            "graft survival curve over time for each induction protocol, "
            "adjusted for donor type and HLA mismatch, not just a one-year "
            "failure rate, thanks for reading through all this."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_06",
        "category": "noisy_verbose",
        "domain": "infectious_disease",
        "text": (
            "not sure how much detail you need so I'll just include "
            "everything, hiv_progression.csv has patients from two clinics "
            "with slightly different visit schedules, but the core request is "
            "the adjusted AIDS-free survival curve over time by antiviral "
            "regimen, adjusted for baseline CD4 count, apologies for the "
            "extra context."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_07",
        "category": "noisy_verbose",
        "domain": "respiratory",
        "text": (
            "going to ramble a little, the pulmonology team changed the "
            "exacerbation definition mid-study which is a headache, but "
            "putting that aside, I need the adjusted exacerbation-free "
            "survival curve over time for each inhaler type in copd_"
            "exacerbation.csv, adjusted for baseline FEV1 and smoking pack "
            "years, thanks for working through this long message."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_08",
        "category": "noisy_verbose",
        "domain": "oncology",
        "text": (
            "sorry, long one, the relapse registry merged with another "
            "database last year so there might be a few duplicate patient IDs "
            "we're still chasing down, but regardless, the ask is the adjusted "
            "relapse-free survival curve over time by adjuvant therapy arm in "
            "relapse_registry.csv, adjusted for stage and biomarker level, "
            "appreciate the patience."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_09",
        "category": "noisy_verbose",
        "domain": "geriatrics",
        "text": (
            "apologies for the length, the nursing home study had a rocky "
            "start with inconsistent frailty scoring across sites which we "
            "later standardized, but what I need now is the adjusted "
            "readmission-free survival curve over time for each care model in "
            "nursing_home_study.csv, adjusted for baseline frailty score, "
            "thanks for reading through the context."
        ),
    },
    {
        "id": "sup_survival_noisy_verbose_10",
        "category": "noisy_verbose",
        "domain": "hematology",
        "text": (
            "sorry for the long note, the leukemia trial had a protocol "
            "amendment partway through that changed the induction dosing "
            "slightly, shouldn't matter for this request though, I need the "
            "adjusted relapse-or-death-free survival curve over time by "
            "induction regimen in leukemia_trial.csv, adjusted for "
            "cytogenetic risk and age, thanks for bearing with the rambling."
        ),
    },
    # --- near_boundary (10) ---------------------------------------------------
    # Boundary vs causal_ate: mentions "treatment effect" language but
    # explicitly requests the curve/trajectory over time, not a single
    # summary number, keeping it unambiguously survival_adjusted_curves per
    # its own description ("Do NOT use this tool for causal effect
    # estimation, even if the outcome is survival or time-to-event" cuts the
    # other way -- so these prompts must clearly ask for curves/plots).
    {
        "id": "sup_survival_near_boundary_01",
        "category": "near_boundary",
        "domain": "oncology",
        "text": (
            "I know you could just give me one treatment-effect number, but I "
            "specifically want to see the adjusted survival curves over time "
            "for chemo_group in oncology_survival.csv, not a single average "
            "effect, adjusted for stage and age."
        ),
    },
    {
        "id": "sup_survival_near_boundary_02",
        "category": "near_boundary",
        "domain": "cardiology",
        "text": (
            "Rather than a single average treatment effect, plot the adjusted "
            "event-free survival curves over time for each treatment arm in "
            "cardiac_outcomes.csv, adjusted for baseline ejection fraction and "
            "age -- I need the full trajectory, not one number."
        ),
    },
    {
        "id": "sup_survival_near_boundary_03",
        "category": "near_boundary",
        "domain": "hepatology",
        "text": (
            "Don't collapse this to one summary treatment effect -- I need the "
            "adjusted survival curve shape over time comparing D-"
            "penicillamine to placebo in PBC_agent.csv, adjusted for age and "
            "edema."
        ),
    },
    {
        "id": "sup_survival_near_boundary_04",
        "category": "near_boundary",
        "domain": "nephrology",
        "text": (
            "I don't want the average difference at one time point, I want the "
            "adjusted survival curves plotted across the whole follow-up for "
            "home vs in-center dialysis in dialysis_survival.csv, adjusted for "
            "comorbidity_index."
        ),
    },
    {
        "id": "sup_survival_near_boundary_05",
        "category": "near_boundary",
        "domain": "transplant",
        "text": (
            "Skip the single summary effect -- plot the adjusted graft-"
            "survival curves over time for each induction protocol in "
            "graft_survival.csv, adjusted for donor type and HLA mismatch, so "
            "I can see how the groups diverge."
        ),
    },
    {
        "id": "sup_survival_near_boundary_06",
        "category": "near_boundary",
        "domain": "infectious_disease",
        "text": (
            "Instead of one overall treatment effect number, generate the "
            "adjusted AIDS-free survival curves over time by antiviral "
            "regimen in hiv_progression.csv, adjusted for baseline CD4 count."
        ),
    },
    {
        "id": "sup_survival_near_boundary_07",
        "category": "near_boundary",
        "domain": "respiratory",
        "text": (
            "I need to see the shape of the adjusted exacerbation-free "
            "survival curves over time by inhaler type in copd_"
            "exacerbation.csv, adjusted for baseline FEV1 and smoking pack "
            "years -- a single treatment-effect estimate won't show that."
        ),
    },
    {
        "id": "sup_survival_near_boundary_08",
        "category": "near_boundary",
        "domain": "oncology",
        "text": (
            "Please don't reduce this to one average effect -- plot the "
            "adjusted relapse-free survival curves over time by adjuvant "
            "therapy arm in relapse_registry.csv, adjusted for stage and "
            "biomarker level, so I can see where the curves cross."
        ),
    },
    {
        "id": "sup_survival_near_boundary_09",
        "category": "near_boundary",
        "domain": "geriatrics",
        "text": (
            "I specifically need the trajectory, not a summary number -- plot "
            "adjusted readmission-free survival curves over time for each "
            "care model in nursing_home_study.csv, adjusted for baseline "
            "frailty score."
        ),
    },
    {
        "id": "sup_survival_near_boundary_10",
        "category": "near_boundary",
        "domain": "hematology",
        "text": (
            "Give me the adjusted relapse-or-death-free survival curves over "
            "time by induction regimen in leukemia_trial.csv, adjusted for "
            "cytogenetic risk and age -- I want to see the whole curve, not "
            "just a single effect size."
        ),
    },
]

assert len(SURVIVAL_PROMPTS) == 60

ALL_SUPPORTED_PROMPTS = [
    {**p, "capability": "causal_ate", "gold_capability_id": "causal_ate"}
    for p in CAUSAL_ATE_PROMPTS
] + [
    {**p, "capability": "survival_adjusted_curves", "gold_capability_id": "survival_adjusted_curves"}
    for p in SURVIVAL_PROMPTS
]

assert len(ALL_SUPPORTED_PROMPTS) == 120
