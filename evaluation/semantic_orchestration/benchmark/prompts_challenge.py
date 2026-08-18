"""
Literal source-of-truth data for the 30 challenge benchmark prompts.

Unlike the supported set, no challenge prompt is assigned a single
`gold_capability_id`, by design: for `underspecified` prompts, more than
one of the 7 registered capabilities would be a defensible pick given the
missing information; for `unsupported_method` and `out_of_scope_mixed`
prompts, none of the 7 registered capabilities correctly answers the
request. Forcing a specific gold capability_id onto any of these would
itself be an ambiguous/incorrect gold label, which the evaluation brief
explicitly prohibits. Instead each entry carries `expected_behavior`
(free text) describing what a well-behaved router/planner should do, and
the analysis code reports the *distribution* of router picks per
challenge_type rather than a pass/fail accuracy number.

See EVALUATION_PROTOCOL.md, "Challenge set scoring", for how these are
analyzed.
"""

from __future__ import annotations

CHALLENGE_TYPES = [
    "underspecified",
    "unsupported_method",
    "out_of_scope_mixed",
]

# -------------------------------------------------------------------------
# underspecified (10): plausible in-scope biomedical requests that omit
# information needed to tell causal_ate from survival_adjusted_curves (or
# from another registered capability) -- i.e. more than one capability_id
# is a defensible router choice given the text alone.
# -------------------------------------------------------------------------
UNDERSPECIFIED_PROMPTS = [
    {
        "id": "chal_underspecified_01",
        "domain": "oncology",
        "text": "Can you analyze the effect of the treatment on outcomes in my dataset?",
        "expected_behavior": (
            "No single capability_id is correct: causal_ate, "
            "survival_adjusted_curves, binary_edrip, linear_regression, and "
            "logistic_regression are all defensible without knowing the "
            "outcome type, time structure, or requested output format."
        ),
    },
    {
        "id": "chal_underspecified_02",
        "domain": "cardiology",
        "text": "study.csv has treatment and outcome columns, tell me if the treatment worked.",
        "expected_behavior": (
            "'Worked' is undefined: could mean a single ATE, an adjusted "
            "survival comparison, or a regression association; column names "
            "give no outcome-type or time-structure signal."
        ),
    },
    {
        "id": "chal_underspecified_03",
        "domain": "general",
        "text": "I have a clinical trial dataset, what's the impact of the drug?",
        "expected_behavior": (
            "No outcome type, time structure, or requested output (single "
            "estimate vs. curve) is given, so causal_ate and "
            "survival_adjusted_curves are equally plausible."
        ),
    },
    {
        "id": "chal_underspecified_04",
        "domain": "nephrology",
        "text": "Compare the two treatment groups in dialysis_data.csv over the study period.",
        "expected_behavior": (
            "'Over the study period' hints at survival, but 'compare the two "
            "groups' without an outcome or event definition could equally "
            "resolve to causal_ate; no time/event/group fields are named."
        ),
    },
    {
        "id": "chal_underspecified_05",
        "domain": "general",
        "text": "Run the causal analysis on my data please.",
        "expected_behavior": (
            "'Causal analysis' matches causal_ate, survival_adjusted_curves, "
            "and binary_edrip descriptions equally; no capability_id is "
            "uniquely correct."
        ),
    },
    {
        "id": "chal_underspecified_06",
        "domain": "diabetes",
        "text": "Does drug X make a difference for patients versus not taking it, using glucose_study.csv?",
        "expected_behavior": (
            "No outcome variable, time structure, or requested output form "
            "is specified, leaving causal_ate and survival_adjusted_curves "
            "both defensible."
        ),
    },
    {
        "id": "chal_underspecified_07",
        "domain": "general",
        "text": "Adjust for confounding and tell me what the treatment group data shows.",
        "expected_behavior": (
            "'Adjust for confounding' matches every registered causal "
            "capability (causal_ate, survival_adjusted_curves, "
            "binary_edrip); no distinguishing outcome or time information "
            "is present."
        ),
    },
    {
        "id": "chal_underspecified_08",
        "domain": "oncology",
        "text": "Patients either got the new therapy or didn't, cancer_outcomes.csv, what's the difference between groups?",
        "expected_behavior": (
            "'Cancer outcomes' could be a binary response, a continuous "
            "tumor measurement, or a time-to-event endpoint; the request "
            "does not disambiguate causal_ate vs survival_adjusted_curves."
        ),
    },
    {
        "id": "chal_underspecified_09",
        "domain": "general",
        "text": "treatment_data.csv -- give me the causal effect estimate, adjusted properly.",
        "expected_behavior": (
            "'Causal effect estimate, adjusted properly' could mean an ATE, "
            "a binary_edrip-style doubly robust estimate, or (if the "
            "estimate is over time) a survival curve; no outcome or time "
            "field is named."
        ),
    },
    {
        "id": "chal_underspecified_10",
        "domain": "psychiatry",
        "text": "Did the intervention help in trial_data.csv? Please account for baseline differences between groups.",
        "expected_behavior": (
            "'Help' and 'account for baseline differences' are consistent "
            "with causal_ate, survival_adjusted_curves, or binary_edrip; no "
            "outcome type or time-to-event structure is specified."
        ),
    },
]

assert len(UNDERSPECIFIED_PROMPTS) == 10

# -------------------------------------------------------------------------
# unsupported_method (10): explicitly named causal-inference methods that
# none of the 7 registered capabilities implement (instrumental variables,
# regression discontinuity, difference-in-differences, synthetic control,
# mediation analysis, propensity score matching as a distinct algorithm,
# g-computation for time-varying treatment, target trial emulation,
# heterogeneous treatment effects / causal forests, front-door adjustment).
# Care is taken NOT to describe binary_edrip (doubly robust / semiparametric
# / EDRIP for a binary outcome), since that method IS implemented.
# -------------------------------------------------------------------------
UNSUPPORTED_METHOD_PROMPTS = [
    {
        "id": "chal_unsupported_method_01",
        "domain": "health_economics",
        "text": (
            "Use an instrumental variables approach with distance-to-clinic "
            "as the instrument to estimate the effect of statin uptake on "
            "cholesterol in access_study.csv."
        ),
        "expected_behavior": (
            "No registered capability implements instrumental-variables "
            "estimation; a correct system would decline or flag as "
            "unsupported rather than silently substituting causal_ate."
        ),
    },
    {
        "id": "chal_unsupported_method_02",
        "domain": "health_policy",
        "text": (
            "Run a regression discontinuity design around the age-65 "
            "Medicare eligibility cutoff to estimate the effect on hospital "
            "utilization in medicare_cutoff.csv."
        ),
        "expected_behavior": (
            "Regression discontinuity is not implemented by any registered "
            "capability; none of the 7 tools correctly answers this."
        ),
    },
    {
        "id": "chal_unsupported_method_03",
        "domain": "health_policy",
        "text": (
            "Estimate a difference-in-differences model for the effect of "
            "the new hospital policy on readmission rates, using pre/post "
            "and treated/control indicators in policy_panel.csv."
        ),
        "expected_behavior": (
            "Difference-in-differences panel estimation is not implemented; "
            "no registered capability correctly answers this."
        ),
    },
    {
        "id": "chal_unsupported_method_04",
        "domain": "health_policy",
        "text": (
            "Build a synthetic control for the state that adopted the new "
            "vaccination mandate and estimate its effect on case counts "
            "using state_panel.csv."
        ),
        "expected_behavior": (
            "Synthetic control methods are not implemented by any "
            "registered capability."
        ),
    },
    {
        "id": "chal_unsupported_method_05",
        "domain": "psychology",
        "text": (
            "Run a causal mediation analysis to decompose the effect of the "
            "intervention on outcome into the portion mediated by adherence "
            "versus the direct effect, using mediation_study.csv."
        ),
        "expected_behavior": (
            "Mediation analysis (natural direct/indirect effect "
            "decomposition) is not implemented by any registered "
            "capability."
        ),
    },
    {
        "id": "chal_unsupported_method_06",
        "domain": "oncology",
        "text": (
            "Fit a causal forest to estimate heterogeneous treatment effects "
            "of the therapy across patient subgroups in oncology_hte.csv."
        ),
        "expected_behavior": (
            "Heterogeneous treatment effect estimation via causal forests "
            "is not implemented; causal_ate only estimates a single average "
            "effect, not subgroup-varying effects."
        ),
    },
    {
        "id": "chal_unsupported_method_07",
        "domain": "epidemiology",
        "text": (
            "Emulate a target trial with a time-varying treatment using the "
            "parametric g-formula to estimate the effect of statin "
            "initiation timing on mortality in ehr_cohort.csv."
        ),
        "expected_behavior": (
            "Time-varying treatment g-computation / target trial emulation "
            "is not implemented; causal_ate only handles a single "
            "point-in-time treatment assignment."
        ),
    },
    {
        "id": "chal_unsupported_method_08",
        "domain": "epidemiology",
        "text": (
            "Use front-door adjustment to estimate the effect of exposure on "
            "outcome in confounded_study.csv, since we can't measure the "
            "confounders directly but can measure the mediator."
        ),
        "expected_behavior": (
            "Front-door adjustment is not implemented by any registered "
            "capability, all of which rely on backdoor/covariate "
            "adjustment."
        ),
    },
    {
        "id": "chal_unsupported_method_09",
        "domain": "epidemiology",
        "text": (
            "Match patients on propensity score using nearest-neighbor "
            "caliper matching (not weighting) and report the matched-sample "
            "average treatment effect for matching_study.csv."
        ),
        "expected_behavior": (
            "Nearest-neighbor caliper matching as a distinct estimation "
            "algorithm is not implemented; the registered causal_ate tool "
            "uses covariate adjustment/weighting, not matching."
        ),
    },
    {
        "id": "chal_unsupported_method_10",
        "domain": "epidemiology",
        "text": (
            "Estimate a marginal structural model with inverse probability "
            "of treatment AND censoring weights for a time-varying exposure "
            "on repeated_measures_cohort.csv."
        ),
        "expected_behavior": (
            "Time-varying-exposure marginal structural models with "
            "censoring weights are not implemented; the registered "
            "survival tool only supports a single fixed baseline group "
            "assignment."
        ),
    },
]

assert len(UNSUPPORTED_METHOD_PROMPTS) == 10

# -------------------------------------------------------------------------
# out_of_scope_mixed (10): requests entirely unrelated to any registered
# capability, or that mix an unrelated ask with an in-scope one.
# -------------------------------------------------------------------------
OUT_OF_SCOPE_MIXED_PROMPTS = [
    {
        "id": "chal_out_of_scope_mixed_01",
        "domain": "none",
        "text": "What's the weather going to be like in Boston this weekend?",
        "expected_behavior": (
            "Entirely unrelated to any registered capability; a correct "
            "system declines rather than forcing a capability_id."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_02",
        "domain": "none",
        "text": "Write a short poem about my dataset called flowers.csv.",
        "expected_behavior": (
            "Creative writing request, not a data-analysis request; no "
            "registered capability applies."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_03",
        "domain": "none",
        "text": "Can you translate the word 'treatment' into French and Spanish?",
        "expected_behavior": (
            "Translation request, unrelated to any registered capability."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_04",
        "domain": "mixed",
        "text": (
            "Please email my manager an update on the project status, and "
            "also fit a regression of outcome on treatment in study.csv."
        ),
        "expected_behavior": (
            "Mixed intent: the email-sending portion is out of scope for "
            "any registered capability, even though the regression portion "
            "alone would be in scope; a correct system should not silently "
            "drop the out-of-scope half."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_05",
        "domain": "none",
        "text": "What's a good recipe for banana bread?",
        "expected_behavior": (
            "Entirely unrelated to any registered capability."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_06",
        "domain": "mixed",
        "text": (
            "Book me a flight to the conference next month, and by the way "
            "can you also summarize outcomes.csv while you're at it?"
        ),
        "expected_behavior": (
            "Mixed intent: flight booking is entirely out of scope; only "
            "the summarize-outcomes.csv portion loosely maps to "
            "summary_stats, so no single capability_id correctly answers "
            "the whole request."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_07",
        "domain": "none",
        "text": "Review this Python function for bugs: def add(a, b): return a - b",
        "expected_behavior": (
            "General code review request, unrelated to any registered "
            "capability."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_08",
        "domain": "none",
        "text": "Who won the World Series in 1998?",
        "expected_behavior": (
            "General trivia question, unrelated to any registered "
            "capability."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_09",
        "domain": "mixed",
        "text": (
            "Can you set a reminder for my doctor's appointment tomorrow, "
            "and separately tell me the average treatment effect of the "
            "drug on outcome in study.csv?"
        ),
        "expected_behavior": (
            "Mixed intent: the reminder-setting portion is entirely out of "
            "scope even though the ATE portion alone would map to "
            "causal_ate; a correct system should not silently drop the "
            "out-of-scope half."
        ),
    },
    {
        "id": "chal_out_of_scope_mixed_10",
        "domain": "none",
        "text": "Give me investment advice on whether I should buy index funds or individual stocks.",
        "expected_behavior": (
            "Financial advice request, unrelated to any registered "
            "capability."
        ),
    },
]

assert len(OUT_OF_SCOPE_MIXED_PROMPTS) == 10

ALL_CHALLENGE_PROMPTS = (
    [{**p, "challenge_type": "underspecified"} for p in UNDERSPECIFIED_PROMPTS]
    + [{**p, "challenge_type": "unsupported_method"} for p in UNSUPPORTED_METHOD_PROMPTS]
    + [{**p, "challenge_type": "out_of_scope_mixed"} for p in OUT_OF_SCOPE_MIXED_PROMPTS]
)

assert len(ALL_CHALLENGE_PROMPTS) == 30
