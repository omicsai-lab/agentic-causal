from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Reuse OpenAI client pattern from your router if available
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


PLANNER_SYSTEM_PROMPT = """
You are a statistical analysis planning assistant inside a causal analysis agent.

Your job is:
1. Read the user's natural language request.
2. Infer the intended statistical goal.
3. Produce a concise analysis plan BEFORE tool selection.
4. Recommend the single best tool from the available tools.

Return ONLY valid JSON.

The plan must be understandable to users and useful for routing.

Required JSON fields:
- analysis_goal: short string
- task_type: one of ["causal_effect", "survival", "regression", "descriptive", "unknown"]
- target_estimand: short string
- outcome_type: one of ["binary", "continuous", "time_to_event", "unknown"]
- recommended_tool: short string
- required_fields: list of strings
- optional_fields: list of strings
- assumptions: list of short strings
- reasoning: short string

Guidance:
- If the user asks about survival over time, adjusted survival curves, Kaplan-Meier-like comparison, or time-to-event outcome comparison, prefer "survival_adjusted_curves".
- If the user asks about average treatment effect, treatment effect, causal effect, or compare treated vs untreated causally, prefer "causal_ate".
- If the user asks for linear regression on a continuous outcome, prefer "linear_regression".
- If the user asks for logistic regression on a binary outcome, prefer "logistic_regression".
- If the user asks for summary / describe / overview / EDA, prefer "summary_stats".
- If uncertain, set task_type to "unknown" and recommended_tool to "summary_stats".

Field guidance:
- survival_adjusted_curves usually requires: group, time, event
- causal_ate usually requires: treatment, outcome
- linear_regression usually requires: outcome
- logistic_regression usually requires: outcome
- summary_stats usually requires no additional fields beyond csv

Keep the reasoning concise and statistical, not conversational.
""".strip()


def _fallback_plan(request: str) -> Dict[str, Any]:
    text = (request or "").lower()

    if any(x in text for x in ["survival", "kaplan", "time-to-event", "time to event", "hazard"]):
        return {
            "analysis_goal": "Compare survival patterns across groups",
            "task_type": "survival",
            "target_estimand": "Adjusted survival curves",
            "outcome_type": "time_to_event",
            "recommended_tool": "survival_adjusted_curves",
            "required_fields": ["group", "time", "event"],
            "optional_fields": ["covariates"],
            "assumptions": [
                "time is follow-up time",
                "event indicates whether the event occurred",
                "group defines comparison groups",
            ],
            "reasoning": "The request is about survival over time rather than a single average treatment effect.",
        }

    if any(x in text for x in ["causal effect", "treatment effect", "average treatment effect", "ate"]):
        return {
            "analysis_goal": "Estimate the causal effect of treatment on outcome",
            "task_type": "causal_effect",
            "target_estimand": "Average treatment effect",
            "outcome_type": "unknown",
            "recommended_tool": "causal_ate",
            "required_fields": ["treatment", "outcome"],
            "optional_fields": ["covariates"],
            "assumptions": [
                "treatment is well defined",
                "outcome is well defined",
                "no unmeasured confounding after adjustment",
            ],
            "reasoning": "The request explicitly asks for a causal or treatment effect.",
        }

    if "logistic regression" in text:
        return {
            "analysis_goal": "Model a binary outcome using logistic regression",
            "task_type": "regression",
            "target_estimand": "Regression coefficients / odds-based association",
            "outcome_type": "binary",
            "recommended_tool": "logistic_regression",
            "required_fields": ["outcome"],
            "optional_fields": ["covariates"],
            "assumptions": [
                "outcome is binary",
            ],
            "reasoning": "The request explicitly asks for logistic regression.",
        }

    if "linear regression" in text:
        return {
            "analysis_goal": "Model a continuous outcome using linear regression",
            "task_type": "regression",
            "target_estimand": "Regression coefficients / mean association",
            "outcome_type": "continuous",
            "recommended_tool": "linear_regression",
            "required_fields": ["outcome"],
            "optional_fields": ["covariates"],
            "assumptions": [
                "outcome is continuous",
            ],
            "reasoning": "The request explicitly asks for linear regression.",
        }

    if any(x in text for x in ["summary", "summarize", "describe", "overview", "eda"]):
        return {
            "analysis_goal": "Summarize the dataset",
            "task_type": "descriptive",
            "target_estimand": "Descriptive summary",
            "outcome_type": "unknown",
            "recommended_tool": "summary_stats",
            "required_fields": [],
            "optional_fields": [],
            "assumptions": [],
            "reasoning": "The request is descriptive rather than inferential.",
        }

    return {
        "analysis_goal": "Understand the user's statistical intent",
        "task_type": "unknown",
        "target_estimand": "Unknown",
        "outcome_type": "unknown",
        "recommended_tool": "summary_stats",
        "required_fields": [],
        "optional_fields": [],
        "assumptions": [],
        "reasoning": "The request is ambiguous, so the safest default is a descriptive summary.",
    }


def _normalize_plan(obj: Dict[str, Any]) -> Dict[str, Any]:
    def _as_list_of_str(x: Any) -> List[str]:
        if not isinstance(x, list):
            return []
        return [str(v).strip() for v in x if str(v).strip()]

    plan = {
        "analysis_goal": str(obj.get("analysis_goal", "")).strip(),
        "task_type": str(obj.get("task_type", "unknown")).strip() or "unknown",
        "target_estimand": str(obj.get("target_estimand", "")).strip(),
        "outcome_type": str(obj.get("outcome_type", "unknown")).strip() or "unknown",
        "recommended_tool": str(obj.get("recommended_tool", "")).strip(),
        "required_fields": _as_list_of_str(obj.get("required_fields", [])),
        "optional_fields": _as_list_of_str(obj.get("optional_fields", [])),
        "assumptions": _as_list_of_str(obj.get("assumptions", [])),
        "reasoning": str(obj.get("reasoning", "")).strip(),
    }

    if not plan["recommended_tool"]:
        plan["recommended_tool"] = "summary_stats"

    return plan


def llm_generate_analysis_plan(
    *,
    request: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a structured analysis plan from a natural-language request.

    Falls back to rule-based planning if OpenAI is unavailable or the call fails.
    """
    if not request or not request.strip():
        return _fallback_plan("")

    if OpenAI is None:
        return _fallback_plan(request)

    try:
        client = OpenAI()

        completion = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
        )

        text = completion.choices[0].message.content
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return _fallback_plan(request)

        return _normalize_plan(obj)

    except Exception:
        return _fallback_plan(request)