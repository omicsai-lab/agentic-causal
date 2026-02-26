from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _cap_dir() -> Path:
    # src/agent/router_llm.py -> src/agent/capabilities
    return Path(__file__).resolve().parent / "capabilities"


def load_capabilities() -> List[Dict[str, Any]]:
    cap_dir = _cap_dir()
    caps: List[Dict[str, Any]] = []

    for p in sorted(cap_dir.glob("cap_*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                caps.append(obj)
        except Exception as e:
            raise RuntimeError(f"Bad capability JSON: {p}: {e}") from e

    if not caps:
        raise RuntimeError(f"No capability JSON files found under {cap_dir} (cap_*.json).")

    return caps


def _capability_ids(caps: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for c in caps:
        cid = c.get("capability_id") or c.get("id")  # support both
        if isinstance(cid, str) and cid.strip():
            out.append(cid.strip())
    # de-dup preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def llm_choose_capability(
    *,
    request: str,
    csv_columns: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return ONLY:
      {"capability_id": "...", "reason": "..."}.

    Loads allowed IDs from src/agent/capabilities/cap_*.json.
    """
    caps = load_capabilities()
    allowed = _capability_ids(caps)
    if not allowed:
        raise RuntimeError("No capability_id found in cap_*.json files.")

    fallback_id = allowed[0]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"capability_id": fallback_id, "reason": "OPENAI_API_KEY not set; defaulting to first capability."}

    if OpenAI is None:
        return {"capability_id": fallback_id, "reason": "openai package not available; defaulting to first capability."}

    cols_text = ""
    if csv_columns:
        cols_text = "\nCSV columns:\n- " + "\n- ".join(map(str, csv_columns))

    system = (
        "You are a strict router. Choose exactly ONE capability_id from the allowed list. "
        "Return valid JSON with keys: capability_id, reason. No extra keys."
    )

    # Provide richer descriptions to help LLM choose correctly 
    cap_lines = []
    for c in caps:
        cid = c.get("capability_id") or c.get("id")
        title = c.get("title") or c.get("name") or ""
        desc = c.get("description") or ""
        if cid:
            cap_lines.append(f"- {cid}: {title} {desc}".strip())

    user = (
        "Allowed capability_id values:\n"
        + str(allowed)
        + "\n\nCapabilities:\n"
        + "\n".join(cap_lines)
        + "\n\nUser request:\n"
        + request
        + cols_text
        + "\n\nReturn JSON only."
    )

    client = OpenAI(api_key=api_key)
    use_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    content = ""
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()

    # Parse robustly 
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return {"capability_id": fallback_id, "reason": "LLM returned non-JSON; defaulting to first capability."}

    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"capability_id": fallback_id, "reason": "LLM returned invalid JSON; defaulting to first capability."}

    cap_id = str(obj.get("capability_id", "")).strip()
    reason = str(obj.get("reason", "")).strip() or "No reason provided."

    if cap_id not in allowed:
        return {
            "capability_id": fallback_id,
            "reason": f"LLM chose invalid capability_id='{cap_id}'; defaulting to '{fallback_id}'.",
        }

    return {"capability_id": cap_id, "reason": reason}
