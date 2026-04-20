from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import BasePlotter


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_ate_from_stdout(stdout: str) -> Dict[str, float]:
    """
    Parse causalmodels stdout lines like:
    1 0.5042728 0.03225413 0.4410559 0.5674897
    """
    result: Dict[str, float] = {}
    try:
        for line in stdout.splitlines():
            s = line.strip()
            if re.match(r"^1\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+$", s):
                parts = s.split()
                if len(parts) >= 5:
                    result["ate"] = float(parts[1])
                    result["ci_lower"] = float(parts[3])
                    result["ci_upper"] = float(parts[4])
                    return result
    except Exception:
        pass
    return result


class CausalATEPlotter(BasePlotter):
    @property
    def capability_id(self) -> str:
        return "causal_ate"

    def can_plot(
        self,
        artifacts: Dict[str, Any],
        stdout: str = "",
        stderr: str = "",
    ) -> bool:
        if _safe_float(artifacts.get("ate")) is not None:
            return True
        parsed = _parse_ate_from_stdout(stdout)
        return _safe_float(parsed.get("ate")) is not None

    def generate(
        self,
        artifacts: Dict[str, Any],
        out_dir: Path,
        stdout: str = "",
        stderr: str = "",
    ) -> List[str]:
        out_dir.mkdir(parents=True, exist_ok=True)

        ate = _safe_float(artifacts.get("ate"))
        lo = _safe_float(artifacts.get("ci_lower"))
        hi = _safe_float(artifacts.get("ci_upper"))

        if ate is None:
            parsed = _parse_ate_from_stdout(stdout)
            ate = _safe_float(parsed.get("ate"))
            lo = _safe_float(parsed.get("ci_lower"))
            hi = _safe_float(parsed.get("ci_upper"))
            if parsed:
                artifacts.update(parsed)

        if ate is None:
            return []

        out_path = out_dir / "effect_plot.png"

        fig, ax = plt.subplots(figsize=(6.4, 2.4))

        if lo is not None and hi is not None:
            ax.errorbar([ate], [0], xerr=[[ate - lo], [hi - ate]], fmt="o", capsize=4)
            xmin = min(lo, 0.0)
            xmax = max(hi, 0.0)
        else:
            ax.scatter([ate], [0])
            xmin = min(ate, 0.0)
            xmax = max(ate, 0.0)

        pad = max((xmax - xmin) * 0.15, 0.1)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_xlabel("Effect size")
        ax.set_title("Estimated treatment effect")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        return [str(out_path)]