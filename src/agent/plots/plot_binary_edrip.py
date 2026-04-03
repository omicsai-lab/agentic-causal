from __future__ import annotations

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


class BinaryEDRIPPlotter(BasePlotter):
    @property
    def capability_id(self) -> str:
        return "binary_edrip"

    def can_plot(
        self,
        artifacts: Dict[str, Any],
        stdout: str = "",
        stderr: str = "",
    ) -> bool:
        return _safe_float(artifacts.get("ate")) is not None

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
        ax.set_title("Binary outcome treatment effect")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        return [str(out_path)]