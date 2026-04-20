from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class BasePlotter(ABC):
    @property
    @abstractmethod
    def capability_id(self) -> str:
        ...

    @abstractmethod
    def can_plot(
        self,
        artifacts: Dict[str, Any],
        stdout: str = "",
        stderr: str = "",
    ) -> bool:
        ...

    @abstractmethod
    def generate(
        self,
        artifacts: Dict[str, Any],
        out_dir: Path,
        stdout: str = "",
        stderr: str = "",
    ) -> List[str]:
        """
        Return a list of generated figure file paths.
        """
        ...