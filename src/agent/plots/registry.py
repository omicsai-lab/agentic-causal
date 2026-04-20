from __future__ import annotations

from typing import Dict, List, Optional

_PLOTTERS: Dict[str, object] = {}


def register_plotter(plotter) -> None:
    _PLOTTERS[plotter.capability_id] = plotter


def get_plotter(capability_id: str):
    return _PLOTTERS.get(capability_id)


def list_plotters() -> List[str]:
    return list(_PLOTTERS.keys())