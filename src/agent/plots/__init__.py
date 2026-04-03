from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Type

from .base import BasePlotter
from .registry import register_plotter


def _iter_plot_module_names() -> list[str]:
    pkg_name = __name__  # "src.agent.plots"
    module_names: list[str] = []
    for m in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if m.ispkg:
            continue
        if m.name.startswith("plot_"):
            module_names.append(f"{pkg_name}.{m.name}")
    return sorted(module_names)


def _register_plotters_from_module(mod) -> None:
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if not issubclass(obj, BasePlotter) or obj is BasePlotter:
            continue
        if inspect.isabstract(obj):
            continue

        plotter_cls: Type[BasePlotter] = obj
        try:
            plotter = plotter_cls()
            register_plotter(plotter)
        except Exception:
            continue


def autodiscover_and_register_plotters() -> None:
    for mod_name in _iter_plot_module_names():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        _register_plotters_from_module(mod)


autodiscover_and_register_plotters()