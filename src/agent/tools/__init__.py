# src/agent/tools/__init__.py
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Type

from .base import BaseTool
from .registry import register


def _iter_tool_module_names() -> list[str]:
    """
    Auto-scan THIS package (src.agent.tools) and return module names that match tool_*.py
    """
    pkg_name = __name__  # "src.agent.tools"
    module_names: list[str] = []
    for m in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if m.ispkg:
            continue
        if m.name.startswith("tool_"):
            module_names.append(f"{pkg_name}.{m.name}")
    return sorted(module_names)


def _register_tools_from_module(mod) -> None:
    """
    Find BaseTool subclasses in a module and register them.
    Assumes each tool class can be instantiated with no args.
    """
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        # Only classes defined in this module (avoid imported aliases)
        if obj.__module__ != mod.__name__:
            continue
        # Must be BaseTool subclass
        if not issubclass(obj, BaseTool) or obj is BaseTool:
            continue
        # Skip abstract classes
        if inspect.isabstract(obj):
            continue

        tool_cls: Type[BaseTool] = obj  # type: ignore[assignment]

        try:
            tool = tool_cls()  # MUST be no-arg init
        except TypeError:
            # If you have a tool that requires init args, you can either:
            # 1) give it defaults, or
            # 2) manually register it elsewhere.
            continue

        try:
            register(tool)
        except Exception:
            # Duplicate capability_id or other registry error -> ignore to keep import safe
            continue


def autodiscover_and_register_tools() -> None:
    """
    Import all tool_*.py modules and auto-register all BaseTool subclasses.
    """
    for mod_name in _iter_tool_module_names():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # If a module fails to import, skip it (you can check logs when debugging).
            continue
        _register_tools_from_module(mod)


# Run on import so `import src.agent.tools` triggers registration.
autodiscover_and_register_tools()
