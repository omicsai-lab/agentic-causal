# src/agent/tools/tool_hello_world.py
from __future__ import annotations

from typing import Tuple

from .base import BaseTool
from src.agent.schemas_io import RunRequest


class HelloWorldTool(BaseTool):
    @property
    def name(self) -> str:
        return "hello_world"

    @property
    def capability_id(self) -> str:
        return "hello_world"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        return True, ""

    def run(self, req: RunRequest):
        return {
            "status": "ok",
            "stdout": "Hello World 🎉",
            "stderr": "",
            "artifacts": {"message": "Hello world tool executed successfully."},
            "error": None,
        }