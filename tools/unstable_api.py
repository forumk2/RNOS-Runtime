"""Failure-prone tool for runtime testing."""

from __future__ import annotations

import random

from .base import ToolResult


class UnstableAPITool:
    name = "unstable_api"

    def __init__(self, failure_rate: float = 0.5) -> None:
        self.failure_rate = failure_rate

    def run(self, resource: str = "/status", **_: object) -> ToolResult:
        if random.random() < self.failure_rate:
            return ToolResult(ok=False, message="Transient upstream failure", data={"resource": resource})
        return ToolResult(ok=True, message="API call succeeded", data={"resource": resource, "status": 200})
