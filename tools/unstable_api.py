"""Failure-prone tool for runtime testing."""

from __future__ import annotations

import random

from .base import ToolResult


class UnstableAPI:
    """Stateful unstable API simulation with worsening behavior over time."""

    def __init__(self) -> None:
        self.call_count = 0
        self.failure_streak = 0

    def call(self) -> tuple[bool, dict]:
        self.call_count += 1

        if self.call_count <= 2:
            self.failure_streak = 0
            return True, {"status": 200, "phase": "stable"}

        if self.call_count <= 5:
            success = random.random() > 0.4
            if success:
                self.failure_streak = 0
                return True, {"status": 200, "phase": "unstable"}

            self.failure_streak += 1
            return False, {"error": "transient_failure", "phase": "unstable"}

        self.failure_streak += 1
        return False, {"error": "cascading_failure", "phase": "collapse"}


class UnstableAPIResult(ToolResult):
    """Compatibility wrapper that also exposes success/result_data fields."""

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def result_data(self) -> dict:
        return self.data


class UnstableAPITool:
    name = "unstable_api"

    def __init__(self, api: UnstableAPI | None = None) -> None:
        self.api = api or UnstableAPI()

    def run(self, resource: str = "/status", **_: object) -> UnstableAPIResult:
        success, result_data = self.api.call()
        result_data = {
            "resource": resource,
            "call_count": self.api.call_count,
            "failure_streak": self.api.failure_streak,
            **result_data,
        }
        message = "API call succeeded" if success else result_data.get("error", "API call failed")
        return UnstableAPIResult(ok=success, message=message, data=result_data)
