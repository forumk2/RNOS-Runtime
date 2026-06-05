"""Scenario tool library for multi-axis evaluation.

Four deterministic API simulators sharing the same ``run()`` interface.
Each takes a seed in its constructor — no module-level RNG state.
"""

from __future__ import annotations

import random

from .base import ToolResult


class ScenarioResult(ToolResult):
    """ToolResult with backward-compatible .success / .result_data aliases."""

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def result_data(self) -> dict:
        return self.data


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------


class CascadeAPI:
    """Formalizes the UnstableAPI cascade pattern.

    Phases:
      - calls 1-2  : always succeed  (stable)
      - calls 3-5  : succeed with p=0.5  (unstable)
      - calls 6+   : always fail  (collapse)
    """

    name = "cascade_api"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.call_count = 0
        self.failure_streak = 0

    def run(self, resource: str = "/status", **_: object) -> ScenarioResult:
        self.call_count += 1

        if self.call_count <= 2:
            self.failure_streak = 0
            return ScenarioResult(
                ok=True,
                message="API call succeeded",
                data={
                    "status": 200,
                    "phase": "stable",
                    "call_count": self.call_count,
                    "failure_streak": 0,
                    "resource": resource,
                },
            )

        if self.call_count <= 5:
            success = self._rng.random() > 0.5
            if success:
                self.failure_streak = 0
                return ScenarioResult(
                    ok=True,
                    message="API call succeeded",
                    data={
                        "status": 200,
                        "phase": "unstable",
                        "call_count": self.call_count,
                        "failure_streak": 0,
                        "resource": resource,
                    },
                )
            self.failure_streak += 1
            return ScenarioResult(
                ok=False,
                message="transient_failure",
                data={
                    "error": "transient_failure",
                    "phase": "unstable",
                    "call_count": self.call_count,
                    "failure_streak": self.failure_streak,
                    "resource": resource,
                },
            )

        self.failure_streak += 1
        return ScenarioResult(
            ok=False,
            message="cascading_failure",
            data={
                "error": "cascading_failure",
                "phase": "collapse",
                "call_count": self.call_count,
                "failure_streak": self.failure_streak,
                "resource": resource,
            },
        )


class FlakyAPI:
    """Independent per-call failure probability=0.3.  No phase structure.

    Tests over-reaction: RNOS should tolerate background noise without refusing.
    """

    name = "flaky_api"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.call_count = 0
        self.failure_streak = 0

    def run(self, resource: str = "/status", **_: object) -> ScenarioResult:
        self.call_count += 1
        success = self._rng.random() > 0.3
        if success:
            self.failure_streak = 0
            return ScenarioResult(
                ok=True,
                message="API call succeeded",
                data={
                    "status": 200,
                    "phase": "flaky",
                    "call_count": self.call_count,
                    "failure_streak": 0,
                    "resource": resource,
                },
            )
        self.failure_streak += 1
        return ScenarioResult(
            ok=False,
            message="random_failure",
            data={
                "error": "random_failure",
                "phase": "flaky",
                "call_count": self.call_count,
                "failure_streak": self.failure_streak,
                "resource": resource,
            },
        )


class RecoveringAPI:
    """High failure rate (0.8) for the first N=4 calls, then 0.05 thereafter.

    Tests false-refusal: a patient agent should succeed after the early turbulence.
    RNOS fires here = false refusal.
    """

    name = "recovering_api"

    def __init__(self, seed: int, n_high: int = 4) -> None:
        self._rng = random.Random(seed)
        self.call_count = 0
        self.failure_streak = 0
        self._n_high = n_high

    def run(self, resource: str = "/status", **_: object) -> ScenarioResult:
        self.call_count += 1
        in_high_phase = self.call_count <= self._n_high
        fail_rate = 0.8 if in_high_phase else 0.05
        phase = "recovering_early" if in_high_phase else "recovering_late"
        success = self._rng.random() > fail_rate
        if success:
            self.failure_streak = 0
            return ScenarioResult(
                ok=True,
                message="API call succeeded",
                data={
                    "status": 200,
                    "phase": phase,
                    "call_count": self.call_count,
                    "failure_streak": 0,
                    "resource": resource,
                },
            )
        self.failure_streak += 1
        return ScenarioResult(
            ok=False,
            message="transient_failure",
            data={
                "error": "transient_failure",
                "phase": phase,
                "call_count": self.call_count,
                "failure_streak": self.failure_streak,
                "resource": resource,
            },
        )


class StableAPI:
    """0.02 background failure rate.  Control scenario.

    RNOS should never refuse this.  Any refusal = false refusal.
    """

    name = "stable_api"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.call_count = 0
        self.failure_streak = 0

    def run(self, resource: str = "/status", **_: object) -> ScenarioResult:
        self.call_count += 1
        success = self._rng.random() > 0.02
        if success:
            self.failure_streak = 0
            return ScenarioResult(
                ok=True,
                message="API call succeeded",
                data={
                    "status": 200,
                    "phase": "stable",
                    "call_count": self.call_count,
                    "failure_streak": 0,
                    "resource": resource,
                },
            )
        self.failure_streak += 1
        return ScenarioResult(
            ok=False,
            message="background_failure",
            data={
                "error": "background_failure",
                "phase": "stable",
                "call_count": self.call_count,
                "failure_streak": self.failure_streak,
                "resource": resource,
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, type] = {
    "cascade": CascadeAPI,
    "flaky": FlakyAPI,
    "recovering": RecoveringAPI,
    "stable": StableAPI,
}
