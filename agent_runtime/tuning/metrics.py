"""Metrics for adaptive recovery runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecoveryMetrics:
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    refusals: int = 0
    degradations: int = 0
    resolution_steps: list[int] = field(default_factory=list)
    refusal_steps: list[int] = field(default_factory=list)

    def record_degradation(self) -> None:
        self.degradations += 1
        self.recovery_attempts += 1

    def record_recovery_attempt(self) -> None:
        self.recovery_attempts += 1

    def record_recovery(self, step: int) -> None:
        self.successful_recoveries += 1
        self.resolution_steps.append(step)

    def record_refusal(self, step: int) -> None:
        self.refusals += 1
        self.refusal_steps.append(step)

    @property
    def average_steps_to_resolution(self) -> float:
        if not self.resolution_steps:
            return 0.0
        return sum(self.resolution_steps) / len(self.resolution_steps)

    @property
    def average_refusal_step(self) -> float:
        if not self.refusal_steps:
            return 0.0
        return sum(self.refusal_steps) / len(self.refusal_steps)
