"""Pipeline model for CI control experiment.

Defines the PipelineState dataclass and Decision enum used across all
controllers and scenarios in the ci_control experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    ALLOW = "allow"
    DEGRADE = "degrade"
    REFUSE = "refuse"

    def __str__(self) -> str:
        return self.value.upper()


@dataclass
class PipelineState:
    """Snapshot of pipeline execution context at a given step.

    Each step represents one pipeline tick (job execution / scheduling).

    Fields
    ------
    step : int
        1-indexed execution step.
    active_jobs : int
        Number of concurrently running jobs this tick. Captures fanout pressure.
    total_jobs_spawned : int
        Cumulative total of all jobs spawned so far (non-resetting).
        Represents structural expansion of the pipeline.
    retry_count : int
        Cumulative number of retries issued so far. Reflects structural
        instability driven by repeated re-execution.
    success : bool
        Whether the job(s) in this tick succeeded.
    """

    step: int
    active_jobs: int
    total_jobs_spawned: int
    retry_count: int
    success: bool
