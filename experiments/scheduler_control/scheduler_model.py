"""Scheduler model for the job scheduler control experiment.

Defines SchedulerState and Decision used across all controllers and
scenarios in the scheduler_control experiment.

Each SchedulerState snapshot represents one scheduling cycle in a
Slurm-style batch job system. The fields model three orthogonal failure
geometries under study:

    active_jobs / total_jobs_spawned / dependency_depth
                          — structural expansion   (RNOS axis)
    failures_last_n       — failure density         (CB axis)
    queue_wait_time / wait_time_trend
                          — queue backlog drift     (Persistence axis)

This is NOT a real scheduler. It is a controlled simulation designed
to test whether tri-modal control (RNOS + CB + Persistence) generalises
to job scheduling systems.
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
class SchedulerState:
    """Snapshot of a batch job scheduler at a given scheduling cycle.

    Fields
    ------
    step : int
        1-indexed scheduling cycle number.
    active_jobs : int
        Number of jobs currently executing (in-flight). Grows rapidly in
        dependency_explosion; stable in other scenarios.
    queued_jobs : int
        Number of jobs waiting in the submission queue. Accumulates slowly
        in queue_backlog_drift; stable or low in other scenarios.
    total_jobs_spawned : int
        Cumulative jobs submitted across all cycles (non-resetting).
        Includes retry resubmissions in failing_jobs_storm.
    dependency_depth : int
        Maximum depth of the job dependency graph. Increments with each
        cycle in dependency_explosion; stable otherwise.
    failures_last_n : int
        Count of failures observed in the last N cycles (tracked externally
        for display; the CB controller maintains its own deque).
    queue_wait_time : float
        Average queue wait time (in time units) observed this cycle.
        Increases monotonically in queue_backlog_drift.
    wait_time_trend : float
        Change in queue_wait_time from the previous cycle (units/step).
        Sustained positive trend → queue saturation detected by Persistence.
    success : bool
        Whether the jobs dispatched this cycle completed without failure.
    """

    step: int
    active_jobs: int
    queued_jobs: int
    total_jobs_spawned: int
    dependency_depth: int
    failures_last_n: int
    queue_wait_time: float
    wait_time_trend: float
    success: bool
