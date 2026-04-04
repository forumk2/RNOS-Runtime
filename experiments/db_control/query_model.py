"""Query model for DB control experiment.

Defines the QueryState dataclass and Decision enum used across all
controllers and scenarios in the db_control experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(Enum):
    ALLOW = "allow"
    DEGRADE = "degrade"
    REFUSE = "refuse"

    def __str__(self) -> str:
        return self.value.upper()


@dataclass
class QueryState:
    """Snapshot of query execution context at a given step.

    Fields
    ------
    step : int
        1-indexed execution step.
    join_depth : int
        Number of JOIN levels in the query plan. Higher = more structural complexity.
    estimated_cost : float
        Query planner's estimated execution cost (arbitrary units, log-scaled by RNOS).
    lock_wait_ms : float
        Time spent waiting on row/table locks (ms). Elevated in contention scenarios.
    success : bool
        Whether this query execution succeeded.
    failures_last_n : list[bool]
        Sliding window of recent outcomes (True = failed). Managed externally.
    cumulative_cost : float
        Running total of estimated_cost across all steps. Non-resetting.
    """

    step: int
    join_depth: int
    estimated_cost: float
    lock_wait_ms: float
    success: bool
    failures_last_n: list[bool] = field(default_factory=list)
    cumulative_cost: float = 0.0
