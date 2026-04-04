"""Service model for microservice control experiment.

Defines the RequestState dataclass and Decision enum used across all
controllers and scenarios in the microservice_control experiment.

Each RequestState snapshot represents one request cycle in a distributed
API / microservice system. The fields model the three orthogonal failure
geometries under study:

    fanout / depth / total_requests  — structural expansion (RNOS axis)
    failures_last_n                  — failure density   (CB axis)
    latency_ms / latency_trend       — latency drift     (Persistence axis)
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
class RequestState:
    """Snapshot of distributed request execution context at a given step.

    Fields
    ------
    step : int
        1-indexed execution step.
    fanout : int
        Number of downstream service calls spawned by this request.
        Grows exponentially in fanout_cascade; stable in other scenarios.
    depth : int
        Call-chain depth (number of service hops). Grows with fanout in
        cascading scenarios; stable otherwise.
    total_requests : int
        Cumulative requests spawned across all steps (non-resetting).
        Includes retry re-transmissions in retry_storm.
    failures_last_n : int
        Count of failures in the last N steps (window tracked externally,
        for display only — the CB controller maintains its own deque).
    latency_ms : float
        Observed end-to-end latency for this request cycle (ms).
    latency_trend : float
        Change in latency_ms from the previous step (ms/step).
        Positive and sustained → latency drift detected by Persistence.
    success : bool
        Whether this request cycle completed successfully.
    """

    step: int
    fanout: int
    depth: int
    total_requests: int
    failures_last_n: int
    latency_ms: float
    latency_trend: float
    success: bool
