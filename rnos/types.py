"""Shared data structures for the RNOS runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    """High-level runtime decisions."""

    ALLOW = "allow"
    DEGRADE = "degrade"
    REFUSE = "refuse"


@dataclass(slots=True)
class ActionRecord:
    """One unit of planned or executed agent behavior."""

    tool_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    retry_count: int = 0
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    cumulative_calls: int = 0


@dataclass(slots=True)
class RuntimeAssessment:
    """Computed runtime state before an action executes."""

    entropy: float
    trust: float
    decision: PolicyDecision
    reasons: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
