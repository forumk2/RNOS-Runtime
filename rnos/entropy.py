"""Entropy calculations for execution instability."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ActionRecord


def calculate_entropy(history: Sequence[ActionRecord], candidate: ActionRecord) -> float:
    """Return a simple bounded instability score."""

    depth_score = min(candidate.depth * 0.75, 4.0)
    retry_score = min(candidate.retry_count * 1.25, 4.0)

    recent_failures = sum(1 for item in history[-5:] if item.success is False)
    failure_score = min(recent_failures * 0.8, 3.0)

    repeated_tool = 0
    if history and history[-1].tool_name == candidate.tool_name:
        repeated_tool = 1
    if len(history) >= 2 and all(item.tool_name == candidate.tool_name for item in history[-2:]):
        repeated_tool = 2

    return round(depth_score + retry_score + failure_score + repeated_tool, 3)
