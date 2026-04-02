"""Entropy calculations for execution instability."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ActionRecord


def calculate_entropy(history: Sequence[ActionRecord], candidate: ActionRecord) -> float:
    """Return a bounded instability score combining structural and runtime signals.

    Signal weights and maximum contributions:

    ┌──────────────────┬───────────────────┬──────────┐
    │ Signal           │ Weight            │ Max      │
    ├──────────────────┼───────────────────┼──────────┤
    │ depth_score      │ 0.6 per step      │ 4.0      │
    │ retry_score      │ 1.0 per retry     │ 4.0      │
    │ failure_score    │ 0.65 per failure  │ 3.0      │
    │ repeated_tool    │ 0 / 1 / 2         │ 2.0      │
    │ latency_score    │ 0.5 per second    │ 2.0      │
    │ cost_score       │ 0.3 per call      │ 2.0      │
    └──────────────────┴───────────────────┴──────────┘

    Weights were tuned so that the latency and cost signals occupy the head-room
    vacated by scaling down the structural weights (depth: 0.75→0.6,
    retry: 1.25→1.0, failure: 0.8→0.65), keeping DEGRADE/REFUSE thresholds
    at 3.0 / 6.0 respectively.
    """

    depth_score = min(candidate.depth * 0.6, 4.0)
    retry_score = min(candidate.retry_count * 1.0, 4.0)

    recent_failures = sum(1 for item in history[-5:] if item.success is False)
    failure_score = min(recent_failures * 0.65, 3.0)

    repeated_tool = 0
    if history and history[-1].tool_name == candidate.tool_name:
        repeated_tool = 1
    if len(history) >= 2 and all(item.tool_name == candidate.tool_name for item in history[-2:]):
        repeated_tool = 2

    # Latency signal: slow planner responses indicate local model stress.
    latency_score = 0.0
    if candidate.latency_ms is not None:
        latency_score = min((candidate.latency_ms / 1000.0) * 0.5, 2.0)

    # Cost signal: pressure accumulates with total work done, not just local retries.
    cost_score = min(candidate.cumulative_calls * 0.3, 2.0)

    return round(
        depth_score + retry_score + failure_score + repeated_tool + latency_score + cost_score,
        3,
    )
