"""Entropy calculations for execution instability."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ActionRecord


def calculate_entropy(history: Sequence[ActionRecord], candidate: ActionRecord) -> float:
    """Return a bounded instability score combining structural and runtime signals.

    Signal weights and maximum contributions:

    ┌──────────────────┬───────────────────────────────────┬──────────┐
    │ Signal           │ Weight                            │ Max      │
    ├──────────────────┼───────────────────────────────────┼──────────┤
    │ depth_score      │ 0.6 per step                      │ 4.0      │
    │ retry_score      │ 1.0 per retry                     │ 4.0      │
    │ failure_score    │ 0.65 per failure (last 5)         │ 3.0      │
    │ repeated_tool    │ 0 / 1 / 2                         │ 2.0      │
    │ latency_score    │ 0.5 per second                    │ 2.0      │
    │ cost_score       │ base (0.025/call) + waste ratio   │ 2.0      │
    │ long_memory_score│ EWMA failure rate × 2 (α=0.10)   │ 2.0      │
    └──────────────────┴───────────────────────────────────┴──────────┘

    ``long_memory_score`` is an EWMA (α=0.10) of the full history failure
    sequence.  It accumulates slowly so that distributed low-rate failure
    (never > 2 consecutive, but e.g. 67% aggregate) builds pressure that
    short windows miss.

    ``cost_score`` replaces the old cumulative-calls×0.3 formula that
    saturated after ~7 calls.  It combines a small per-call base with a
    marginal waste ratio (calls-per-successful-output in the last 10 steps),
    so it reacts to efficiency degradation rather than collapsing into a
    constant floor.

    Thresholds in ``rnos/policy.py``:
        DEGRADE at entropy ≥ 3.0  (default; see also EXP2_POLICY for
        ConfigurableAPI-based experiments which uses higher values tuned for
        that harness)
        REFUSE  at entropy ≥ 6.0
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

    # Marginal cost signal: base per-call accumulation + efficiency waste ratio.
    # Unlike the old cumulative×0.3 formula, this doesn't saturate at 7 calls;
    # it responds to whether calls are producing successful outputs.
    cost_score = _calculate_cost_score(history, candidate)

    # Long-memory failure rate: EWMA over the full history (α = 0.10).
    # Detects distributed, low-rate failure patterns that the 5-step
    # failure_score window misses.
    long_memory_score = _calculate_long_memory_score(history)

    return round(
        depth_score
        + retry_score
        + failure_score
        + repeated_tool
        + latency_score
        + cost_score
        + long_memory_score,
        3,
    )


def _calculate_cost_score(
    history: Sequence[ActionRecord],
    candidate: ActionRecord,
) -> float:
    """Marginal cost: per-call base plus calls-per-successful-output ratio."""
    # Small per-call base cost; saturates at 0.5 after 20+ calls.
    base_cost = min(candidate.cumulative_calls * 0.025, 0.5)

    recent = list(history[-10:])
    if not recent:
        return round(base_cost, 3)

    successes = sum(1 for item in recent if item.success is True)
    # tokens-per-useful-output: rises when failures waste compute
    spend_per_success = len(recent) / max(1, successes)
    marginal_waste = min((spend_per_success - 1.0) * 0.50, 1.5)

    return round(min(base_cost + max(0.0, marginal_waste), 2.0), 3)


def _calculate_long_memory_score(history: Sequence[ActionRecord]) -> float:
    """EWMA failure rate (α=0.10) over full history, scaled to [0, 2.0]."""
    ewma = 0.0
    for item in history:
        if item.success is not None:
            signal = 0.0 if item.success else 1.0
            ewma = 0.10 * signal + 0.90 * ewma
    return round(min(ewma * 2.0, 2.0), 3)
