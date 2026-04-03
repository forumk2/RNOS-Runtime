"""Configurable, deterministic failure source for RNOS Experiment 2.

A single parameterised class replaces separate per-scenario API files.
Factory functions produce the four preset scenarios required by Experiment 2.
All behaviour is seeded and fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StepOutcome:
    """Result of one simulated API call."""

    success: bool
    latency_ms: float
    cost: float
    step: int
    consecutive_failures: int  # updated count *after* this step
    metadata: dict[str, Any] = field(default_factory=dict)


class ConfigurableAPI:
    """Parameterised failure source with seeded-deterministic behaviour.

    Supports two complementary scheduling modes:

    *Explicit schedule* (``step_schedule``):
        A list of ``{"success": bool}`` dicts.  The first ``len(step_schedule)``
        calls use these entries verbatim — no RNG involved.

    *Probabilistic mode* (``fail_probs``):
        A list of base failure probabilities, one per step beyond the explicit
        schedule (last value extended to cover all remaining steps).
        When ``compound_factor != 1.0``, the effective probability is
        ``min(base * compound_factor ** consecutive_failures, 1.0)``,
        making the failure probability grow after each consecutive failure —
        useful for modelling a runaway cascade.

    Latency and cost profiles are indexed by step number (last value extended).
    Neither uses the primary RNG so they never perturb the failure sequence.
    """

    def __init__(
        self,
        name: str,
        step_schedule: list[dict[str, Any]] | None = None,
        fail_probs: list[float] | None = None,
        latency_profile: list[float] | None = None,
        cost_profile: list[float] | None = None,
        compound_factor: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.name = name
        self._step_schedule: list[dict[str, Any]] = step_schedule or []
        self._fail_probs: list[float] = fail_probs or [0.0]
        self._latency_profile: list[float] = latency_profile or [100.0]
        self._cost_profile: list[float] = cost_profile or [0.01]
        self._compound_factor = compound_factor
        self._seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def call(self) -> StepOutcome:
        """Execute one step and return the outcome."""
        self._step += 1
        step = self._step

        # Outcome -------------------------------------------------------
        if step <= len(self._step_schedule):
            success = bool(self._step_schedule[step - 1].get("success", True))
        else:
            idx = min(step - 1 - len(self._step_schedule), len(self._fail_probs) - 1)
            base_prob = self._fail_probs[idx]
            if self._compound_factor != 1.0 and self._consecutive_failures > 0:
                prob = min(base_prob * (self._compound_factor ** self._consecutive_failures), 1.0)
            else:
                prob = base_prob
            # Always consume one RNG value for reproducibility even when certain.
            success = self._rng.random() > prob

        # Latency (deterministic from profile; no RNG consumed) ---------
        lat_idx = min(step - 1, len(self._latency_profile) - 1)
        latency = self._latency_profile[lat_idx]

        # Cost ----------------------------------------------------------
        cost_idx = min(step - 1, len(self._cost_profile) - 1)
        cost = self._cost_profile[cost_idx]

        # State update --------------------------------------------------
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        return StepOutcome(
            success=success,
            latency_ms=latency,
            cost=cost,
            step=step,
            consecutive_failures=self._consecutive_failures,
        )

    def reset(self) -> None:
        """Return to initial state with the original seed."""
        self._rng = random.Random(self._seed)
        self._step = 0
        self._consecutive_failures = 0


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


def make_transient_blip(seed: int = 42) -> ConfigurableAPI:
    """Steps 1-2 succeed, step 3 fails, steps 4+ succeed.

    Ground truth: recoverable — should NOT intervene.
    """
    schedule = [
        {"success": True},
        {"success": True},
        {"success": False},   # single transient failure
    ]
    latency = [80.0, 80.0, 500.0] + [80.0] * 17   # latency spike on failure
    return ConfigurableAPI(
        name="transient_blip",
        step_schedule=schedule,
        fail_probs=[0.0],           # steps 4+: always succeed
        latency_profile=latency,
        cost_profile=[0.01] * 20,
        seed=seed,
    )


def make_rough_patch(seed: int = 42, failure_length: int = 3) -> ConfigurableAPI:
    """Steps 1-2 succeed, steps 3-(2+failure_length) fail, then recover.

    Ground truth: recoverable — should generally NOT intervene.

    Args:
        failure_length: Number of consecutive failures in the failure window
            (default 3).  Used by the phase-transition sweep.
    """
    schedule = (
        [{"success": True}, {"success": True}]
        + [{"success": False}] * failure_length
    )
    tail = max(0, 20 - 2 - failure_length)
    latency = [80.0, 80.0] + [400.0] * failure_length + [80.0] * tail
    return ConfigurableAPI(
        name="rough_patch",
        step_schedule=schedule,
        fail_probs=[0.0],           # post-schedule: always succeed
        latency_profile=latency,
        cost_profile=[0.01] * 20,
        seed=seed,
    )


def make_slow_burn(seed: int = 42) -> ConfigurableAPI:
    """Gradually degrading success probability — analysis scenario only.

    Success probability: ~95% → 85% → 70% → 55% → 40%.

    Ground truth: borderline — excluded from binary selectivity scoring.
    The policy label is unclear because the failure rate never becomes
    absorbing; it merely increases the probability of consecutive failures
    that RNOS would eventually flag.
    """
    fail_probs = [
        0.05, 0.05, 0.05, 0.05,    # steps 1-4:  ~95% success
        0.15, 0.15, 0.15, 0.15,    # steps 5-8:  ~85% success
        0.30, 0.30, 0.30, 0.30,    # steps 9-12: ~70% success
        0.45, 0.45, 0.45, 0.45,    # steps 13-16: ~55% success
        0.60, 0.60, 0.60, 0.60,    # steps 17-20: ~40% success
    ]
    latency = [80.0 + i * 15.0 for i in range(20)]
    cost = [0.01 + i * 0.002 for i in range(20)]
    return ConfigurableAPI(
        name="slow_burn",
        fail_probs=fail_probs,
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )


def make_matched_recovery(seed: int = 42) -> ConfigurableAPI:
    """Matched pair (recovery branch): identical to make_matched_collapse through step 6.

    Steps 1-2 succeed (normal, 80ms).
    Step 3 succeeds with high latency (500ms) — instability signal begins.
    Steps 4-6 fail (400ms) — shared instability window.
    Steps 7+ succeed with decreasing latency — genuine recovery.

    Ground truth: benign — should NOT intervene.

    The step-6 entropy is identical to make_matched_collapse; divergence
    becomes observable only from step 7 onward.
    """
    schedule = [
        {"success": True},
        {"success": True},
        {"success": True},    # step 3: success but slow (instability onset)
        {"success": False},   # step 4: first shared failure
        {"success": False},   # step 5: second shared failure
        {"success": False},   # step 6: third shared failure
        {"success": True},    # step 7: recovery begins (divergence point)
        {"success": True},
        {"success": True},
        {"success": True},
        {"success": True},
        {"success": True},
    ]
    latency = [
        80.0, 80.0, 500.0,        # steps 1-3
        400.0, 400.0, 400.0,      # steps 4-6 (shared failure window)
        350.0, 280.0, 200.0, 150.0, 100.0, 80.0,  # steps 7-12: recovery
    ] + [80.0] * 8                # steps 13-20
    return ConfigurableAPI(
        name="matched_recovery",
        step_schedule=schedule,
        fail_probs=[0.0],          # steps beyond schedule: always succeed
        latency_profile=latency,
        cost_profile=[0.01] * 20,
        seed=seed,
    )


def make_matched_collapse(seed: int = 42) -> ConfigurableAPI:
    """Matched pair (collapse branch): identical to make_matched_recovery through step 6.

    Steps 1-2 succeed (normal, 80ms).
    Step 3 succeeds with high latency (500ms) — instability signal begins.
    Steps 4-6 fail (400ms) — shared instability window.
    Steps 7-9 fail explicitly [420ms, 450ms, 480ms] — collapse deepens.
    Steps 10+ always fail (fail_probs=[1.0]) at 500ms — absorbing failure regime.

    Ground truth: structural — should intervene.

    After step 6 consecutive_failures=3; divergence point is step 7.
    """
    schedule = [
        {"success": True},
        {"success": True},
        {"success": True},    # step 3: success but slow
        {"success": False},   # step 4: first shared failure
        {"success": False},   # step 5: second shared failure
        {"success": False},   # step 6: third shared failure (end of shared window)
        {"success": False},   # step 7: collapse deepens (divergence point)
        {"success": False},   # step 8
        {"success": False},   # step 9
    ]
    latency = [
        80.0, 80.0, 500.0,              # steps 1-3
        400.0, 400.0, 400.0,            # steps 4-6 (shared failure window)
        420.0, 450.0, 480.0,            # steps 7-9 (collapse deepens)
    ] + [500.0] * 11                    # steps 10-20 (absorbing regime)
    return ConfigurableAPI(
        name="matched_collapse",
        step_schedule=schedule,
        fail_probs=[1.0],              # steps 10+: certain failure
        latency_profile=latency,
        cost_profile=[0.01] * 20,
        seed=seed,
    )


def make_runaway_cascade(seed: int = 42) -> ConfigurableAPI:
    """Steps 1-2 succeed; steps 3-4 fail explicitly; steps 5+ certain failure.

    After step 4 there are 2 consecutive failures (``consecutive_failures=2``).
    The compound formula then gives:
        fail_prob = 0.5 × 1.8² = 1.62 → capped at 1.0

    Steps 5 onwards are therefore always failures — the scenario has entered
    an absorbing failure regime.  The critical threshold is
    ``consecutive_failures >= 2``.

    Ground truth: structural failure — should intervene.
    """
    schedule = [
        {"success": True},
        {"success": True},
        {"success": False},   # explicit first trigger
        {"success": False},   # explicit second; sets consecutive=2
    ]
    # Step 5+: base=0.5, compound=1.8  →  0.5*1.8²=1.62 → 1.0 (certain)
    latency = [80.0, 80.0, 400.0, 450.0] + [500.0 + i * 30.0 for i in range(16)]
    cost = [0.01, 0.01] + [0.02 + i * 0.01 for i in range(18)]
    return ConfigurableAPI(
        name="runaway_cascade",
        step_schedule=schedule,
        fail_probs=[0.5],
        compound_factor=1.8,
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
