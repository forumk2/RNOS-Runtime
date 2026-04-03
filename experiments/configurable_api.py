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


def make_noisy_recovery(seed: int = 42) -> tuple["ConfigurableAPI", list[dict[str, Any]]]:
    """Noisy but genuinely recovering scenario.

    Steps 1-10 have diffuse instability (~40% failure rate, <=2 consecutive).
    Steps 11-14 transition: failure rate falls to ~25%, latency begins normalising.
    Steps 15-20 stabilise: 6 consecutive clean successes at baseline latency.

    The failure pattern in steps 1-10 is deliberately identical to
    make_smoldering_instability so the two scenarios are locally indistinguishable
    during the early phase.

    Constraints:
      - No more than 2 consecutive failures at any point.
      - Steps 15-20: all successes, latency < 90 ms (genuine stabilisation).

    Ground truth: benign -- should NOT be permanently intervened on.
    Temporary DEGRADE is acceptable; REFUSE is incorrect.
    """
    schedule = [
        {"success": True},  {"success": True},                      #  1-2  stable
        {"success": False}, {"success": False},                     #  3-4  mini-burst 1 (2 consecutive)
        {"success": True},  {"success": True},                      #  5-6
        {"success": False},                                         #  7    isolated failure
        {"success": True},                                          #  8
        {"success": False},                                         #  9    isolated failure
        {"success": True},                                          # 10
        {"success": False},                                         # 11   residual noise
        {"success": True},  {"success": True},                      # 12-13 transition
        {"success": True},                                          # 14
        {"success": True},  {"success": True},  {"success": True},  # 15-17 stabilisation
        {"success": True},  {"success": True},  {"success": True},  # 18-20
    ]
    latency = [
         80.0,  80.0,                   #  1-2  baseline
        350.0, 380.0,                   #  3-4  burst
        120.0, 100.0,                   #  5-6  elevated recovery
        320.0,                          #  7    failure
         80.0,                          #  8
        300.0,                          #  9    failure
         80.0,                          # 10
        250.0,                          # 11   residual noise
        100.0,  90.0,  85.0,            # 12-14 normalising
         80.0,  80.0,  80.0,            # 15-17 stable
         80.0,  80.0,  80.0,            # 18-20
    ]
    cost = (
        [0.010] * 2          # 1-2
        + [0.020] * 2        # 3-4
        + [0.012] * 2        # 5-6
        + [0.018]            # 7
        + [0.010]            # 8
        + [0.018]            # 9
        + [0.010] * 11       # 10-20
    )
    segments: list[dict[str, Any]] = [
        {"segment_index": 0, "start_step":  1, "end_step":  2, "segment_type": "stable"},
        {"segment_index": 1, "start_step":  3, "end_step": 10, "segment_type": "noisy"},
        {"segment_index": 2, "start_step": 11, "end_step": 14, "segment_type": "transition"},
        {"segment_index": 3, "start_step": 15, "end_step": 20, "segment_type": "stable"},
    ]
    api = ConfigurableAPI(
        name="noisy_recovery",
        step_schedule=schedule,
        fail_probs=[0.0],
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
    return api, segments


def make_smoldering_instability(seed: int = 42) -> tuple["ConfigurableAPI", list[dict[str, Any]]]:
    """Persistent low-grade instability that never stabilises.

    Steps 1-10 are deliberately identical in failure pattern to make_noisy_recovery
    (4/10 = 40%, <= 2 consecutive), making the two scenarios indistinguishable
    to a short-horizon observer during the early phase.

    Steps 11-20 continue to fail at ~40% (failures at steps 13, 14, 16, 17).
    All success steps in 11-20 carry elevated latency (200-230 ms, >= 1.5x baseline),
    so no clean stabilisation window forms.

    The adversarial design: consecutive failure streaks never exceed 2, which caps
    RNOS's retry_score at 2.0 and keeps peak entropy just below the DEGRADE
    threshold (9.0) even at worst-case windows.

    The adaptive CB detects the scenario at step 18 when the window fills with
    the FFSFF pattern (4/5 = 0.80 > 0.60 strict threshold).

    Constraints:
      - No more than 2 consecutive failures at any point.
      - All success steps from step 5 onward have latency >= 140 ms (>= 1.5x baseline),
        so stability_score never reaches 3 after step 10.
      - Rolling 5-step failure rate never drops below 0.20 from step 13 onward.

    Ground truth: structural failure -- should eventually be intervened on.
    """
    schedule = [
        {"success": True},  {"success": True},                      #  1-2  stable (mirrors noisy)
        {"success": False}, {"success": False},                     #  3-4  mini-burst 1 (mirrors noisy)
        {"success": True},  {"success": True},                      #  5-6  (mirrors noisy)
        {"success": False},                                         #  7    (mirrors noisy)
        {"success": True},                                          #  8    (mirrors noisy)
        {"success": False},                                         #  9    (mirrors noisy)
        {"success": True},                                          # 10   (mirrors noisy)
        {"success": True},  {"success": True},                      # 11-12 deceptive clean window
        {"success": False}, {"success": False},                     # 13-14 mini-burst 2 (2 consecutive)
        {"success": True},                                          # 15   elevated success
        {"success": False}, {"success": False},                     # 16-17 mini-burst 3 (2 consecutive)
        {"success": True},  {"success": True},  {"success": True},  # 18-20 elevated successes
    ]
    latency = [
         80.0,  80.0,                   #  1-2  baseline
        350.0, 380.0,                   #  3-4  burst (same as noisy)
        160.0, 140.0,                   #  5-6  elevated (vs 120/100 in noisy)
        320.0,                          #  7    failure (same as noisy)
        160.0,                          #  8    elevated (vs 80 in noisy)
        300.0,                          #  9    failure (same as noisy)
        150.0,                          # 10   elevated (vs 80 in noisy)
        220.0, 230.0,                   # 11-12 elevated even on success
        400.0, 420.0,                   # 13-14 burst 2
        220.0,                          # 15   elevated
        390.0, 410.0,                   # 16-17 burst 3
        200.0, 210.0, 200.0,            # 18-20 elevated successes (no clean recovery)
    ]
    cost = (
        [0.010] * 2          # 1-2
        + [0.020] * 2        # 3-4
        + [0.014] * 2        # 5-6
        + [0.018]            # 7
        + [0.014]            # 8
        + [0.018]            # 9
        + [0.014]            # 10
        + [0.020] * 2        # 11-12
        + [0.030] * 2        # 13-14
        + [0.022]            # 15
        + [0.028] * 2        # 16-17
        + [0.020] * 3        # 18-20
    )
    segments: list[dict[str, Any]] = [
        {"segment_index": 0, "start_step":  1, "end_step":  2, "segment_type": "stable"},
        {"segment_index": 1, "start_step":  3, "end_step": 10, "segment_type": "noisy"},
        {"segment_index": 2, "start_step": 11, "end_step": 20, "segment_type": "chronic"},
    ]
    api = ConfigurableAPI(
        name="smoldering_instability",
        step_schedule=schedule,
        fail_probs=[0.0],
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
    return api, segments


def make_bursty_recovery(seed: int = 42) -> tuple["ConfigurableAPI", list[dict[str, Any]]]:
    """Two short failure bursts followed by genuine sustained recovery.

    Pattern (20 steps):
      steps  1-2:  stable successes (80 ms)
      steps  3-5:  burst 1 — 3 failures (400 ms)
      steps  6-7:  recovery 1 — 2 successes (200 ms, elevated but stabilising)
      steps  8-9:  burst 2 — 2 failures (350 ms)
      steps 10-20: sustained recovery — latency normalises 180 → 80 ms

    Ground truth: benign — should NOT permanently intervene.
    Temporary DEGRADE is acceptable; REFUSE is incorrect.
    """
    schedule = [
        {"success": True},  {"success": True},                              # 1-2
        {"success": False}, {"success": False}, {"success": False},         # 3-5 burst 1
        {"success": True},  {"success": True},                              # 6-7 recovery 1
        {"success": False}, {"success": False},                             # 8-9 burst 2
        {"success": True},  {"success": True},  {"success": True},          # 10-12
        {"success": True},  {"success": True},  {"success": True},          # 13-15
        {"success": True},  {"success": True},  {"success": True},          # 16-18
        {"success": True},  {"success": True},                              # 19-20
    ]
    latency = [
        80.0, 80.0,                                  # 1-2
        400.0, 400.0, 400.0,                         # 3-5 burst 1
        200.0, 200.0,                                # 6-7 recovery 1 (elevated)
        350.0, 350.0,                                # 8-9 burst 2
        180.0, 160.0, 140.0, 120.0, 100.0, 90.0,    # 10-15 normalising
        85.0, 82.0, 80.0, 80.0, 80.0,               # 16-20 stable
    ]
    cost = (
        [0.010, 0.010]          # 1-2
        + [0.020] * 3           # 3-5 burst 1
        + [0.015, 0.015]        # 6-7 recovery 1
        + [0.020, 0.020]        # 8-9 burst 2
        + [0.010] * 11          # 10-20 recovery
    )
    burst_segments: list[dict[str, Any]] = [
        {"burst_index": 0, "start_step":  1, "end_step":  2, "segment_type": "stable"},
        {"burst_index": 1, "start_step":  3, "end_step":  5, "segment_type": "burst"},
        {"burst_index": 1, "start_step":  6, "end_step":  7, "segment_type": "recovery"},
        {"burst_index": 2, "start_step":  8, "end_step":  9, "segment_type": "burst"},
        {"burst_index": 2, "start_step": 10, "end_step": 20, "segment_type": "recovery"},
    ]
    api = ConfigurableAPI(
        name="bursty_recovery",
        step_schedule=schedule,
        fail_probs=[0.0],
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
    return api, burst_segments


def make_intermittent_cascade(seed: int = 42) -> tuple["ConfigurableAPI", list[dict[str, Any]]]:
    """Three failure bursts with dirty recovery windows; third burst arrives late.

    Pattern (20 steps):
      steps  1-2:  stable successes (80 ms)
      steps  3-5:  burst 1 — 3 failures (400 ms)
      steps  6-7:  recovery 1 — 2 *dirty* successes (280 ms, elevated — not clean)
      steps  8-10: burst 2 — 3 failures (430 ms)
      steps 11-13: recovery 2 — 3 *deceptive* successes (300 ms elevated — appears stable)
      steps 14-16: burst 3 — 3 failures (460 ms)
      steps 17-20: absorbing failure — fail_probs=[1.0] at 480 ms

    Recovery windows have persistently elevated latency throughout — dirty recoveries.
    Burst 3 arrives after a deceptively long 3-step recovery window (steps 11-13),
    testing whether the controller maintains vigilance after apparent stabilisation.

    Ground truth: structural — should intervene.

    Key design: burst 2 has 3 failures (vs 2 for bursty_recovery), which is enough
    to push RNOS's retry_score + failure_score past the DEGRADE threshold when combined
    with the structural floor (cost_score + repeated_tool).  The adaptive CB's 3/5=0.60
    rate check uses strict '>' so it does NOT trip at the same point.
    """
    schedule = [
        {"success": True},  {"success": True},                              # 1-2 stable
        {"success": False}, {"success": False}, {"success": False},         # 3-5 burst 1
        {"success": True},  {"success": True},                              # 6-7 recovery 1 (dirty)
        {"success": False}, {"success": False}, {"success": False},         # 8-10 burst 2
        {"success": True},  {"success": True},  {"success": True},          # 11-13 recovery 2 (deceptive)
        {"success": False}, {"success": False}, {"success": False},         # 14-16 burst 3
    ]
    latency = [
        80.0, 80.0,                  # 1-2
        400.0, 400.0, 400.0,         # 3-5 burst 1
        280.0, 280.0,                # 6-7 recovery 1 (elevated, dirty)
        430.0, 430.0, 430.0,         # 8-10 burst 2
        300.0, 300.0, 300.0,         # 11-13 recovery 2 (elevated, deceptive)
        460.0, 460.0, 460.0,         # 14-16 burst 3
    ] + [480.0] * 4                  # 17-20 absorbing regime
    cost = (
        [0.010, 0.010]               # 1-2
        + [0.025] * 3                # 3-5 burst 1
        + [0.020, 0.020]             # 6-7 recovery 1
        + [0.030] * 3                # 8-10 burst 2
        + [0.025] * 3                # 11-13 recovery 2
        + [0.035] * 3                # 14-16 burst 3
        + [0.040] * 4                # 17-20
    )
    burst_segments: list[dict[str, Any]] = [
        {"burst_index": 0, "start_step":  1, "end_step":  2, "segment_type": "stable"},
        {"burst_index": 1, "start_step":  3, "end_step":  5, "segment_type": "burst"},
        {"burst_index": 1, "start_step":  6, "end_step":  7, "segment_type": "recovery"},
        {"burst_index": 2, "start_step":  8, "end_step": 10, "segment_type": "burst"},
        {"burst_index": 2, "start_step": 11, "end_step": 13, "segment_type": "recovery"},
        {"burst_index": 3, "start_step": 14, "end_step": 20, "segment_type": "burst"},
    ]
    api = ConfigurableAPI(
        name="intermittent_cascade",
        step_schedule=schedule,
        fail_probs=[1.0],            # steps 17+: certain failure
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
    return api, burst_segments


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
