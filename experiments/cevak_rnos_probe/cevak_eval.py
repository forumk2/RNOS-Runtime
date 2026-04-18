"""Simplified CEVAK evaluator for the probe experiment.

Uses the real ActionDistributionEntropy and classify_drift implementations
from rnos_cevak.core — faithful to the actual formulation.

Simplified only in evaluation style: we run the full calibration/runtime
sequence then classify at the final step. The underlying math is unchanged.

Dimension mapping from Step fields to CEVAK C/E/V/A/K:
  C (Consistency)  <- step.consistency
  E (Evidence)     <- step.evidence_strength
  V (Variance)     <- step.variance
  A (Agreement)    <- step.agreement
  K (Confidence)   <- step.confidence
"""

from __future__ import annotations

from dataclasses import dataclass

from rnos_cevak.core.ade import ActionDistributionEntropy
from rnos_cevak.core.drift import DriftMode, classify_drift

from .scenario_generator import Step, Trace

# ---------------------------------------------------------------------------
# Configuration — matches CevakConfig defaults
# ---------------------------------------------------------------------------
BASELINE_WINDOW: int = 20
ADE_WINDOW: int = 20
ADE_MAX: float = 2.0
DIM_THRESHOLD: float = 0.25
ADE_THRESHOLD: float = 0.35
CLEAN_MASK_REQUIRED: int = 3
INCOHERENCE_THRESHOLD: float = 0.40

_DIM_NAMES = ["C", "E", "V", "A", "K"]


@dataclass(slots=True)
class CevakResult:
    mode: str               # none | overreach | echo_chamber | incoherence | evasion
    action: str             # CONTINUE | MONITOR | FLAG
    ade_score: float
    dim_drifts: dict[str, float]
    novel_actions: bool


def _dims(step: Step) -> tuple[float, float, float, float, float]:
    """C, E, V, A, K order."""
    return (
        step.consistency,
        step.evidence_strength,
        step.variance,
        step.agreement,
        step.confidence,
    )


def _mode_to_action(mode: DriftMode) -> str:
    if mode == DriftMode.EVASION:
        return "FLAG"
    if mode in (DriftMode.OVERREACH, DriftMode.ECHO_CHAMBER, DriftMode.INCOHERENCE):
        return "MONITOR"
    return "CONTINUE"


def evaluate(trace: Trace) -> CevakResult:
    """Evaluate a trace with the simplified CEVAK evaluator.

    Calibration: first BASELINE_WINDOW steps set P (action baseline) and
    the reference output vector (mean of calibration dims).
    Runtime: remaining steps feed the ADE sliding window.
    Classification: applied at the final step.
    """
    steps = trace.steps
    n = len(steps)
    split = min(BASELINE_WINDOW, max(1, n // 2))

    ade = ActionDistributionEntropy(window_size=ADE_WINDOW, max_ade=ADE_MAX)

    # --- Calibration phase ---
    cal: list[tuple[float, ...]] = []
    for step in steps[:split]:
        ade.record_action(step.action_type)
        cal.append(_dims(step))

    baseline = tuple(sum(d[i] for d in cal) / len(cal) for i in range(5))
    ade.set_baseline()

    # --- Runtime phase: feed the sliding window ---
    for step in steps[split:]:
        ade.record_action(step.action_type)

    # --- Classify at final step ---
    final = steps[-1]
    current = _dims(final)
    live_ade = ade.get_score()

    drifts = [abs(current[i] - baseline[i]) for i in range(5)]
    dim_drifts = dict(zip(_DIM_NAMES, drifts))

    drift_result = classify_drift(
        dim_drifts=drifts,
        ade_score=live_ade,
        dim_names=_DIM_NAMES,
        dim_threshold=DIM_THRESHOLD,
        ade_threshold=ADE_THRESHOLD,
        clean_mask_required=CLEAN_MASK_REQUIRED,
        incoherence_threshold=INCOHERENCE_THRESHOLD,
    )

    return CevakResult(
        mode=drift_result.mode.value,
        action=_mode_to_action(drift_result.mode),
        ade_score=round(live_ade, 4),
        dim_drifts={k: round(v, 4) for k, v in dim_drifts.items()},
        novel_actions=ade.has_novel_actions(),
    )
