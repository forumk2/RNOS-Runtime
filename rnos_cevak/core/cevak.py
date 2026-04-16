"""CEVAK monitor: 6D cognitive drift detection.

CevakVector holds the six dimension scores for one evaluation step:
  C — Consistency   : self-consistency of outputs across similar inputs
  E — Evidence      : fraction of claims backed by cited evidence
  V — Variance      : diversity of output content (low = echo chamber risk)
  A — Agreement     : compliance with stated instructions / expected outputs
  K — Confidence    : calibration quality (stated confidence vs. accuracy)
  ADE               : Action Distribution Entropy (action-layer drift)

Dimensions C/E/V/A/K are output-layer signals in [0.0, 1.0] where higher
values indicate healthier behaviour. ADE is a divergence score in [0.0, 1.0]
where 0.0 means the action distribution matches the evaluation baseline and
higher values indicate increasing divergence.

CevakMonitor tracks a baseline vector and computes per-step drift. It
exposes both per-dimension drift and a classified DriftResult. The evasion
failure mode fires when ADE is elevated while all original dimensions remain
close to their baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rnos_cevak.core.ade import ActionDistributionEntropy
from rnos_cevak.core.drift import DriftMode, DriftResult, classify_drift


@dataclass(slots=True)
class CevakVector:
    """One observation of the 6D CEVAK state.

    Args:
        consistency: C — [0.0, 1.0], higher is better.
        evidence: E — [0.0, 1.0], higher is better.
        variance: V — [0.0, 1.0], higher is better.
        agreement: A — [0.0, 1.0], higher is better.
        confidence: K — [0.0, 1.0], higher is better.
        ade: ADE — [0.0, 1.0], lower is better (0.0 = baseline match).
    """

    consistency: float
    evidence: float
    variance: float
    agreement: float
    confidence: float
    ade: float = 0.0

    def original_dims(self) -> tuple[float, float, float, float, float]:
        """Return the five original dimensions as a tuple (C, E, V, A, K)."""
        return (
            self.consistency,
            self.evidence,
            self.variance,
            self.agreement,
            self.confidence,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "C": self.consistency,
            "E": self.evidence,
            "V": self.variance,
            "A": self.agreement,
            "K": self.confidence,
            "ADE": self.ade,
        }


@dataclass
class CevakConfig:
    """Threshold configuration for the CEVAK monitor.

    Args:
        dim_threshold: Per-dimension drift magnitude above which an original
            dimension is considered triggered. Default 0.25.
        ade_threshold: ADE score above which the action distribution is
            considered shifted. Default 0.35.
        clean_mask_required: Minimum number of original dimensions that must
            remain below dim_threshold for evasion to be classified.
            Default 3 (majority of 5).
        incoherence_threshold: Drift magnitude for C or K alone that triggers
            the INCOHERENCE failure mode. Default 0.40.
        ade_window_size: Sliding window size for ADE computation. Default 10.
        ade_max: KL divergence value that maps to ADE score 1.0. Default 2.0.
    """

    dim_threshold: float = 0.25
    ade_threshold: float = 0.35
    clean_mask_required: int = 3
    incoherence_threshold: float = 0.40
    ade_window_size: int = 10
    ade_max: float = 2.0


@dataclass(slots=True)
class CevakResult:
    """Per-step output of the CEVAK monitor."""

    step: int
    current: CevakVector
    baseline: CevakVector | None
    dim_drifts: dict[str, float]
    """Per-dimension drift magnitudes |current - baseline| for C/E/V/A/K."""
    drift_result: DriftResult
    """Classified drift mode and severity."""
    novel_actions: bool
    novel_action_ratio: float


_DIM_NAMES = ["C", "E", "V", "A", "K"]


class CevakMonitor:
    """Stateful 6D drift monitor.

    Usage pattern:

        monitor = CevakMonitor()

        # Calibration phase — record actions and output scores.
        for step in calibration_steps:
            monitor.ade.record_action(step.action_type, step.task_class)
            monitor.record_calibration_vector(CevakVector(...))

        # Freeze baseline.
        monitor.set_baseline(task_class="default")

        # Runtime phase — evaluate each step.
        for step in runtime_steps:
            monitor.ade.record_action(step.action_type, step.task_class)
            result = monitor.evaluate(CevakVector(...), task_class="default")
            if result.drift_result.mode == DriftMode.EVASION:
                # Alert.
    """

    def __init__(self, config: CevakConfig | None = None) -> None:
        self._config = config or CevakConfig()
        self.ade = ActionDistributionEntropy(
            window_size=self._config.ade_window_size,
            max_ade=self._config.ade_max,
        )
        self._baseline: CevakVector | None = None
        self._calibration_vectors: list[CevakVector] = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def record_calibration_vector(self, vec: CevakVector) -> None:
        """Accumulate an output-layer vector during calibration.

        These vectors are averaged when ``set_baseline`` is called to
        produce the reference point for drift computation.
        """
        self._calibration_vectors.append(vec)

    def set_baseline(self, task_class: str = "default") -> CevakVector:
        """Freeze the baseline from accumulated calibration data.

        Also freezes the ADE baseline for *task_class*.

        Returns the frozen baseline CevakVector.

        Raises ValueError if no calibration vectors have been recorded.
        """
        if not self._calibration_vectors:
            raise ValueError(
                "No calibration vectors recorded. "
                "Call record_calibration_vector() before set_baseline()."
            )
        # Average the calibration vectors.
        n = len(self._calibration_vectors)
        self._baseline = CevakVector(
            consistency=round(sum(v.consistency for v in self._calibration_vectors) / n, 6),
            evidence=round(sum(v.evidence for v in self._calibration_vectors) / n, 6),
            variance=round(sum(v.variance for v in self._calibration_vectors) / n, 6),
            agreement=round(sum(v.agreement for v in self._calibration_vectors) / n, 6),
            confidence=round(sum(v.confidence for v in self._calibration_vectors) / n, 6),
            ade=0.0,  # ADE baseline is implicitly 0.0 (no divergence from self).
        )
        self.ade.set_baseline(task_class)
        return self._baseline

    @property
    def baseline(self) -> CevakVector | None:
        """The frozen baseline vector, or None if not yet set."""
        return self._baseline

    # ------------------------------------------------------------------
    # Runtime evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        current: CevakVector,
        task_class: str = "default",
    ) -> CevakResult:
        """Evaluate the current CEVAK state against the frozen baseline.

        The ``current.ade`` field is *overwritten* by the live ADE score
        computed from the action window. Pass any value; it will be replaced.

        Args:
            current: Observed output-layer scores for this step. The ``ade``
                field is ignored; the live ADE score is computed internally.
            task_class: Task class for ADE computation.

        Returns:
            :class:`CevakResult` with drift classification.
        """
        self._step += 1

        # Override ade with live measurement.
        live_ade = self.ade.get_score(task_class)
        current = CevakVector(
            consistency=current.consistency,
            evidence=current.evidence,
            variance=current.variance,
            agreement=current.agreement,
            confidence=current.confidence,
            ade=live_ade,
        )

        if self._baseline is None:
            dim_drifts = {name: 0.0 for name in _DIM_NAMES}
            drift_result = DriftResult(
                mode=DriftMode.NONE,
                severity=0.0,
                triggered_dimensions=[],
                clean_mask_count=5,
                notes=["No baseline set; drift cannot be computed."],
            )
            return CevakResult(
                step=self._step,
                current=current,
                baseline=None,
                dim_drifts=dim_drifts,
                drift_result=drift_result,
                novel_actions=self.ade.has_novel_actions(task_class),
                novel_action_ratio=self.ade.novel_action_ratio(task_class),
            )

        # Compute per-dimension drift (absolute deviation from baseline).
        baseline_dims = self._baseline.original_dims()
        current_dims = current.original_dims()
        dim_drifts = {
            name: round(abs(current_dims[i] - baseline_dims[i]), 6)
            for i, name in enumerate(_DIM_NAMES)
        }

        drift_result = classify_drift(
            dim_drifts=list(dim_drifts.values()),
            ade_score=live_ade,
            dim_names=_DIM_NAMES,
            dim_threshold=self._config.dim_threshold,
            ade_threshold=self._config.ade_threshold,
            clean_mask_required=self._config.clean_mask_required,
            incoherence_threshold=self._config.incoherence_threshold,
        )

        return CevakResult(
            step=self._step,
            current=current,
            baseline=self._baseline,
            dim_drifts=dim_drifts,
            drift_result=drift_result,
            novel_actions=self.ade.has_novel_actions(task_class),
            novel_action_ratio=self.ade.novel_action_ratio(task_class),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_line(self, result: CevakResult) -> str:
        """Format a single-line summary of a CevakResult for logging."""
        drifts = " ".join(
            f"{k}={v:.3f}" for k, v in result.dim_drifts.items()
        )
        novel = f" novel={result.novel_action_ratio:.2f}" if result.novel_actions else ""
        return (
            f"step={result.step} "
            f"ADE={result.current.ade:.3f} "
            f"{drifts}"
            f"{novel} "
            f"mode={result.drift_result.mode.value} "
            f"severity={result.drift_result.severity:.3f}"
        )
