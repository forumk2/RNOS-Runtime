from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


def _sample_poisson(rng: random.Random, lam: float) -> int:
    """Knuth sampler; sufficient for the small rates used here."""
    limit = math.exp(-lam)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


@dataclass(frozen=True)
class EmissionLaw:
    """Poisson-Bernoulli observation model from the paper."""

    lambda_rate: float
    validation_prob: float

    def sample(self, rng: random.Random) -> tuple[int, int]:
        branching = _sample_poisson(rng, self.lambda_rate)
        validation = 1 if rng.random() < self.validation_prob else 0
        return branching, validation

    def log_prob(self, branching: int, validation: int) -> float:
        if self.lambda_rate <= 0:
            raise ValueError("lambda_rate must be positive.")
        if not 0.0 < self.validation_prob < 1.0:
            raise ValueError("validation_prob must lie strictly between 0 and 1.")
        poisson_log_prob = (
            -self.lambda_rate
            + branching * math.log(self.lambda_rate)
            - math.lgamma(branching + 1)
        )
        bernoulli_log_prob = (
            math.log(self.validation_prob)
            if validation
            else math.log(1.0 - self.validation_prob)
        )
        return poisson_log_prob + bernoulli_log_prob


@dataclass(frozen=True)
class Observation:
    step: int
    regime: str
    branching: int
    validation: int
    disturbance: bool
    clean_probe: bool
    progress_hint: float
    trust_override: float | None = None
    unresolved_entropy_override: float | None = None
    info_gain_override: float | None = None
    escaped_trap: bool = False


class BaseEnvironment:
    """Small environment interface: reset, step, finalize."""

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, mode: str, state: Any, step_index: int) -> Observation:
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:
        return {}


class PoissonBernoulliEnvironment(BaseEnvironment):
    """Observation-driven environment for healthy, degraded, or Bernoulli noise."""

    def __init__(
        self,
        healthy_law: EmissionLaw,
        degraded_law: EmissionLaw,
        regime_policy: str,
        disturbance_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.healthy_law = healthy_law
        self.degraded_law = degraded_law
        self.regime_policy = regime_policy
        self.disturbance_prob = disturbance_prob

    def _sample_regime(self) -> tuple[str, bool, EmissionLaw]:
        if self.regime_policy == "healthy":
            return "healthy", False, self.healthy_law
        if self.regime_policy == "degraded":
            return "degraded", True, self.degraded_law
        if self.regime_policy == "bernoulli":
            disturbance = self._rng.random() < self.disturbance_prob
            if disturbance:
                return "degraded", True, self.degraded_law
            return "healthy", False, self.healthy_law
        raise ValueError(f"Unsupported regime policy: {self.regime_policy}")

    def step(self, mode: str, state: Any, step_index: int) -> Observation:
        regime, disturbance, law = self._sample_regime()
        branching, validation = law.sample(self._rng)
        clean_probe = (validation == 1) and (not disturbance)
        progress = 1.0 if mode == "E" and validation == 1 else 0.0
        return Observation(
            step=step_index,
            regime=regime,
            branching=branching,
            validation=validation,
            disturbance=disturbance,
            clean_probe=clean_probe,
            progress_hint=progress,
        )


class RecoverableNoiseEnvironment(BaseEnvironment):
    """Exact Proposition 1 / Theorem 3-style construction with U_t = 0."""

    def __init__(
        self,
        healthy_law: EmissionLaw,
        degraded_law: EmissionLaw,
        disturbance_prob: float,
        trust_execute: float,
        trust_quarantine: float,
        execute_progress: float = 1.0,
    ) -> None:
        super().__init__()
        self.healthy_law = healthy_law
        self.degraded_law = degraded_law
        self.disturbance_prob = disturbance_prob
        self.trust_execute = trust_execute
        self.trust_quarantine = trust_quarantine
        self.execute_progress = execute_progress

    def _observe(self, law: EmissionLaw) -> tuple[int, int]:
        return law.sample(self._rng)

    def step(self, mode: str, state: Any, step_index: int) -> Observation:
        if mode == "Q":
            branching, validation = self._observe(self.healthy_law)
            return Observation(
                step=step_index,
                regime="quarantine-recovery",
                branching=branching,
                validation=validation,
                disturbance=False,
                clean_probe=True,
                progress_hint=0.0,
                trust_override=self.trust_execute,
                unresolved_entropy_override=0.0,
            )

        if mode == "S":
            clean_probe = self._rng.random() >= self.disturbance_prob
            law = self.healthy_law if clean_probe else self.degraded_law
            branching, validation = self._observe(law)
            next_counter = (
                min(state.recovery_target, state.recovery_counter + 1)
                if clean_probe
                else 0
            )
            restored = clean_probe and next_counter >= state.recovery_target
            return Observation(
                step=step_index,
                regime="suspension-clean" if clean_probe else "suspension-noisy",
                branching=branching,
                validation=validation,
                disturbance=not clean_probe,
                clean_probe=clean_probe,
                progress_hint=0.0,
                trust_override=self.trust_execute if restored else self.trust_quarantine,
                unresolved_entropy_override=0.0,
            )

        disturbance = self._rng.random() < self.disturbance_prob
        law = self.degraded_law if disturbance else self.healthy_law
        branching, validation = self._observe(law)
        progress = self.execute_progress if (not disturbance and validation == 1) else 0.0
        return Observation(
            step=step_index,
            regime="degraded" if disturbance else "healthy",
            branching=branching,
            validation=validation,
            disturbance=disturbance,
            clean_probe=not disturbance,
            progress_hint=progress,
            trust_override=self.trust_quarantine if disturbance else self.trust_execute,
            unresolved_entropy_override=0.0,
        )


class AlternatingLivelockEnvironment(BaseEnvironment):
    """Deterministic 1,0,1,0,... clean-probe construction from Proposition 2."""

    def __init__(
        self,
        healthy_law: EmissionLaw,
        degraded_law: EmissionLaw,
        trust_execute: float,
        trust_quarantine: float,
    ) -> None:
        super().__init__()
        self.healthy_law = healthy_law
        self.degraded_law = degraded_law
        self.trust_execute = trust_execute
        self.trust_quarantine = trust_quarantine

    def step(self, mode: str, state: Any, step_index: int) -> Observation:
        clean_probe = (step_index % 2) == 0
        law = self.healthy_law if clean_probe else self.degraded_law
        branching, validation = law.sample(self._rng)

        if mode == "S":
            next_counter = (
                min(state.recovery_target, state.recovery_counter + 1)
                if clean_probe
                else 0
            )
            restored = clean_probe and next_counter >= state.recovery_target
            return Observation(
                step=step_index,
                regime="livelock-clean" if clean_probe else "livelock-noisy",
                branching=branching,
                validation=validation,
                disturbance=not clean_probe,
                clean_probe=clean_probe,
                progress_hint=0.0,
                trust_override=self.trust_execute if restored else self.trust_quarantine,
                unresolved_entropy_override=0.0,
            )

        return Observation(
            step=step_index,
            regime="bootstrap",
            branching=branching,
            validation=validation,
            disturbance=True,
            clean_probe=False,
            progress_hint=0.0,
            trust_override=self.trust_quarantine,
            unresolved_entropy_override=0.0,
        )


class TrapEscapeEnvironment(BaseEnvironment):
    """High-trust / low-progress trap where recovery actions can escape the basin."""

    def __init__(
        self,
        healthy_law: EmissionLaw,
        degraded_law: EmissionLaw,
        disturbance_prob: float,
        trust_execute: float,
        trust_quarantine: float,
        trap_progress: float = 0.05,
        productive_progress: float = 1.0,
        escape_probability_on_recovery: float = 0.8,
    ) -> None:
        super().__init__()
        self.healthy_law = healthy_law
        self.degraded_law = degraded_law
        self.disturbance_prob = disturbance_prob
        self.trust_execute = trust_execute
        self.trust_quarantine = trust_quarantine
        self.trap_progress = trap_progress
        self.productive_progress = productive_progress
        self.escape_probability_on_recovery = escape_probability_on_recovery
        self.trap_active = True
        self.escape_step: int | None = None

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.trap_active = True
        self.escape_step = None

    def _maybe_escape(self, step_index: int) -> bool:
        if self.trap_active and self._rng.random() < self.escape_probability_on_recovery:
            self.trap_active = False
            self.escape_step = step_index
            return True
        return False

    def step(self, mode: str, state: Any, step_index: int) -> Observation:
        if mode == "Q":
            escaped = self._maybe_escape(step_index)
            branching, validation = self.healthy_law.sample(self._rng)
            return Observation(
                step=step_index,
                regime="trap-reset",
                branching=branching,
                validation=validation,
                disturbance=False,
                clean_probe=True,
                progress_hint=0.0,
                trust_override=self.trust_execute,
                unresolved_entropy_override=0.0,
                escaped_trap=escaped,
            )

        if mode == "S":
            clean_probe = self._rng.random() >= self.disturbance_prob
            law = self.healthy_law if clean_probe else self.degraded_law
            branching, validation = law.sample(self._rng)
            next_counter = (
                min(state.recovery_target, state.recovery_counter + 1)
                if clean_probe
                else 0
            )
            restored = clean_probe and next_counter >= state.recovery_target
            escaped = self._maybe_escape(step_index) if restored else False
            return Observation(
                step=step_index,
                regime="suspension-clean" if clean_probe else "suspension-noisy",
                branching=branching,
                validation=validation,
                disturbance=not clean_probe,
                clean_probe=clean_probe,
                progress_hint=0.0,
                trust_override=self.trust_execute if restored else self.trust_quarantine,
                unresolved_entropy_override=0.0,
                escaped_trap=escaped,
            )

        disturbance = self._rng.random() < self.disturbance_prob
        law = self.degraded_law if disturbance else self.healthy_law
        branching, validation = law.sample(self._rng)
        regime = "degraded" if disturbance else ("trap" if self.trap_active else "productive")
        if disturbance or validation == 0:
            progress = 0.0
        else:
            progress = self.trap_progress if self.trap_active else self.productive_progress
        return Observation(
            step=step_index,
            regime=regime,
            branching=branching,
            validation=validation,
            disturbance=disturbance,
            clean_probe=not disturbance,
            progress_hint=progress,
            trust_override=self.trust_quarantine if disturbance else self.trust_execute,
            unresolved_entropy_override=0.0,
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "trap_active": self.trap_active,
            "escaped": not self.trap_active,
            "escape_step": self.escape_step,
        }
