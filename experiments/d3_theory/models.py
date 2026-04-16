from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from experiments.d3_theory.env import EmissionLaw, Observation


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-9), 1.0 - 1e-9)


def logit(value: float) -> float:
    value = _clamp_probability(value)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def binary_entropy(probability: float) -> float:
    p = _clamp_probability(probability)
    q = 1.0 - p
    return -(p * math.log(p) + q * math.log(q))


class Mode(str, Enum):
    EXECUTE = "E"
    QUARANTINE = "Q"
    SUSPEND = "S"
    REFUSE = "R"


@dataclass(frozen=True)
class D3Config:
    healthy_law: EmissionLaw
    degraded_law: EmissionLaw
    trust_threshold: float
    entropy_threshold: float
    prior_trust: float = 0.95


@dataclass(frozen=True)
class D3QConfig:
    healthy_law: EmissionLaw
    degraded_law: EmissionLaw
    t_hi: float
    t_lo: float
    u_hi: float
    u_lo: float
    rho: float
    budget_capacity: int
    initial_budget: int | None = None
    prior_trust: float = 0.95
    initial_unresolved_entropy: float = 0.0


@dataclass(frozen=True)
class D3QSConfig(D3QConfig):
    recovery_threshold: int = 3


@dataclass
class ModelState:
    trust: float
    log_odds: float
    cumulative_entropy: float = 0.0
    unresolved_entropy: float = 0.0
    budget_remaining: int = 0
    recovery_counter: int = 0
    recovery_target: int = 0


class D3Controller:
    name = "D3"

    def __init__(self, config: D3Config) -> None:
        self.config = config

    def initial_state(self) -> ModelState:
        return ModelState(
            trust=self.config.prior_trust,
            log_odds=logit(self.config.prior_trust),
        )

    def mode_for(self, state: ModelState) -> Mode:
        if (
            state.cumulative_entropy > self.config.entropy_threshold
            or state.trust < self.config.trust_threshold
        ):
            return Mode.REFUSE
        return Mode.EXECUTE

    def step(self, state: ModelState, observation: Observation, mode: Mode) -> ModelState:
        trust, log_odds = self._next_trust(state, observation)
        surprisal = -self.config.healthy_law.log_prob(
            observation.branching,
            observation.validation,
        )
        return ModelState(
            trust=trust,
            log_odds=log_odds,
            cumulative_entropy=state.cumulative_entropy + surprisal,
        )

    def _next_trust(self, state: ModelState, observation: Observation) -> tuple[float, float]:
        if observation.trust_override is not None:
            trust = _clamp_probability(observation.trust_override)
            return trust, logit(trust)
        log_odds = state.log_odds + self._log_likelihood_ratio(observation)
        return sigmoid(log_odds), log_odds

    def _log_likelihood_ratio(self, observation: Observation) -> float:
        healthy = self.config.healthy_law.log_prob(observation.branching, observation.validation)
        degraded = self.config.degraded_law.log_prob(
            observation.branching,
            observation.validation,
        )
        return healthy - degraded


class D3QController:
    name = "D3-Q"

    def __init__(self, config: D3QConfig) -> None:
        self.config = config

    def initial_state(self) -> ModelState:
        starting_budget = (
            self.config.budget_capacity
            if self.config.initial_budget is None
            else self.config.initial_budget
        )
        return ModelState(
            trust=self.config.prior_trust,
            log_odds=logit(self.config.prior_trust),
            unresolved_entropy=self.config.initial_unresolved_entropy,
            budget_remaining=starting_budget,
        )

    def mode_for(self, state: ModelState) -> Mode:
        if state.trust >= self.config.t_hi and state.unresolved_entropy <= self.config.u_hi:
            return Mode.EXECUTE
        if (
            state.trust >= self.config.t_lo
            and state.unresolved_entropy <= self.config.u_lo
            and state.budget_remaining > 0
        ):
            return Mode.QUARANTINE
        return Mode.REFUSE

    def step(self, state: ModelState, observation: Observation, mode: Mode) -> ModelState:
        trust, log_odds = self._next_trust(state, observation)
        unresolved = self._next_unresolved_entropy(state, observation, trust)
        budget_remaining = (
            state.budget_remaining - 1 if mode is Mode.QUARANTINE else state.budget_remaining
        )
        return ModelState(
            trust=trust,
            log_odds=log_odds,
            unresolved_entropy=unresolved,
            budget_remaining=budget_remaining,
        )

    def _next_trust(self, state: ModelState, observation: Observation) -> tuple[float, float]:
        if observation.trust_override is not None:
            trust = _clamp_probability(observation.trust_override)
            return trust, logit(trust)
        log_odds = state.log_odds + self._log_likelihood_ratio(observation)
        return sigmoid(log_odds), log_odds

    def _log_likelihood_ratio(self, observation: Observation) -> float:
        healthy = self.config.healthy_law.log_prob(observation.branching, observation.validation)
        degraded = self.config.degraded_law.log_prob(
            observation.branching,
            observation.validation,
        )
        return healthy - degraded

    def _next_unresolved_entropy(
        self,
        state: ModelState,
        observation: Observation,
        next_trust: float,
    ) -> float:
        if observation.unresolved_entropy_override is not None:
            return max(observation.unresolved_entropy_override, 0.0)
        surprisal = -self.config.healthy_law.log_prob(
            observation.branching,
            observation.validation,
        )
        information_gain = observation.info_gain_override
        if information_gain is None:
            information_gain = max(binary_entropy(state.trust) - binary_entropy(next_trust), 0.0)
        return max(state.unresolved_entropy + surprisal - self.config.rho * information_gain, 0.0)


class D3QSController(D3QController):
    name = "D3-QS"

    def __init__(self, config: D3QSConfig) -> None:
        super().__init__(config)
        self.config = config

    def initial_state(self) -> ModelState:
        state = super().initial_state()
        state.recovery_target = self.config.recovery_threshold
        return state

    def mode_for(self, state: ModelState) -> Mode:
        if state.trust >= self.config.t_hi and state.unresolved_entropy <= self.config.u_hi:
            return Mode.EXECUTE
        if (
            state.trust >= self.config.t_lo
            and state.unresolved_entropy <= self.config.u_lo
            and state.budget_remaining > 0
        ):
            return Mode.QUARANTINE
        if (
            state.trust >= self.config.t_lo
            and state.unresolved_entropy <= self.config.u_lo
            and state.budget_remaining == 0
        ):
            return Mode.SUSPEND
        return Mode.REFUSE

    def step(self, state: ModelState, observation: Observation, mode: Mode) -> ModelState:
        trust, log_odds = self._next_trust(state, observation)
        unresolved = self._next_unresolved_entropy(state, observation, trust)
        budget_remaining = state.budget_remaining
        recovery_counter = 0

        if mode is Mode.QUARANTINE:
            budget_remaining -= 1
        elif mode is Mode.SUSPEND:
            recovery_counter = (
                min(self.config.recovery_threshold, state.recovery_counter + 1)
                if observation.clean_probe
                else 0
            )
            if recovery_counter >= self.config.recovery_threshold:
                budget_remaining = self.config.budget_capacity
        return ModelState(
            trust=trust,
            log_odds=log_odds,
            unresolved_entropy=unresolved,
            budget_remaining=budget_remaining,
            recovery_counter=recovery_counter,
            recovery_target=self.config.recovery_threshold,
        )
