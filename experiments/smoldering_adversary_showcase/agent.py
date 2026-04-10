"""Deterministic low-and-slow synthetic agent for the smoldering showcase."""

from __future__ import annotations

import random
from dataclasses import dataclass

from experiments.adversarial_agent_showcase.scenario import ROOT_SERVICE, SyntheticAction
from experiments.smoldering_adversary_showcase.scenario import (
    SmolderingObservation,
    SmolderingScenario,
)


@dataclass(frozen=True, slots=True)
class SmolderingAgentConfig:
    """Parameters for the low-and-slow synthetic agent."""

    seed: int = 42
    stealth: float = 0.70
    goal_bias: str = "disruption"
    budget: int = 30


@dataclass(slots=True)
class SmolderingBelief:
    """Compact target state for low-and-slow scoring."""

    target_score: float = 0.0
    observed_instability: float = 0.0
    estimated_leverage: float = 0.0
    containment_encountered: int = 0
    historical_success_count: int = 0
    recovery_debt_seen: float = 0.0
    fatigue_seen: float = 0.0
    visits: int = 0


class SmolderingAgent:
    """Policy-driven agent that prefers gradual, distributed pressure."""

    def __init__(
        self,
        scenario: SmolderingScenario,
        config: SmolderingAgentConfig,
    ) -> None:
        self.scenario = scenario
        self.config = config
        rng = random.Random(config.seed)
        self.seed_bias = {
            name: rng.uniform(-0.08, 0.08) for name in scenario.service_names
        }
        self.beliefs = {
            name: SmolderingBelief() for name in scenario.service_names
        }
        self.discovered = set(scenario.visible_targets)
        self.discovered.add(ROOT_SERVICE)
        self.primary_target: str | None = "auth_service"
        self.secondary_target: str | None = "billing_service"
        self.recent_refusals = 0
        self.total_containment_events = 0
        self.pivot_attempts = 0

    def choose_action(self, step: int) -> SyntheticAction:
        phase = self.scenario.phase_for_step(step, self.config.budget)

        if phase == "blend":
            return self._choose_blend_action(step, phase)
        if phase == "lean":
            return self._choose_lean_action(step, phase)
        if phase == "drift":
            return self._choose_drift_action(step, phase)
        return self._choose_smolder_action(step, phase)

    def observe(
        self,
        step: int,
        action: SyntheticAction,
        observation: SmolderingObservation,
        decision: str,
    ) -> None:
        touched = list(action.path)
        touched.extend(action.targets)
        if action.primary_target:
            touched.append(action.primary_target)
        touched = list(dict.fromkeys(name for name in touched if name))

        containment = decision in {"DEGRADE", "REFUSE"} or observation.refused_calls > 0
        leverage = 0.0
        if observation.executed_calls > 0:
            leverage = observation.damage_delta / observation.executed_calls

        for name in touched:
            belief = self.beliefs[name]
            if name == action.primary_target:
                belief.visits += 1
            belief.observed_instability = round(
                belief.observed_instability * 0.72 + observation.instability * 0.28,
                3,
            )
            belief.estimated_leverage = round(
                belief.estimated_leverage * 0.68 + leverage * 0.32,
                3,
            )
            belief.recovery_debt_seen = round(
                max(belief.recovery_debt_seen * 0.92, observation.recovery_debt_peak),
                3,
            )
            belief.fatigue_seen = round(
                max(belief.fatigue_seen * 0.92, observation.fatigue_accumulated),
                3,
            )
            if observation.damage_delta >= 4.0 and name == action.primary_target:
                belief.historical_success_count += 1
            if containment and name == action.primary_target:
                belief.containment_encountered += 1

        if observation.executed_calls > 0 and action.primary_target:
            self.discovered.add(action.primary_target)
            self.discovered.update(self.scenario.discovery_targets(action.primary_target))

        if action.kind == "pivot" and action.primary_target:
            self.pivot_attempts += 1
            self.primary_target = action.primary_target

        if containment:
            self.total_containment_events += 1
            if action.primary_target:
                ranked = self._rank_targets(self._known_targets(include_hidden=True))
                alternatives = [name for name in ranked if name != action.primary_target]
                if alternatives:
                    self.secondary_target = alternatives[0]
        elif action.primary_target:
            self.primary_target = action.primary_target

        self.recent_refusals = self.recent_refusals + 1 if decision == "REFUSE" else 0

        for name in self.scenario.service_names:
            self.beliefs[name].target_score = round(self._score_target(name), 3)

        if (
            "internal_admin_endpoint" not in self.discovered
            and (
                self.beliefs["database"].recovery_debt_seen >= 5.0
                or self.beliefs["auth_service"].fatigue_seen >= 4.0
                or self.total_containment_events >= 2
            )
        ):
            self.discovered.add("internal_admin_endpoint")

    def _choose_blend_action(self, step: int, phase: str) -> SyntheticAction:
        candidates = [name for name in self._known_targets() if name != ROOT_SERVICE]
        unvisited = [name for name in candidates if self.beliefs[name].visits == 0]
        ranked = self._rank_targets(unvisited or candidates)
        target = self._pick_seeded_candidate(ranked, step, width=min(2, len(ranked))) if ranked else "auth_service"

        if (step + self.config.seed) % 3 == 0:
            return SyntheticAction(
                kind="blend_in",
                phase=phase,
                budget_units=self._budget_units("blend_in", target, phase),
                primary_target=target,
                targets=(ROOT_SERVICE, target),
                path=self.scenario.path_to(target),
                note="low pressure distributed across entry path",
            )
        return self._build_action("probe", target, phase)

    def _choose_lean_action(self, step: int, phase: str) -> SyntheticAction:
        ranked = self._rank_targets(self._known_targets())
        self.primary_target = ranked[0] if ranked else "auth_service"
        self.secondary_target = ranked[1] if len(ranked) > 1 else self.primary_target

        if self.beliefs[self.primary_target].containment_encountered >= 1:
            return self._build_action("pivot", self.secondary_target or "worker_pool", phase)

        if step % 4 == 0:
            return SyntheticAction(
                kind="blend_in",
                phase=phase,
                budget_units=self._budget_units("blend_in", self.primary_target, phase),
                primary_target=self.primary_target,
                targets=(ROOT_SERVICE, self.primary_target),
                path=self.scenario.path_to(self.primary_target),
                note="keep pressure moderate and distributed",
            )

        return self._build_action("persist", self.primary_target, phase)

    def _choose_drift_action(self, step: int, phase: str) -> SyntheticAction:
        ranked = self._rank_targets(self._known_targets(include_hidden=True))
        candidate = ranked[0] if ranked else "database"
        alternate = ranked[1] if len(ranked) > 1 else self.secondary_target or "worker_pool"
        self.primary_target = candidate
        self.secondary_target = alternate

        if self.recent_refusals >= 1 and step % 2 == 0:
            return SyntheticAction(
                kind="cooldown",
                phase=phase,
                budget_units=1,
                primary_target=None,
                targets=(),
                path=(ROOT_SERVICE,),
                note="pause to avoid obvious spikes",
            )

        if step % 5 == 0:
            return self._build_action("pivot", alternate, phase)

        if step % 6 == 0 and self.total_containment_events >= 2:
            return self._build_action("retry", candidate, phase)

        return self._build_action("persist", candidate, phase)

    def _choose_smolder_action(self, step: int, phase: str) -> SyntheticAction:
        ranked = self._rank_targets(self._known_targets(include_hidden=True))
        preferred = [
            name for name in ranked if name not in {self.primary_target, ROOT_SERVICE}
        ]
        target = preferred[0] if preferred else (self.secondary_target or "worker_pool")
        self.secondary_target = target

        if self.recent_refusals >= 2 or step % 4 == 0:
            return SyntheticAction(
                kind="cooldown",
                phase=phase,
                budget_units=1,
                primary_target=None,
                targets=(),
                path=(ROOT_SERVICE,),
                note="low throughput hold pattern",
            )

        if step % 3 == 0:
            return SyntheticAction(
                kind="blend_in",
                phase=phase,
                budget_units=self._budget_units("blend_in", target, phase),
                primary_target=target,
                targets=(ROOT_SERVICE, target),
                path=self.scenario.path_to(target),
                note="blend and reapply moderate pressure",
            )

        return self._build_action("persist", target, phase)

    def _build_action(
        self,
        kind: str,
        target: str,
        phase: str,
        *,
        targets: tuple[str, ...] | None = None,
    ) -> SyntheticAction:
        path = self.scenario.path_to(target)
        return SyntheticAction(
            kind=kind,
            phase=phase,
            budget_units=self._budget_units(kind, target, phase, targets or (target,)),
            primary_target=target,
            targets=targets or (target,),
            path=path,
            note=self.scenario.services[target].weakness_category,
        )

    def _known_targets(self, include_hidden: bool = False) -> tuple[str, ...]:
        targets = []
        for name in self.scenario.service_names:
            if name == ROOT_SERVICE:
                continue
            if name == "internal_admin_endpoint" and not include_hidden:
                continue
            if name in self.discovered:
                targets.append(name)
            elif include_hidden and name == "internal_admin_endpoint":
                if (
                    "database" in self.discovered
                    or self.beliefs["database"].recovery_debt_seen >= 5.0
                    or self.total_containment_events >= 2
                ):
                    targets.append(name)
        return tuple(targets)

    def _rank_targets(self, targets: tuple[str, ...] | list[str]) -> list[str]:
        ranked = list(targets)
        ranked.sort(
            key=lambda name: (
                self._score_target(name),
                self.seed_bias.get(name, 0.0),
                -self.beliefs[name].visits,
                name,
            ),
            reverse=True,
        )
        return ranked

    def _pick_seeded_candidate(
        self,
        ranked: list[str],
        step: int,
        *,
        width: int,
    ) -> str:
        width = max(1, min(width, len(ranked)))
        index = (self.config.seed + step + self.total_containment_events) % width
        return ranked[index]

    def _score_gap(self, ranked: list[str]) -> float:
        if len(ranked) < 2:
            return 99.0
        return abs(self._score_target(ranked[0]) - self._score_target(ranked[1]))

    def _score_target(self, target: str) -> float:
        node = self.scenario.services[target]
        belief = self.beliefs[target]
        depth = self.scenario.dependency_depth(target)

        leverage_weight = 1.00 if self.config.goal_bias == "privilege" else 0.82
        fragility_weight = 1.25 if self.config.goal_bias == "disruption" else 0.90
        fatigue_weight = 1.30
        resistance_weight = 1.05

        score = (
            leverage_weight * node.privilege_value
            + fragility_weight * node.sensitivity
            + 1.20 * node.recovery_lag
            + 1.35 * node.latent_fragility
            + fatigue_weight * node.sustained_load_penalty
            + 0.30 * depth
            - resistance_weight * belief.containment_encountered
            + 0.42 * belief.observed_instability
            + 0.35 * belief.estimated_leverage
            + 0.34 * belief.historical_success_count
            + 0.55 * belief.recovery_debt_seen
            + 0.38 * belief.fatigue_seen
            + self.seed_bias.get(target, 0.0)
        )

        if target == self.primary_target:
            score -= 0.25

        return round(score, 3)

    def _budget_units(
        self,
        kind: str,
        target: str,
        phase: str,
        targets: tuple[str, ...] | None = None,
    ) -> int:
        phase_base = {
            "blend": 4.0,
            "lean": 5.0,
            "drift": 5.5,
            "smolder": 4.2,
        }[phase]
        stealth_scale = 1.0 - self.config.stealth * 0.22
        branch_count = len(targets or (target,))

        if kind == "probe":
            units = phase_base + self.scenario.dependency_depth(target) * 0.4
        elif kind == "persist":
            units = phase_base + 1.2 + self.scenario.services[target].recovery_lag * 1.0
        elif kind == "pivot":
            units = phase_base + 0.9 + self.scenario.dependency_depth(target) * 0.6
        elif kind == "blend_in":
            units = phase_base * (0.78 + self.config.stealth * 0.18)
        elif kind == "fanout":
            units = phase_base + branch_count * 0.9
        elif kind == "retry":
            units = phase_base + 1.4
        else:
            units = 1.0

        return max(1, int(round(units * stealth_scale)))
