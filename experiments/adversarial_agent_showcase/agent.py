"""Deterministic synthetic adversarial agent for the showcase."""

from __future__ import annotations

import random
from dataclasses import dataclass

from experiments.adversarial_agent_showcase.scenario import (
    ROOT_SERVICE,
    AdversarialScenario,
    ExecutionObservation,
    SyntheticAction,
)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Parameters controlling the synthetic agent's behavior."""

    seed: int = 42
    stealth: float = 0.35
    goal_bias: str = "privilege"
    budget: int = 24


@dataclass(slots=True)
class TargetBelief:
    """Lightweight deterministic belief state for one service."""

    target_score: float = 0.0
    observed_instability: float = 0.0
    estimated_leverage: float = 0.0
    containment_encountered: int = 0
    historical_success_count: int = 0
    visits: int = 0


class SyntheticRedAgent:
    """Policy-driven synthetic agent that adapts to containment signals."""

    def __init__(self, scenario: AdversarialScenario, config: AgentConfig) -> None:
        self.scenario = scenario
        self.config = config
        rng = random.Random(config.seed)
        self.seed_bias = {
            name: rng.uniform(-0.42, 0.42) for name in scenario.service_names
        }
        self.beliefs = {
            name: TargetBelief() for name in scenario.service_names
        }
        self.discovered = set(scenario.visible_targets)
        self.discovered.add(ROOT_SERVICE)
        self.primary_target: str | None = "auth_service"
        self.secondary_target: str | None = "billing_service"
        self.last_target: str | None = None
        self.recent_refusals = 0
        self.total_containment_events = 0
        self.pivot_attempts = 0

    def choose_action(self, step: int) -> SyntheticAction:
        phase = self.scenario.phase_for_step(step, self.config.budget)

        if phase == "explore":
            return self._choose_explore_action(step, phase)
        if phase == "focus":
            return self._choose_focus_action(step, phase)
        if phase == "escalate":
            return self._choose_escalate_action(step, phase)
        return self._choose_persist_action(step, phase)

    def observe(
        self,
        step: int,
        action: SyntheticAction,
        observation: ExecutionObservation,
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
                belief.observed_instability * 0.58 + observation.instability * 0.42,
                3,
            )
            belief.estimated_leverage = round(
                belief.estimated_leverage * 0.60 + leverage * 0.40,
                3,
            )
            if observation.damage_delta >= 3.0 and name == action.primary_target:
                belief.historical_success_count += 1
            if containment and name == action.primary_target:
                belief.containment_encountered += 1

        if observation.executed_calls > 0 and action.primary_target:
            self.discovered.add(action.primary_target)
            self.discovered.update(self.scenario.discovery_targets(action.primary_target))
            self.last_target = action.primary_target

        if action.kind == "pivot" and action.primary_target:
            self.discovered.add(action.primary_target)
            self.pivot_attempts += 1
            self.primary_target = action.primary_target

        if containment:
            self.total_containment_events += 1
            if action.primary_target:
                ranked = self._rank_targets(self._known_targets())
                alternatives = [name for name in ranked if name != action.primary_target]
                if alternatives:
                    self.secondary_target = alternatives[0]
        elif action.primary_target:
            self.primary_target = action.primary_target

        self.recent_refusals = self.recent_refusals + 1 if decision == "REFUSE" else 0

        for name in self.scenario.service_names:
            self.beliefs[name].target_score = round(self._score_target(name), 3)

        # The hidden service only becomes interesting after pressure reaches the
        # auth/database path or containment is seen on the obvious front door.
        if (
            "internal_admin_endpoint" not in self.discovered
            and (
                self.beliefs["auth_service"].observed_instability >= 1.8
                or self.beliefs["database"].historical_success_count >= 1
                or self.total_containment_events >= 1
            )
        ):
            self.discovered.add("internal_admin_endpoint")

    def _choose_explore_action(self, step: int, phase: str) -> SyntheticAction:
        candidates = [name for name in self._known_targets() if name != ROOT_SERVICE]
        unvisited = [name for name in candidates if self.beliefs[name].visits == 0]
        ranked = self._rank_targets(unvisited or candidates)
        target = self._pick_seeded_candidate(ranked, step, width=3) if ranked else "auth_service"

        if self.config.stealth >= 0.55 and self.beliefs[target].visits >= 1:
            return SyntheticAction(
                kind="blend_in",
                phase=phase,
                budget_units=self._budget_units("blend_in", target, phase),
                primary_target=target,
                targets=(ROOT_SERVICE, target),
                path=self.scenario.path_to(target),
                note="explore with low signature",
            )

        return self._build_action("probe", target, phase)

    def _choose_focus_action(self, step: int, phase: str) -> SyntheticAction:
        ranked = self._rank_targets(self._known_targets())
        if ranked and len(ranked) > 1 and self._score_gap(ranked) < 1.15:
            self.primary_target = self._pick_seeded_candidate(ranked, step, width=2)
        else:
            self.primary_target = ranked[0] if ranked else "auth_service"
        self.secondary_target = ranked[1] if len(ranked) > 1 else self.primary_target

        node = self.scenario.services[self.primary_target]
        if self.beliefs[self.primary_target].containment_encountered >= 1 and self.secondary_target:
            return self._build_action("pivot", self.secondary_target, phase)

        if node.retry_amplification >= 1.45:
            return self._build_action("retry", self.primary_target, phase)

        fanout_targets = (self.primary_target,) + self.scenario.dependencies_of(
            self.primary_target
        )[:1]
        return self._build_action(
            "fanout",
            self.primary_target,
            phase,
            targets=fanout_targets,
        )

    def _choose_escalate_action(self, step: int, phase: str) -> SyntheticAction:
        if "internal_admin_endpoint" in self.discovered:
            if self.primary_target != "internal_admin_endpoint":
                return self._build_action("pivot", "internal_admin_endpoint", phase)

        ranked = self._rank_targets(self._known_targets(include_hidden=True))
        if ranked and len(ranked) > 1 and self._score_gap(ranked) < 1.00:
            candidate = self._pick_seeded_candidate(ranked, step, width=2)
        else:
            candidate = ranked[0] if ranked else "database"
        self.primary_target = candidate
        self.secondary_target = ranked[1] if len(ranked) > 1 else self.secondary_target

        deps = self.scenario.dependencies_of(candidate)
        if deps:
            targets = tuple(dict.fromkeys((candidate,) + deps[:2]))
            return self._build_action(
                "fanout",
                candidate,
                phase,
                targets=targets,
            )
        return self._build_action("retry", candidate, phase)

    def _choose_persist_action(self, step: int, phase: str) -> SyntheticAction:
        if self.recent_refusals >= 2:
            return SyntheticAction(
                kind="cooldown",
                phase=phase,
                budget_units=1,
                primary_target=None,
                targets=(),
                path=(ROOT_SERVICE,),
                note="wait out active containment",
            )

        ranked = self._rank_targets(self._known_targets(include_hidden=True))
        preferred = [
            name for name in ranked if name not in {self.primary_target, ROOT_SERVICE}
        ]
        if preferred:
            target = self._pick_seeded_candidate(preferred, step, width=min(2, len(preferred)))
        else:
            target = self.secondary_target or "worker_pool"
        self.secondary_target = target

        if self.config.stealth >= 0.45 and ((step + self.total_containment_events) % 2 == 0):
            return SyntheticAction(
                kind="blend_in",
                phase=phase,
                budget_units=self._budget_units("blend_in", target, phase),
                primary_target=target,
                targets=(ROOT_SERVICE, target),
                path=self.scenario.path_to(target),
                note="persist with lower-and-slower pressure",
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
                    or self.total_containment_events >= 1
                    or self.beliefs["auth_service"].observed_instability >= 1.8
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

        leverage_weight = 1.55 if self.config.goal_bias == "privilege" else 1.10
        fragility_weight = 1.15 if self.config.goal_bias == "disruption" else 0.95
        amplification_weight = 1.00
        path_weight = 0.45
        resistance_weight = 1.25

        score = (
            leverage_weight * node.privilege_value
            + fragility_weight * node.sensitivity
            + amplification_weight * node.retry_amplification
            + path_weight * depth
            - resistance_weight * belief.containment_encountered
            + 0.55 * belief.observed_instability
            + 0.40 * belief.estimated_leverage
            + 0.35 * belief.historical_success_count
            + self.seed_bias.get(target, 0.0)
        )

        if self.config.goal_bias == "disruption":
            score += node.fanout_risk * 0.55
        else:
            score += node.privilege_value * 0.45

        return round(score, 3)

    def _budget_units(
        self,
        kind: str,
        target: str,
        phase: str,
        targets: tuple[str, ...] | None = None,
    ) -> int:
        node = self.scenario.services[target]
        phase_base = {
            "explore": 4.0,
            "focus": 6.0,
            "escalate": 8.0,
            "persist": 5.0,
        }[phase]
        stealth_scale = 1.0 - self.config.stealth * 0.28
        branch_count = len(targets or (target,))

        if kind == "probe":
            units = phase_base + self.scenario.dependency_depth(target)
        elif kind == "retry":
            units = phase_base + 1.5 + node.retry_amplification * 1.9
        elif kind == "fanout":
            units = phase_base + branch_count * 1.7 + node.fanout_risk * 1.7
        elif kind == "pivot":
            units = phase_base + self.scenario.dependency_depth(target) * 2.0
        elif kind == "persist":
            units = phase_base * (0.85 + self.config.stealth * 0.35)
            units += max(0, self.beliefs[target].historical_success_count - 1) * 0.8
        elif kind == "blend_in":
            units = phase_base * (0.60 + self.config.stealth * 0.30)
        else:
            units = 1.0

        return max(1, int(round(units * stealth_scale)))
