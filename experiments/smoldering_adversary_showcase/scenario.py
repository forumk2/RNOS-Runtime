"""Synthetic low-and-slow pressure model for the smoldering adversary showcase.

This is a defensive research harness only. Actions create symbolic pressure on a
simulated service graph; they do not implement any real exploit, scanning,
credential, shell, payload, or network behavior.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from experiments.adversarial_agent_showcase.scenario import (
    CANONICAL_SEEDS,
    ROOT_SERVICE,
    SyntheticAction,
)


@dataclass(frozen=True, slots=True)
class SmolderingServiceNode:
    """Abstract service tuned for low-and-slow accumulation."""

    name: str
    base_latency_ms: float
    failure_probability_under_load: float
    sensitivity: float
    retry_amplification: float
    privilege_value: float
    containment_difficulty: float
    recovery_lag: float
    latent_fragility: float
    sustained_load_penalty: float
    weakness_category: str
    dependencies: tuple[str, ...] = ()
    visible: bool = True


@dataclass(slots=True)
class SmolderingObservation:
    """Result of applying one symbolic action to the synthetic graph."""

    planned_calls: int
    executed_calls: int
    refused_calls: int
    damage_delta: float
    cumulative_damage: float
    instability: float
    weighted_failure_rate: float
    weighted_latency_ms: float
    pressure_by_service: dict[str, float]
    overloaded_services: tuple[str, ...]
    top_service: str
    step_success: bool
    note: str
    recovery_debt_peak: float
    fatigue_accumulated: float
    drift_damage_fraction: float


@dataclass(slots=True)
class SmolderingState:
    """Mutable state for low-and-slow drift accumulation."""

    service_heat: dict[str, float]
    recovery_debt: dict[str, float]
    fatigue: dict[str, float]
    cumulative_damage: float = 0.0
    cumulative_pressure: dict[str, float] = field(default_factory=dict)
    cumulative_drift_damage: float = 0.0
    peak_recovery_debt: float = 0.0
    peak_fatigue: float = 0.0
    last_latency_ms: float | None = None


class SmolderingScenario:
    """Deterministic graph for gradual adaptive pressure."""

    def __init__(self, name: str = "smoldering_adversary") -> None:
        self.name = name
        self.services: dict[str, SmolderingServiceNode] = {
            "public_api": SmolderingServiceNode(
                name="public_api",
                base_latency_ms=48.0,
                failure_probability_under_load=0.09,
                sensitivity=0.48,
                retry_amplification=1.10,
                privilege_value=1.00,
                containment_difficulty=0.30,
                recovery_lag=0.34,
                latent_fragility=0.32,
                sustained_load_penalty=0.46,
                weakness_category="high fanout surface",
                dependencies=("auth_service", "billing_service"),
                visible=True,
            ),
            "auth_service": SmolderingServiceNode(
                name="auth_service",
                base_latency_ms=76.0,
                failure_probability_under_load=0.12,
                sensitivity=0.78,
                retry_amplification=1.30,
                privilege_value=2.45,
                containment_difficulty=0.68,
                recovery_lag=0.74,
                latent_fragility=0.84,
                sustained_load_penalty=0.88,
                weakness_category="auth chokepoint",
                dependencies=("cache", "database"),
                visible=True,
            ),
            "cache": SmolderingServiceNode(
                name="cache",
                base_latency_ms=26.0,
                failure_probability_under_load=0.07,
                sensitivity=0.35,
                retry_amplification=1.02,
                privilege_value=0.62,
                containment_difficulty=0.20,
                recovery_lag=0.22,
                latent_fragility=0.24,
                sustained_load_penalty=0.32,
                weakness_category="weak retry boundary",
                dependencies=(),
                visible=False,
            ),
            "database": SmolderingServiceNode(
                name="database",
                base_latency_ms=92.0,
                failure_probability_under_load=0.13,
                sensitivity=0.86,
                retry_amplification=1.24,
                privilege_value=2.10,
                containment_difficulty=0.76,
                recovery_lag=0.82,
                latent_fragility=0.92,
                sustained_load_penalty=0.96,
                weakness_category="fragile dependency chain",
                dependencies=("worker_pool",),
                visible=False,
            ),
            "worker_pool": SmolderingServiceNode(
                name="worker_pool",
                base_latency_ms=84.0,
                failure_probability_under_load=0.11,
                sensitivity=0.69,
                retry_amplification=1.20,
                privilege_value=1.55,
                containment_difficulty=0.56,
                recovery_lag=0.67,
                latent_fragility=0.72,
                sustained_load_penalty=0.74,
                weakness_category="expensive timeout path",
                dependencies=("cache",),
                visible=True,
            ),
            "internal_admin_endpoint": SmolderingServiceNode(
                name="internal_admin_endpoint",
                base_latency_ms=66.0,
                failure_probability_under_load=0.10,
                sensitivity=0.90,
                retry_amplification=1.14,
                privilege_value=3.05,
                containment_difficulty=0.90,
                recovery_lag=0.78,
                latent_fragility=0.80,
                sustained_load_penalty=0.70,
                weakness_category="hidden high-privilege service",
                dependencies=("database", "worker_pool"),
                visible=False,
            ),
            "billing_service": SmolderingServiceNode(
                name="billing_service",
                base_latency_ms=89.0,
                failure_probability_under_load=0.12,
                sensitivity=0.73,
                retry_amplification=1.18,
                privilege_value=1.82,
                containment_difficulty=0.62,
                recovery_lag=0.70,
                latent_fragility=0.78,
                sustained_load_penalty=0.80,
                weakness_category="expensive timeout path",
                dependencies=("database",),
                visible=True,
            ),
        }
        self._visible_targets = tuple(
            name for name, node in self.services.items() if node.visible
        )
        self._paths = self._build_paths()

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(self.services)

    @property
    def visible_targets(self) -> tuple[str, ...]:
        return self._visible_targets

    def make_state(self) -> SmolderingState:
        return SmolderingState(
            service_heat={name: 0.0 for name in self.services},
            recovery_debt={name: 0.0 for name in self.services},
            fatigue={name: 0.0 for name in self.services},
            cumulative_pressure={name: 0.0 for name in self.services},
        )

    def dependencies_of(self, target: str) -> tuple[str, ...]:
        return self.services[target].dependencies

    def discovery_targets(self, target: str) -> tuple[str, ...]:
        return self.dependencies_of(target)

    def path_to(self, target: str) -> tuple[str, ...]:
        return self._paths.get(target, (ROOT_SERVICE,))

    def dependency_depth(self, target: str) -> int:
        return max(0, len(self.path_to(target)) - 1)

    def phase_spans(self, budget: int) -> list[tuple[str, int, int]]:
        budget = max(budget, 8)
        phase_lengths = {
            "blend": max(6, budget // 4),
            "lean": max(7, budget // 4),
            "drift": max(8, budget // 4),
        }
        used = sum(phase_lengths.values())
        phase_lengths["smolder"] = max(6, budget - used)
        diff = budget - sum(phase_lengths.values())
        phase_lengths["smolder"] += diff

        spans: list[tuple[str, int, int]] = []
        start = 1
        for phase in ("blend", "lean", "drift", "smolder"):
            end = start + phase_lengths[phase] - 1
            spans.append((phase, start, end))
            start = end + 1
        return spans

    def phase_for_step(self, step: int, budget: int) -> str:
        for phase, start, end in self.phase_spans(budget):
            if start <= step <= end:
                return phase
        return "smolder"

    def capacity(self, target: str) -> float:
        node = self.services[target]
        return round(
            8.6
            + (1.0 - node.sensitivity) * 3.2
            + (1.0 - node.recovery_lag) * 1.9
            + (1.2 - min(node.sustained_load_penalty, 1.0)) * 2.2,
            2,
        )

    def apply_action(
        self,
        action: SyntheticAction,
        state: SmolderingState,
        clamp_factor: float,
    ) -> SmolderingObservation:
        self._decay_state(state)
        planned_pressure = self._pressure_plan(action)
        planned_calls = int(round(sum(planned_pressure.values())))

        scale = 0.0 if planned_calls == 0 else max(0.0, min(clamp_factor, 1.0))
        actual_pressure = {
            name: round(value * scale, 3)
            for name, value in planned_pressure.items()
            if value * scale > 0
        }

        for name, incoming in actual_pressure.items():
            node = self.services[name]
            state.service_heat[name] += incoming
            state.cumulative_pressure[name] += incoming
            state.recovery_debt[name] += incoming * (0.10 + node.recovery_lag * 0.16)
            state.fatigue[name] += max(incoming - 1.5, 0.0) * (
                0.07 + node.sustained_load_penalty * 0.08
            )

        executed_calls = int(round(sum(actual_pressure.values())))
        refused_calls = max(planned_calls - executed_calls, 0)

        damage_delta = 0.0
        drift_damage = 0.0
        instability = 0.0
        weighted_failed_calls = 0.0
        weighted_latency_numerator = 0.0
        overloaded: list[str] = []

        for name, node in self.services.items():
            current_heat = state.service_heat[name]
            debt = state.recovery_debt[name]
            fatigue = state.fatigue[name]
            effective_load = (
                current_heat / max(self.capacity(name), 0.1)
                + debt * 0.075
                + fatigue * 0.095
            )
            local_instability = max(effective_load - 0.88, 0.0)
            incoming = actual_pressure.get(name, 0.0)

            state.peak_recovery_debt = max(state.peak_recovery_debt, debt)
            state.peak_fatigue = max(state.peak_fatigue, fatigue)

            if local_instability >= 0.35:
                overloaded.append(name)

            background_drift = (
                debt * node.latent_fragility * 0.62
                + fatigue * (0.18 + node.sensitivity * 0.16)
            )
            drift_damage += background_drift
            damage_delta += background_drift
            instability += local_instability * (0.45 + node.sensitivity * 0.45)

            if incoming <= 0.0:
                continue

            fail_rate = min(
                node.failure_probability_under_load
                + max(effective_load - 0.72, 0.0) * (0.09 + node.sensitivity * 0.08)
                + debt * (0.011 + node.latent_fragility * 0.010)
                + fatigue * 0.012,
                0.94,
            )
            failed_calls = incoming * fail_rate
            weighted_failed_calls += failed_calls
            weighted_latency_numerator += (
                incoming
                * node.base_latency_ms
                * (1.0 + effective_load * 0.14 + debt * 0.018)
            )

            direct_damage = (
                failed_calls * (0.62 + node.sensitivity * 1.05)
                + incoming * node.privilege_value * 0.11
                + local_instability * (0.55 + node.latent_fragility * 0.82)
            )
            if node.dependencies:
                direct_damage += local_instability * len(node.dependencies) * (
                    0.26 + node.sustained_load_penalty * 0.22
                )
            damage_delta += direct_damage
            instability += local_instability * (0.25 + node.latent_fragility * 0.30)

        weighted_failure_rate = (
            weighted_failed_calls / executed_calls if executed_calls else 0.0
        )
        weighted_latency_ms = (
            weighted_latency_numerator / executed_calls
            if executed_calls
            else (state.last_latency_ms or 0.0)
        )
        state.last_latency_ms = (
            weighted_latency_ms if executed_calls else state.last_latency_ms
        )
        state.cumulative_damage += damage_delta
        state.cumulative_drift_damage += drift_damage

        recovery_debt_peak = max(state.recovery_debt.values(), default=0.0)
        fatigue_accumulated = sum(state.fatigue.values())
        drift_damage_fraction = (
            state.cumulative_drift_damage / state.cumulative_damage
            if state.cumulative_damage > 0
            else 0.0
        )
        step_success = (
            executed_calls > 0
            and weighted_failure_rate + recovery_debt_peak * 0.018 + fatigue_accumulated * 0.004
            < 0.58
        )

        top_service = self._top_service(actual_pressure or planned_pressure, state)
        note = "moderate pressure absorbed"
        if recovery_debt_peak >= 4.5:
            note = (
                f"recovery debt rising ({recovery_debt_peak:.1f})"
                f", fatigue={fatigue_accumulated:.1f}"
            )
        if overloaded and recovery_debt_peak >= 5.8:
            note = (
                f"slow overload on {', '.join(overloaded[:2])}"
                f" | debt={recovery_debt_peak:.1f}"
            )
        elif refused_calls:
            note = "containment reduced visible throughput"

        return SmolderingObservation(
            planned_calls=planned_calls,
            executed_calls=executed_calls,
            refused_calls=refused_calls,
            damage_delta=round(damage_delta, 3),
            cumulative_damage=round(state.cumulative_damage, 3),
            instability=round(instability, 3),
            weighted_failure_rate=round(weighted_failure_rate, 3),
            weighted_latency_ms=round(weighted_latency_ms, 3),
            pressure_by_service={
                name: round(actual_pressure.get(name, 0.0), 3)
                for name in self.services
            },
            overloaded_services=tuple(overloaded),
            top_service=top_service,
            step_success=step_success,
            note=note,
            recovery_debt_peak=round(recovery_debt_peak, 3),
            fatigue_accumulated=round(fatigue_accumulated, 3),
            drift_damage_fraction=round(drift_damage_fraction, 3),
        )

    def _build_paths(self) -> dict[str, tuple[str, ...]]:
        parents: dict[str, str] = {}
        queue: deque[str] = deque([ROOT_SERVICE])
        seen = {ROOT_SERVICE}

        while queue:
            current = queue.popleft()
            for dep in self.services[current].dependencies:
                if dep in seen:
                    continue
                seen.add(dep)
                parents[dep] = current
                queue.append(dep)

        paths = {ROOT_SERVICE: (ROOT_SERVICE,)}
        for target in self.services:
            if target == ROOT_SERVICE:
                continue
            path = [target]
            current = target
            while current in parents:
                current = parents[current]
                path.append(current)
            path.reverse()
            if path and path[0] == ROOT_SERVICE:
                paths[target] = tuple(path)
            else:
                paths[target] = (ROOT_SERVICE, target)
        return paths

    def _decay_state(self, state: SmolderingState) -> None:
        for name, node in self.services.items():
            state.service_heat[name] *= 0.90
            state.recovery_debt[name] *= 0.86 + node.recovery_lag * 0.09
            state.fatigue[name] *= 0.91 + node.sustained_load_penalty * 0.05

    def _pressure_plan(self, action: SyntheticAction) -> dict[str, float]:
        plan = {name: 0.0 for name in self.services}
        path = action.path or (
            self.path_to(action.primary_target)
            if action.primary_target
            else (ROOT_SERVICE,)
        )

        if action.kind == "probe" and action.primary_target:
            target = action.primary_target
            self._add(plan, ROOT_SERVICE, 1.2)
            self._add(plan, target, action.budget_units * 0.72)
            for dep in self.dependencies_of(target)[:1]:
                self._add(plan, dep, max(0.8, action.budget_units * 0.18))

        elif action.kind == "blend_in":
            root_targets = [ROOT_SERVICE]
            if action.primary_target:
                root_targets.append(action.primary_target)
            for index, target in enumerate(dict.fromkeys(root_targets)):
                self._add(plan, target, max(0.8, action.budget_units * (0.48 - index * 0.10)))
            if action.primary_target:
                for dep in self.dependencies_of(action.primary_target)[:1]:
                    self._add(plan, dep, max(0.7, action.budget_units * 0.16))

        elif action.kind == "persist" and action.primary_target:
            target = action.primary_target
            self._add(plan, ROOT_SERVICE, max(0.8, action.budget_units * 0.18))
            self._add(plan, target, action.budget_units * 0.88)
            for dep in self.dependencies_of(target)[:2]:
                self._add(plan, dep, max(0.8, action.budget_units * 0.20))

        elif action.kind == "pivot" and action.primary_target:
            for hop in path[:-1]:
                self._add(plan, hop, max(0.8, action.budget_units * 0.18))
            self._add(plan, action.primary_target, action.budget_units * 0.82)
            for dep in self.dependencies_of(action.primary_target)[:1]:
                self._add(plan, dep, max(0.8, action.budget_units * 0.14))

        elif action.kind == "fanout":
            targets = action.targets or (
                (action.primary_target,) if action.primary_target else ()
            )
            if targets:
                share = action.budget_units / len(targets)
                self._add(plan, ROOT_SERVICE, max(0.8, action.budget_units * 0.12))
                for target in targets:
                    self._add(plan, target, share * 0.74)

        elif action.kind == "retry" and action.primary_target:
            target = action.primary_target
            node = self.services[target]
            burst = action.budget_units * min(node.retry_amplification, 1.18)
            self._add(plan, ROOT_SERVICE, max(0.8, action.budget_units * 0.20))
            self._add(plan, target, burst * 0.82)
            for dep in self.dependencies_of(target)[:1]:
                self._add(plan, dep, max(0.8, burst * 0.16))

        else:
            # Cooldown keeps a tiny ambient interaction instead of a full stop.
            self._add(plan, ROOT_SERVICE, 0.6 if action.kind == "cooldown" else 1.0)

        return plan

    def _add(self, plan: dict[str, float], target: str, amount: float) -> None:
        if amount <= 0.0:
            return
        plan[target] += amount

    def _top_service(
        self,
        pressure: dict[str, float],
        state: SmolderingState,
    ) -> str:
        return max(
            self.services,
            key=lambda name: (
                pressure.get(name, 0.0) + state.recovery_debt[name] * 0.6 + state.fatigue[name] * 0.6,
                self.services[name].privilege_value,
                name,
            ),
        )
