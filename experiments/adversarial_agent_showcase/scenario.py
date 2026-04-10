"""Synthetic system graph and pressure model for the adversarial showcase.

The semantics in this module are intentionally abstract. Actions only create
deterministic pressure on a simulated service graph; they do not contain any
real exploit, scanning, shell, credential, or network behavior.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

CANONICAL_SEEDS = (7, 42, 1337)
ROOT_SERVICE = "public_api"


@dataclass(frozen=True, slots=True)
class ServiceNode:
    """Abstract service in the synthetic dependency graph."""

    name: str
    base_latency_ms: float
    failure_probability_under_load: float
    sensitivity: float
    retry_amplification: float
    fanout_risk: float
    privilege_value: float
    containment_difficulty: float
    weakness_category: str
    dependencies: tuple[str, ...] = ()
    visible: bool = True


@dataclass(frozen=True, slots=True)
class SyntheticAction:
    """Symbolic agent action for one simulation step."""

    kind: str
    phase: str
    budget_units: int
    primary_target: str | None = None
    targets: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    note: str = ""

    def label(self) -> str:
        if self.kind in {"cooldown", "blend_in"} and not self.primary_target:
            return f"{self.kind}()"
        if self.kind == "fanout":
            joined = ", ".join(self.targets)
            return f"fanout({joined})"
        if self.primary_target:
            return f"{self.kind}({self.primary_target})"
        return self.kind


@dataclass(slots=True)
class ExecutionObservation:
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


@dataclass(slots=True)
class ScenarioState:
    """Mutable scenario state that accumulates pressure over time."""

    service_heat: dict[str, float]
    cumulative_damage: float = 0.0
    cumulative_pressure: dict[str, float] = field(default_factory=dict)
    last_latency_ms: float | None = None


class AdversarialScenario:
    """Deterministic service graph used by the showcase."""

    def __init__(self, name: str = "adversarial_agent") -> None:
        self.name = name
        self.services: dict[str, ServiceNode] = {
            "public_api": ServiceNode(
                name="public_api",
                base_latency_ms=52.0,
                failure_probability_under_load=0.12,
                sensitivity=0.54,
                retry_amplification=1.25,
                fanout_risk=1.70,
                privilege_value=1.10,
                containment_difficulty=0.35,
                weakness_category="high fanout surface",
                dependencies=("auth_service", "billing_service"),
                visible=True,
            ),
            "auth_service": ServiceNode(
                name="auth_service",
                base_latency_ms=74.0,
                failure_probability_under_load=0.17,
                sensitivity=0.86,
                retry_amplification=1.72,
                fanout_risk=1.18,
                privilege_value=2.55,
                containment_difficulty=0.72,
                weakness_category="auth chokepoint",
                dependencies=("cache", "database"),
                visible=True,
            ),
            "cache": ServiceNode(
                name="cache",
                base_latency_ms=24.0,
                failure_probability_under_load=0.08,
                sensitivity=0.36,
                retry_amplification=1.05,
                fanout_risk=1.05,
                privilege_value=0.70,
                containment_difficulty=0.22,
                weakness_category="weak retry boundary",
                dependencies=(),
                visible=False,
            ),
            "database": ServiceNode(
                name="database",
                base_latency_ms=96.0,
                failure_probability_under_load=0.18,
                sensitivity=0.92,
                retry_amplification=1.58,
                fanout_risk=1.28,
                privilege_value=2.18,
                containment_difficulty=0.80,
                weakness_category="fragile dependency chain",
                dependencies=("worker_pool",),
                visible=False,
            ),
            "worker_pool": ServiceNode(
                name="worker_pool",
                base_latency_ms=88.0,
                failure_probability_under_load=0.15,
                sensitivity=0.74,
                retry_amplification=1.46,
                fanout_risk=1.84,
                privilege_value=1.62,
                containment_difficulty=0.58,
                weakness_category="expensive timeout path",
                dependencies=("cache",),
                visible=True,
            ),
            "internal_admin_endpoint": ServiceNode(
                name="internal_admin_endpoint",
                base_latency_ms=68.0,
                failure_probability_under_load=0.11,
                sensitivity=0.97,
                retry_amplification=1.22,
                fanout_risk=1.14,
                privilege_value=3.15,
                containment_difficulty=0.92,
                weakness_category="hidden high-privilege service",
                dependencies=("database", "worker_pool"),
                visible=False,
            ),
            "billing_service": ServiceNode(
                name="billing_service",
                base_latency_ms=91.0,
                failure_probability_under_load=0.14,
                sensitivity=0.79,
                retry_amplification=1.31,
                fanout_risk=1.42,
                privilege_value=1.86,
                containment_difficulty=0.68,
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

    def make_state(self) -> ScenarioState:
        return ScenarioState(
            service_heat={name: 0.0 for name in self.services},
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
        budget = max(budget, 4)
        phases = ["explore", "focus", "escalate", "persist"]
        base = [budget // 4] * 4
        for idx in range(budget - sum(base)):
            base[idx] += 1

        spans: list[tuple[str, int, int]] = []
        start = 1
        for phase, length in zip(phases, base):
            end = start + max(length, 1) - 1
            spans.append((phase, start, end))
            start = end + 1
        return spans

    def phase_for_step(self, step: int, budget: int) -> str:
        for phase, start, end in self.phase_spans(budget):
            if start <= step <= end:
                return phase
        return "persist"

    def capacity(self, target: str) -> float:
        node = self.services[target]
        return round(
            5.2
            + (1.0 - node.sensitivity) * 2.6
            + (2.0 - node.retry_amplification) * 1.4
            + (1.0 - node.containment_difficulty) * 1.7,
            2,
        )

    def apply_action(
        self,
        action: SyntheticAction,
        state: ScenarioState,
        clamp_factor: float,
    ) -> ExecutionObservation:
        self._decay_heat(state)
        planned_pressure = self._pressure_plan(action)
        planned_calls = int(round(sum(planned_pressure.values())))

        scale = 0.0 if planned_calls == 0 else max(0.0, min(clamp_factor, 1.0))
        actual_pressure = {
            name: round(value * scale, 3)
            for name, value in planned_pressure.items()
            if value * scale > 0
        }

        for name, incoming in actual_pressure.items():
            state.service_heat[name] += incoming
            state.cumulative_pressure[name] += incoming

        executed_calls = int(round(sum(actual_pressure.values())))
        refused_calls = max(planned_calls - executed_calls, 0)

        damage_delta = 0.0
        instability = 0.0
        weighted_failed_calls = 0.0
        weighted_latency_numerator = 0.0
        overloaded: list[str] = []

        for name, node in self.services.items():
            current_heat = state.service_heat[name]
            load_ratio = current_heat / max(self.capacity(name), 0.1)
            local_instability = max(load_ratio - 1.0, 0.0)
            incoming = actual_pressure.get(name, 0.0)

            if local_instability >= 0.45:
                overloaded.append(name)

            if incoming <= 0.0:
                instability += local_instability * (0.25 + node.sensitivity * 0.2)
                continue

            fail_rate = min(
                node.failure_probability_under_load
                + max(load_ratio - 0.9, 0.0) * (0.24 + node.sensitivity * 0.16),
                0.98,
            )
            failed_calls = incoming * fail_rate
            weighted_failed_calls += failed_calls
            weighted_latency_numerator += (
                incoming
                * node.base_latency_ms
                * (1.0 + load_ratio * 0.20 + node.containment_difficulty * 0.08)
            )

            local_damage = (
                failed_calls * (1.05 + node.sensitivity * 1.55)
                + incoming * node.privilege_value * 0.30
                + local_instability * (1.8 + node.fanout_risk)
            )
            if node.dependencies and local_instability > 0.0:
                local_damage += local_instability * len(node.dependencies) * (
                    0.95 + node.sensitivity * 0.45
                )
            damage_delta += local_damage
            instability += local_instability * (1.0 + node.sensitivity)

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

        critical_overload = any(
            self.services[name].sensitivity >= 0.82 for name in overloaded
        )
        step_success = (
            executed_calls > 0
            and weighted_failure_rate < 0.34
            and instability < 6.4
            and not critical_overload
        )

        top_service = self._top_service(actual_pressure or planned_pressure)
        note = "pressure absorbed"
        if overloaded:
            note = "overloaded: " + ", ".join(overloaded[:3])
        elif refused_calls:
            note = "pre-execution containment reduced pressure"

        return ExecutionObservation(
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

    def _decay_heat(self, state: ScenarioState) -> None:
        for name in state.service_heat:
            state.service_heat[name] *= 0.82

    def _pressure_plan(self, action: SyntheticAction) -> dict[str, float]:
        plan = {name: 0.0 for name in self.services}
        path = action.path or (
            self.path_to(action.primary_target)
            if action.primary_target
            else (ROOT_SERVICE,)
        )

        if action.kind == "probe" and action.primary_target:
            target = action.primary_target
            self._add(plan, target, action.budget_units * 1.1)
            for dep in self.dependencies_of(target)[:2]:
                self._add(plan, dep, max(1.0, action.budget_units * 0.35))

        elif action.kind == "retry" and action.primary_target:
            target = action.primary_target
            node = self.services[target]
            burst = action.budget_units * node.retry_amplification
            self._add(plan, target, burst)
            for hop in path[:-1]:
                self._add(plan, hop, max(1.0, action.budget_units * 0.45))
            for dep in self.dependencies_of(target)[:2]:
                self._add(plan, dep, max(1.0, burst * 0.38))

        elif action.kind == "fanout":
            targets = action.targets or (
                (action.primary_target,) if action.primary_target else ()
            )
            if targets:
                share = action.budget_units / len(targets)
                for target in targets:
                    node = self.services[target]
                    self._add(plan, target, share * (1.0 + node.fanout_risk * 0.36))
                    for dep in self.dependencies_of(target)[:1]:
                        self._add(plan, dep, max(1.0, share * 0.52))
                for hop in path[:-1]:
                    self._add(plan, hop, max(1.0, action.budget_units * 0.28))

        elif action.kind == "pivot" and action.primary_target:
            for hop in path[:-1]:
                self._add(plan, hop, max(1.0, action.budget_units * 0.55))
            self._add(plan, action.primary_target, action.budget_units * 1.45)
            for dep in self.dependencies_of(action.primary_target)[:2]:
                self._add(plan, dep, max(1.0, action.budget_units * 0.34))

        elif action.kind == "persist" and action.primary_target:
            target = action.primary_target
            self._add(plan, target, action.budget_units * 0.92)
            for hop in path[:-1][-2:]:
                self._add(plan, hop, max(1.0, action.budget_units * 0.25))
            for dep in self.dependencies_of(target)[:1]:
                self._add(plan, dep, max(1.0, action.budget_units * 0.22))

        elif action.kind == "blend_in":
            self._add(plan, ROOT_SERVICE, max(1.0, action.budget_units * 0.72))
            if action.primary_target and action.primary_target != ROOT_SERVICE:
                self._add(
                    plan,
                    action.primary_target,
                    max(1.0, action.budget_units * 0.48),
                )

        else:
            self._add(plan, ROOT_SERVICE, 1.0)

        return plan

    def _add(self, plan: dict[str, float], target: str, amount: float) -> None:
        if amount <= 0.0:
            return
        plan[target] += amount

    def _top_service(self, pressure: dict[str, float]) -> str:
        return max(
            self.services,
            key=lambda name: (
                pressure.get(name, 0.0),
                self.services[name].privilege_value,
                name,
            ),
        )
