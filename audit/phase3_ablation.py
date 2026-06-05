"""Phase 3 — Component ablation + perturbation harness.

Runs all RNOS variants vs tuned CB across fresh seeds and perturbation scenarios.
Outputs raw_results.jsonl for analysis.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from experiments.adversarial_agent_showcase.agent import AgentConfig, SyntheticRedAgent
from experiments.adversarial_agent_showcase.scenario import (
    AdversarialScenario,
    ExecutionObservation,
    ScenarioState,
    SyntheticAction,
)
from rnos.entropy import calculate_entropy
from rnos.policy import PolicyConfig, evaluate_policy
from rnos.runtime import RNOSRuntime
from rnos.trust import calculate_trust
from rnos.types import ActionRecord

# ---------------------------------------------------------------------------
# Frozen tuned CB config (from Phase 2)
# ---------------------------------------------------------------------------
TUNED_CB_WINDOW = 3
TUNED_CB_THRESHOLD = 0.40
TUNED_CB_COOLDOWN = 3

# ---------------------------------------------------------------------------
# Policy configs
# ---------------------------------------------------------------------------
# Showcase policy (thresholds set high, trust disabled)
SHOWCASE_POLICY = PolicyConfig(
    degrade_entropy=8.4,
    refuse_entropy=10.2,
    degrade_trust=-0.1,
    refuse_trust=-0.1,
)
# Tuned RNOS (default.yaml values — more aggressive)
TUNED_RNOS_POLICY = PolicyConfig(
    degrade_entropy=4.5,
    refuse_entropy=7.0,
    degrade_trust=-0.1,  # still trust-disabled for fair comparison
    refuse_trust=-0.1,
)
# Aggressive RNOS (aggressive.yaml values)
AGGRESSIVE_RNOS_POLICY = PolicyConfig(
    degrade_entropy=3.5,
    refuse_entropy=6.0,
    degrade_trust=-0.1,
    refuse_trust=-0.1,
)
# Trust-enabled RNOS (for trust ablation)
TRUST_ENABLED_POLICY = PolicyConfig(
    degrade_entropy=8.4,
    refuse_entropy=10.2,
    degrade_trust=0.45,
    refuse_trust=0.20,
)
# Shifted +25% thresholds (Perturbation B)
SHIFTED_RNOS_POLICY = PolicyConfig(
    degrade_entropy=10.5,
    refuse_entropy=12.75,
    degrade_trust=-0.1,
    refuse_trust=-0.1,
)

# ---------------------------------------------------------------------------
# Fresh seeds (not in canonical {7, 42, 1337})
# ---------------------------------------------------------------------------
FRESH_SEEDS = [100, 200, 300, 400, 500]
CANONICAL_SEEDS = [7, 42, 1337]
ALL_SEEDS = FRESH_SEEDS + CANONICAL_SEEDS

TRACE_PATH = REPO_ROOT / "audit" / "_phase3_trace.jsonl"
TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPO_ROOT / "audit" / "raw_results.jsonl"


def _make_runtime(policy: PolicyConfig) -> RNOSRuntime:
    TRACE_PATH.write_text("", encoding="utf-8")
    return RNOSRuntime(trace_path=TRACE_PATH, policy_config=policy)


def _make_tuned_cb() -> AdaptiveCircuitBreaker:
    return AdaptiveCircuitBreaker(
        window_size=TUNED_CB_WINDOW,
        initial_failure_rate=TUNED_CB_THRESHOLD,
        initial_cooldown_steps=TUNED_CB_COOLDOWN,
        max_cooldown_steps=TUNED_CB_COOLDOWN * 4,
        max_total_blocked=200,
    )


def _make_showcase_cb() -> AdaptiveCircuitBreaker:
    """CB as configured in the original showcase (baseline CB)."""
    return AdaptiveCircuitBreaker(
        window_size=5,
        initial_failure_rate=0.60,
        initial_cooldown_steps=2,
        max_cooldown_steps=10,
        max_total_blocked=200,
    )


def _build_action_record(
    action: SyntheticAction,
    executed_calls: int,
    prev_latency: float | None,
    retry_streak: int,
) -> ActionRecord:
    return ActionRecord(
        tool_name=action.primary_target or "public_api",
        payload={"action": action.kind, "phase": action.phase},
        depth=max(0, len(action.path) - 1),
        retry_count=retry_streak,
        latency_ms=prev_latency,
        cumulative_calls=executed_calls,
    )


# ---------------------------------------------------------------------------
# Entropy variants for rnos_minus_* ablations
# ---------------------------------------------------------------------------

def _entropy_minus_failure_and_repeated(history, candidate: ActionRecord) -> float:
    """rnos_minus_hysteresis: remove failure_score and repeated_tool."""
    depth_score = min(candidate.depth * 0.6, 4.0)
    retry_score = min(candidate.retry_count * 1.0, 4.0)
    latency_score = 0.0
    if candidate.latency_ms is not None:
        latency_score = min((candidate.latency_ms / 1000.0) * 0.5, 2.0)
    cost_score = min(candidate.cumulative_calls * 0.3, 2.0)
    return round(depth_score + retry_score + latency_score + cost_score, 3)


def _entropy_zero(_history, _candidate: ActionRecord) -> float:
    """rnos_minus_entropy: entropy is always 0."""
    return 0.0


def _evaluate_policy_custom(
    entropy_fn,
    history,
    action: ActionRecord,
    policy: PolicyConfig,
) -> tuple[str, float, float]:
    """Return (decision, entropy, trust) using a custom entropy function."""
    entropy = entropy_fn(history, action)
    trust = calculate_trust(history, entropy)
    assessment = evaluate_policy(entropy, trust, policy)
    return assessment.decision.value.upper(), entropy, trust


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    variant: str
    seed: int
    budget: int = 24
    stealth: float = 0.35
    goal_bias: str = "privilege"
    scenario_name: str = "adversarial"
    budget_units_scale: float = 1.0
    budget_override: int | None = None


def run_variant(cfg: RunConfig) -> dict[str, Any]:
    """Run one (variant, seed, scenario) combination. Returns a result dict."""
    budget = cfg.budget_override if cfg.budget_override is not None else cfg.budget

    scenario = AdversarialScenario(name="adversarial_agent")
    agent_cfg = AgentConfig(seed=cfg.seed, stealth=cfg.stealth,
                            goal_bias=cfg.goal_bias, budget=budget)
    state: ScenarioState = scenario.make_state()
    agent = SyntheticRedAgent(scenario, agent_cfg)

    # State tracking
    history: list[ActionRecord] = []
    total_calls = 0
    refused_calls_total = 0
    prev_latency: float | None = None
    retry_streak = 0
    first_refuse_step: int | None = None
    first_trip_step: int | None = None
    peak_entropy = 0.0

    # Controller setup per variant
    policy = SHOWCASE_POLICY
    entropy_fn = calculate_entropy
    use_cb: AdaptiveCircuitBreaker | None = None
    monitoring_only = False

    if cfg.variant == "baseline_no_control":
        monitoring_only = True
        policy = SHOWCASE_POLICY
    elif cfg.variant == "full_rnos":
        policy = SHOWCASE_POLICY
    elif cfg.variant == "tuned_rnos":
        policy = TUNED_RNOS_POLICY
    elif cfg.variant == "aggressive_rnos":
        policy = AGGRESSIVE_RNOS_POLICY
    elif cfg.variant == "trust_enabled_rnos":
        policy = TRUST_ENABLED_POLICY
    elif cfg.variant == "rnos_minus_entropy":
        policy = SHOWCASE_POLICY
        entropy_fn = _entropy_zero
    elif cfg.variant == "rnos_minus_hysteresis":
        policy = SHOWCASE_POLICY
        entropy_fn = _entropy_minus_failure_and_repeated
    elif cfg.variant == "rnos_minus_coherence":
        # Coherence is not in the control loop — identical to full_rnos
        policy = SHOWCASE_POLICY
    elif cfg.variant == "rnos_minus_trust":
        # Trust disabled in showcase policy — identical to full_rnos
        policy = SHOWCASE_POLICY
    elif cfg.variant == "tuned_cb":
        use_cb = _make_tuned_cb()
    elif cfg.variant == "showcase_cb":
        use_cb = _make_showcase_cb()
    elif cfg.variant == "shifted_rnos":
        policy = SHIFTED_RNOS_POLICY
    elif cfg.variant == "shifted_cb":
        # CB at +25% threshold
        use_cb = AdaptiveCircuitBreaker(
            window_size=TUNED_CB_WINDOW,
            initial_failure_rate=0.75,
            initial_cooldown_steps=TUNED_CB_COOLDOWN,
            max_cooldown_steps=TUNED_CB_COOLDOWN * 4,
            max_total_blocked=200,
        )
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    is_cb_only = use_cb is not None
    runtime: RNOSRuntime | None = None
    if not is_cb_only:
        runtime = _make_runtime(policy)

    for step in range(1, budget + 1):
        action = agent.choose_action(step)
        action_record = _build_action_record(
            action,
            executed_calls=total_calls,
            prev_latency=prev_latency,
            retry_streak=retry_streak,
        )

        decision = "ALLOW"
        cb_reason = "closed"
        entropy = 0.0

        if monitoring_only:
            # Baseline: compute entropy but never gate
            entropy = entropy_fn(history, action_record)
            peak_entropy = max(peak_entropy, entropy)

        elif is_cb_only:
            use_cb.tick()
            allowed, cb_reason = use_cb.should_execute()
            if not allowed:
                decision = "REFUSE"
                if first_trip_step is None:
                    first_trip_step = step
            elif cb_reason == "half_open_probe":
                decision = "DEGRADE"

        else:
            # RNOS variant (may use custom entropy fn)
            if entropy_fn is not calculate_entropy:
                entropy = entropy_fn(history, action_record)
                trust = calculate_trust(history, entropy)
                assessment = evaluate_policy(entropy, trust, policy)
                decision = assessment.decision.value.upper()
            else:
                assessment = runtime.evaluate(action_record)
                entropy = assessment.entropy
                decision = assessment.decision.value.upper()

            peak_entropy = max(peak_entropy, entropy)

        if decision == "REFUSE" and first_refuse_step is None:
            first_refuse_step = step
            if not is_cb_only:
                first_trip_step = step

        # Clamp factor
        if decision == "REFUSE":
            clamp = 0.0
        elif decision == "DEGRADE":
            if is_cb_only and cb_reason == "half_open_probe":
                clamp = 0.35
            else:
                clamp = 0.52
        else:
            clamp = 1.0

        # Apply fanout scale for perturbation A
        if cfg.budget_units_scale != 1.0 and action.kind == "fanout":
            from dataclasses import replace
            action = replace(action,
                             budget_units=max(1, int(round(action.budget_units * cfg.budget_units_scale))))

        observation = scenario.apply_action(action, state, clamp)

        if observation.executed_calls > 0:
            action_record.latency_ms = observation.weighted_latency_ms

            if monitoring_only:
                # Track history manually for entropy computation
                action_record.success = observation.step_success
                history.append(action_record)
            elif is_cb_only:
                use_cb.record_result(success=observation.step_success)
                action_record.success = observation.step_success
                history.append(action_record)
            else:
                if entropy_fn is not calculate_entropy:
                    action_record.success = observation.step_success
                    history.append(action_record)
                else:
                    # runtime.record_outcome manages its own self.history — don't double-append
                    runtime.record_outcome(action_record, success=observation.step_success)

            prev_latency = observation.weighted_latency_ms
            retry_streak = retry_streak + 1 if not observation.step_success else 0
        else:
            retry_streak = max(0, retry_streak - 1)

        total_calls += observation.executed_calls
        refused_calls_total += observation.refused_calls

        agent.observe(step, action, observation, decision)

    return {
        "variant": cfg.variant,
        "seed": cfg.seed,
        "budget": budget,
        "stealth": cfg.stealth,
        "goal_bias": cfg.goal_bias,
        "scenario": cfg.scenario_name,
        "cumulative_damage": round(state.cumulative_damage, 3),
        "refused_calls": refused_calls_total,
        "first_refuse_step": first_refuse_step,
        "first_trip_step": first_trip_step,
        "peak_entropy": round(peak_entropy, 3),
        "total_executed_calls": total_calls,
    }


# ---------------------------------------------------------------------------
# Benign high-load scenario (Perturbation E)
# ---------------------------------------------------------------------------

def run_benign_scenario(variant: str, seed: int) -> dict[str, Any]:
    """Agent only probes public_api (benign load). Correct answer: never block."""
    budget = 24
    scenario = AdversarialScenario(name="adversarial_agent")
    state: ScenarioState = scenario.make_state()

    policy = SHOWCASE_POLICY
    entropy_fn = calculate_entropy

    if variant in ("tuned_cb", "showcase_cb"):
        is_cb = True
        cb = _make_tuned_cb() if variant == "tuned_cb" else _make_showcase_cb()
    else:
        is_cb = False
        if variant == "full_rnos":
            policy = SHOWCASE_POLICY
        elif variant == "tuned_rnos":
            policy = TUNED_RNOS_POLICY
        elif variant == "aggressive_rnos":
            policy = AGGRESSIVE_RNOS_POLICY
        elif variant == "rnos_minus_hysteresis":
            entropy_fn = _entropy_minus_failure_and_repeated
        cb = None

    history: list[ActionRecord] = []
    runtime = _make_runtime(policy) if not is_cb else None

    total_planned = 0
    total_executed = 0
    blocks = 0

    rng_seed = seed
    import random
    rng = random.Random(rng_seed)

    for step in range(1, budget + 1):
        # Benign action: probe public_api
        action = SyntheticAction(
            kind="probe",
            phase="stable",
            budget_units=3,
            primary_target="public_api",
            targets=("public_api",),
            path=("public_api",),
            note="benign_load",
        )
        # Small random jitter in budget_units
        action = SyntheticAction(
            kind="probe",
            phase="stable",
            budget_units=max(1, int(3 + rng.randint(-1, 1))),
            primary_target="public_api",
            targets=("public_api",),
            path=("public_api",),
            note="benign_load",
        )

        action_record = _build_action_record(action, total_executed, None, 0)

        decision = "ALLOW"

        if is_cb:
            cb.tick()
            allowed, cb_reason = cb.should_execute()
            if not allowed:
                decision = "REFUSE"
            elif cb_reason == "half_open_probe":
                decision = "DEGRADE"
        else:
            if entropy_fn is not calculate_entropy:
                entropy = entropy_fn(history, action_record)
                trust = calculate_trust(history, entropy)
                assessment = evaluate_policy(entropy, trust, policy)
                decision = assessment.decision.value.upper()
            else:
                assessment = runtime.evaluate(action_record)
                decision = assessment.decision.value.upper()

        clamp = 0.0 if decision == "REFUSE" else (0.52 if decision == "DEGRADE" else 1.0)
        observation = scenario.apply_action(action, state, clamp)

        total_planned += observation.planned_calls
        total_executed += observation.executed_calls
        if decision == "REFUSE":
            blocks += 1

        if observation.executed_calls > 0:
            action_record.success = observation.step_success
            history.append(action_record)
            if is_cb:
                cb.record_result(success=observation.step_success)
            elif entropy_fn is calculate_entropy:
                runtime.record_outcome(action_record, success=observation.step_success)
                history = runtime.history

    allowed_rate = total_executed / max(total_planned, 1)

    return {
        "variant": variant,
        "seed": seed,
        "scenario": "benign_high_load",
        "total_planned": total_planned,
        "total_executed": total_executed,
        "allowed_rate": round(allowed_rate, 4),
        "blocks": blocks,
        "cumulative_damage": round(state.cumulative_damage, 3),
    }


# ---------------------------------------------------------------------------
# OOD adversary scenario (Perturbation D)
# ---------------------------------------------------------------------------

def run_ood_scenario(variant: str, seed: int) -> dict[str, Any]:
    """OOD adversary: cooldown for first 4 steps, then fanout-all for steps 5-24."""
    budget = 24
    scenario = AdversarialScenario(name="adversarial_agent")
    state: ScenarioState = scenario.make_state()

    policy = SHOWCASE_POLICY
    entropy_fn = calculate_entropy

    if variant in ("tuned_cb", "showcase_cb"):
        is_cb = True
        cb = _make_tuned_cb() if variant == "tuned_cb" else _make_showcase_cb()
    else:
        is_cb = False
        if variant == "tuned_rnos":
            policy = TUNED_RNOS_POLICY
        elif variant == "aggressive_rnos":
            policy = AGGRESSIVE_RNOS_POLICY
        elif variant == "rnos_minus_hysteresis":
            entropy_fn = _entropy_minus_failure_and_repeated
        cb = None

    history: list[ActionRecord] = []
    runtime = _make_runtime(policy) if not is_cb else None

    total_executed = 0
    refused_calls_total = 0
    first_refuse_step: int | None = None
    peak_entropy = 0.0

    service_names = list(scenario.service_names)

    for step in range(1, budget + 1):
        if step <= 4:
            # Silent phase: cooldown (no pressure)
            action = SyntheticAction(
                kind="cooldown",
                phase="explore",
                budget_units=1,
                primary_target=None,
                targets=(),
                path=("public_api",),
                note="ood_silent",
            )
        else:
            # Burst phase: fanout to all services
            all_targets = tuple(n for n in service_names if n != "public_api")
            action = SyntheticAction(
                kind="fanout",
                phase="persist",
                budget_units=12,
                primary_target="public_api",
                targets=all_targets,
                path=("public_api",),
                note="ood_burst",
            )

        action_record = _build_action_record(action, total_executed, None, 0)

        decision = "ALLOW"
        entropy = 0.0

        if is_cb:
            cb.tick()
            allowed, cb_reason = cb.should_execute()
            if not allowed:
                decision = "REFUSE"
            elif cb_reason == "half_open_probe":
                decision = "DEGRADE"
        else:
            if entropy_fn is not calculate_entropy:
                entropy = entropy_fn(history, action_record)
                trust = calculate_trust(history, entropy)
                assessment = evaluate_policy(entropy, trust, policy)
                decision = assessment.decision.value.upper()
            else:
                assessment = runtime.evaluate(action_record)
                entropy = assessment.entropy
                decision = assessment.decision.value.upper()
            peak_entropy = max(peak_entropy, entropy)

        if decision == "REFUSE" and first_refuse_step is None:
            first_refuse_step = step

        clamp = 0.0 if decision == "REFUSE" else (0.52 if decision == "DEGRADE" else 1.0)
        observation = scenario.apply_action(action, state, clamp)

        if observation.executed_calls > 0:
            action_record.success = observation.step_success
            history.append(action_record)
            if is_cb:
                cb.record_result(success=observation.step_success)
            elif entropy_fn is calculate_entropy:
                runtime.record_outcome(action_record, success=observation.step_success)
                history = runtime.history

        total_executed += observation.executed_calls
        refused_calls_total += observation.refused_calls

    return {
        "variant": variant,
        "seed": seed,
        "scenario": "ood_silent_then_burst",
        "cumulative_damage": round(state.cumulative_damage, 3),
        "refused_calls": refused_calls_total,
        "first_refuse_step": first_refuse_step,
        "peak_entropy": round(peak_entropy, 3),
        "total_executed": total_executed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Phase 3: Component Ablation + Perturbation ===")

    all_results = []

    # --- Part A: Core ablation on fresh seeds (adversarial scenario) ---
    print("\n[Part A] Core ablation on fresh seeds (adversarial_agent, 24-step)")

    core_variants = [
        "baseline_no_control",
        "full_rnos",
        "tuned_rnos",
        "aggressive_rnos",
        "trust_enabled_rnos",
        "rnos_minus_entropy",
        "rnos_minus_hysteresis",
        "rnos_minus_coherence",
        "rnos_minus_trust",
        "tuned_cb",
        "showcase_cb",
    ]

    for variant in core_variants:
        damages = []
        for seed in FRESH_SEEDS:
            cfg = RunConfig(variant=variant, seed=seed, scenario_name="adversarial_fresh")
            try:
                result = run_variant(cfg)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} damages={[round(d,1) for d in damages]} avg={avg:.1f}")

    # --- Part B: Canonical seeds (in-repo) for comparison ---
    print("\n[Part B] Canonical seeds (adversarial_agent, 24-step)")

    for variant in core_variants:
        damages = []
        for seed in CANONICAL_SEEDS:
            cfg = RunConfig(variant=variant, seed=seed, scenario_name="adversarial_canonical")
            try:
                result = run_variant(cfg)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} damages={[round(d,1) for d in damages]} avg={avg:.1f}")

    # --- Part C: Perturbation A (high fanout, scale 1.75) ---
    print("\n[Part C] Perturbation A: high fanout scale=1.75")

    for variant in ["full_rnos", "tuned_rnos", "tuned_cb", "baseline_no_control"]:
        damages = []
        for seed in FRESH_SEEDS:
            cfg = RunConfig(
                variant=variant,
                seed=seed,
                scenario_name="perturb_high_fanout",
                budget_units_scale=1.75,
            )
            try:
                result = run_variant(cfg)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} avg={avg:.1f}")

    # --- Part D: Perturbation B (threshold shift +25%) ---
    print("\n[Part D] Perturbation B: threshold shift +25%")

    for variant in ["shifted_rnos", "shifted_cb", "full_rnos", "tuned_cb"]:
        damages = []
        for seed in FRESH_SEEDS:
            cfg = RunConfig(variant=variant, seed=seed, scenario_name="perturb_threshold_shift")
            try:
                result = run_variant(cfg)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} avg={avg:.1f}")

    # --- Part E: Perturbation C (slow burn, 48 steps, 0.5x budget_units) ---
    print("\n[Part E] Perturbation C: slow burn (budget=48, scale=0.5)")

    for variant in ["full_rnos", "tuned_rnos", "tuned_cb", "baseline_no_control"]:
        damages = []
        for seed in FRESH_SEEDS:
            cfg = RunConfig(
                variant=variant,
                seed=seed,
                scenario_name="perturb_slow_burn",
                budget_units_scale=0.5,
                budget_override=48,
            )
            try:
                result = run_variant(cfg)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} avg={avg:.1f}")

    # --- Part F: OOD adversary (silent then burst) ---
    print("\n[Part F] Perturbation D: OOD adversary (silent then burst)")

    for variant in ["full_rnos", "tuned_rnos", "aggressive_rnos", "tuned_cb", "showcase_cb", "baseline_no_control"]:
        damages = []
        for seed in FRESH_SEEDS:
            try:
                result = run_ood_scenario(variant, seed)
                all_results.append(result)
                damages.append(result["cumulative_damage"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                damages.append(float("nan"))
        valid = [d for d in damages if not math.isnan(d)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {variant:<30} avg={avg:.1f}")

    # --- Part G: Benign high-load (false positive test) ---
    print("\n[Part G] Perturbation E: benign high-load (correct: ALLOW all)")

    for variant in ["full_rnos", "tuned_rnos", "aggressive_rnos", "tuned_cb", "showcase_cb"]:
        rates = []
        blocks_list = []
        for seed in FRESH_SEEDS:
            try:
                result = run_benign_scenario(variant, seed)
                all_results.append(result)
                rates.append(result["allowed_rate"])
                blocks_list.append(result["blocks"])
            except Exception as exc:
                print(f"  ERROR {variant} seed={seed}: {exc}")
                rates.append(float("nan"))
        valid = [r for r in rates if not math.isnan(r)]
        avg_rate = sum(valid) / len(valid) if valid else float("nan")
        avg_blocks = sum(blocks_list) / len(blocks_list) if blocks_list else 0
        print(f"  {variant:<30} avg_allowed_rate={avg_rate:.3f} avg_blocks={avg_blocks:.1f}")

    # Save all results
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for record in all_results:
            fh.write(json.dumps(record) + "\n")

    print(f"\nRaw results saved to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Total records: {len(all_results)}")


if __name__ == "__main__":
    main()
