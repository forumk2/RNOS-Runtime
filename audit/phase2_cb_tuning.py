"""Phase 2 — CB hyperparameter sweep on training seeds 7, 42.

Evaluates AdaptiveCircuitBreaker across a grid of (window_size, threshold, cooldown)
on the adversarial_agent showcase with budget=24, stealth=0.35, goal_bias=privilege.

Objective: find the CB config that minimizes cumulative_damage_score on training seeds,
then freeze for evaluation on held-out seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from experiments.adversarial_agent_showcase.agent import AgentConfig, SyntheticRedAgent
from experiments.adversarial_agent_showcase.scenario import (
    AdversarialScenario,
    ScenarioState,
    SyntheticAction,
)
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord
import logging

logging.disable(logging.CRITICAL)

TRAINING_SEEDS = [7, 42]
TEST_SEEDS = [1337, 100, 200]
BUDGET = 24
STEALTH = 0.35
GOAL_BIAS = "privilege"
TRACE_PATH = REPO_ROOT / "audit" / "_cb_tune_trace.jsonl"
TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

SHOWCASE_RNOS_POLICY = PolicyConfig(
    degrade_entropy=8.4,
    refuse_entropy=10.2,
    degrade_trust=-0.1,
    refuse_trust=-0.1,
)

WINDOW_SIZES = [3, 5, 7]
THRESHOLDS = [0.40, 0.50, 0.60, 0.70]
COOLDOWNS = [1, 2, 3]


def _make_entropy_runtime() -> RNOSRuntime:
    TRACE_PATH.write_text("", encoding="utf-8")
    return RNOSRuntime(trace_path=TRACE_PATH, policy_config=SHOWCASE_RNOS_POLICY)


def _build_action_record(action: SyntheticAction, executed_calls: int,
                          prev_latency: float | None, retry_streak: int) -> ActionRecord:
    return ActionRecord(
        tool_name=action.primary_target or "public_api",
        payload={"action": action.kind, "phase": action.phase},
        depth=max(0, len(action.path) - 1),
        retry_count=retry_streak,
        latency_ms=prev_latency,
        cumulative_calls=executed_calls,
    )


def _clamp_for_cb(cb_reason: str) -> float:
    if cb_reason in {"open_blocked", "permanently_open"}:
        return 0.0
    if cb_reason == "half_open_probe":
        return 0.35
    return 1.0


def run_cb_config(window_size: int, threshold: float, cooldown: int,
                   seed: int) -> dict:
    scenario = AdversarialScenario(name="adversarial_agent")
    config = AgentConfig(seed=seed, stealth=STEALTH, goal_bias=GOAL_BIAS, budget=BUDGET)
    state: ScenarioState = scenario.make_state()
    agent = SyntheticRedAgent(scenario, config)

    cb = AdaptiveCircuitBreaker(
        window_size=window_size,
        initial_failure_rate=threshold,
        initial_cooldown_steps=cooldown,
        max_cooldown_steps=cooldown * 4,
        max_total_blocked=BUDGET * 2,
    )
    entropy_runtime = _make_entropy_runtime()

    total_calls = 0
    refused_calls = 0
    prev_latency = None
    retry_streak = 0
    first_refuse = None

    for step in range(1, BUDGET + 1):
        action = agent.choose_action(step)
        action_record = _build_action_record(action, total_calls, prev_latency, retry_streak)

        cb.tick()
        assessment = entropy_runtime.evaluate(action_record)
        allowed, cb_reason = cb.should_execute()
        decision = "ALLOW" if allowed else "REFUSE"
        if cb_reason == "half_open_probe":
            decision = "DEGRADE"

        if decision == "REFUSE" and first_refuse is None:
            first_refuse = step

        clamp = _clamp_for_cb(cb_reason)
        observation = scenario.apply_action(action, state, clamp)

        if observation.executed_calls > 0:
            action_record.latency_ms = observation.weighted_latency_ms
            entropy_runtime.record_outcome(action_record, success=observation.step_success)
            cb.record_result(success=observation.step_success)
            prev_latency = observation.weighted_latency_ms
            retry_streak = retry_streak + 1 if not observation.step_success else 0
        else:
            retry_streak = max(0, retry_streak - 1)

        total_calls += observation.executed_calls
        refused_calls += observation.refused_calls

        agent.observe(step, action, observation, decision)

    return {
        "window_size": window_size,
        "threshold": threshold,
        "cooldown": cooldown,
        "seed": seed,
        "cumulative_damage": round(state.cumulative_damage, 3),
        "refused_calls": refused_calls,
        "first_refuse": first_refuse,
    }


def run_rnos_mode(seed: int) -> dict:
    """Run full RNOS (no CB) for comparison."""
    scenario = AdversarialScenario(name="adversarial_agent")
    config = AgentConfig(seed=seed, stealth=STEALTH, goal_bias=GOAL_BIAS, budget=BUDGET)
    state: ScenarioState = scenario.make_state()
    agent = SyntheticRedAgent(scenario, config)

    runtime = _make_entropy_runtime()

    total_calls = 0
    refused_calls = 0
    prev_latency = None
    retry_streak = 0
    first_refuse = None

    for step in range(1, BUDGET + 1):
        action = agent.choose_action(step)
        action_record = _build_action_record(action, total_calls, prev_latency, retry_streak)

        assessment = runtime.evaluate(action_record)
        decision = assessment.decision.value.upper()

        if decision == "REFUSE" and first_refuse is None:
            first_refuse = step

        clamp = 0.0 if decision == "REFUSE" else (0.52 if decision == "DEGRADE" else 1.0)
        observation = scenario.apply_action(action, state, clamp)

        if observation.executed_calls > 0:
            action_record.latency_ms = observation.weighted_latency_ms
            runtime.record_outcome(action_record, success=observation.step_success)
            prev_latency = observation.weighted_latency_ms
            retry_streak = retry_streak + 1 if not observation.step_success else 0
        else:
            retry_streak = max(0, retry_streak - 1)

        total_calls += observation.executed_calls
        refused_calls += observation.refused_calls

        agent.observe(step, action, observation, decision)

    return {
        "variant": "full_rnos",
        "seed": seed,
        "cumulative_damage": round(state.cumulative_damage, 3),
        "refused_calls": refused_calls,
        "first_refuse": first_refuse,
    }


def main() -> None:
    print("=== Phase 2: CB Hyperparameter Sweep ===")
    print(f"Training seeds: {TRAINING_SEEDS}")
    print(f"Grid: window_sizes={WINDOW_SIZES}, thresholds={THRESHOLDS}, cooldowns={COOLDOWNS}")
    print()

    all_results = []
    config_scores: dict[tuple, list[float]] = {}

    total_configs = len(WINDOW_SIZES) * len(THRESHOLDS) * len(COOLDOWNS)
    done = 0

    for window_size in WINDOW_SIZES:
        for threshold in THRESHOLDS:
            for cooldown in COOLDOWNS:
                key = (window_size, threshold, cooldown)
                damages = []
                for seed in TRAINING_SEEDS:
                    result = run_cb_config(window_size, threshold, cooldown, seed)
                    all_results.append(result)
                    damages.append(result["cumulative_damage"])
                config_scores[key] = damages
                done += 1
                avg = sum(damages) / len(damages)
                if done % 6 == 0:
                    print(f"  [{done}/{total_configs}] w={window_size} t={threshold:.2f} c={cooldown} -> avg_damage={avg:.2f}")

    # Find best config on training seeds (minimizes avg damage)
    best_key = min(config_scores, key=lambda k: sum(config_scores[k]) / len(config_scores[k]))
    best_scores = config_scores[best_key]
    best_avg = sum(best_scores) / len(best_scores)

    print()
    print(f"BEST CB CONFIG (training): window={best_key[0]}, threshold={best_key[1]:.2f}, cooldown={best_key[2]}")
    print(f"  Training damages: {best_scores} -> avg={best_avg:.2f}")

    # Get RNOS scores on training seeds for comparison
    print()
    print("RNOS (full) on training seeds:")
    rnos_train_damages = []
    for seed in TRAINING_SEEDS:
        rnos_result = run_rnos_mode(seed)
        rnos_train_damages.append(rnos_result["cumulative_damage"])
        print(f"  seed={seed}: damage={rnos_result['cumulative_damage']:.2f}, refused={rnos_result['refused_calls']}, first_refuse={rnos_result['first_refuse']}")
    rnos_train_avg = sum(rnos_train_damages) / len(rnos_train_damages)
    print(f"  RNOS avg: {rnos_train_avg:.2f}")

    # Print top 5 CB configs
    ranked = sorted(config_scores, key=lambda k: sum(config_scores[k]) / len(config_scores[k]))
    print()
    print("Top 5 CB configs by training avg damage:")
    for key in ranked[:5]:
        scores = config_scores[key]
        avg = sum(scores) / len(scores)
        print(f"  w={key[0]} t={key[1]:.2f} c={key[2]} -> {scores} avg={avg:.2f}")

    tuning_output = {
        "training_seeds": TRAINING_SEEDS,
        "grid": {
            "window_sizes": WINDOW_SIZES,
            "thresholds": THRESHOLDS,
            "cooldowns": COOLDOWNS,
        },
        "best_cb_config": {
            "window_size": best_key[0],
            "threshold": best_key[1],
            "cooldown": best_key[2],
            "training_avg_damage": round(best_avg, 3),
        },
        "rnos_training_avg_damage": round(rnos_train_avg, 3),
        "all_config_scores": {
            f"w{k[0]}_t{k[1]:.2f}_c{k[2]}": {
                "damages": v,
                "avg": round(sum(v) / len(v), 3),
            }
            for k, v in config_scores.items()
        },
    }

    output_path = REPO_ROOT / "audit" / "cb_tuning_results.json"
    output_path.write_text(json.dumps(tuning_output, indent=2), encoding="utf-8")
    print(f"\nTuning results saved to {output_path.relative_to(REPO_ROOT)}")
    print(f"\nFROZEN TUNED CB CONFIG: window_size={best_key[0]}, threshold={best_key[1]:.2f}, cooldown={best_key[2]}")


if __name__ == "__main__":
    main()
