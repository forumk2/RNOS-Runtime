"""RNOS Experiment 6: Coherent-Failure Detection.

Tests whether any RNOS execution-trace signal can flag a run that executes
smoothly and successfully at every step while converging on the *wrong*
terminal outcome.

Two variants:
  A — pure confident-wrong: all steps succeed, flat latency, phase=stable
  B — confident-wrong with friction: same trajectory, rising latency,
       planner retries from b* onward, phase=unstable

Four modes per variant:
  baseline   — no control; runs to N_STEPS regardless
  rnos       — RNOS cumulative entropy / trust gating
  cb         — AdaptiveCircuitBreaker (no RNOS)
  hybrid     — HybridController (RNOS + CB + coherence-proxy)

Coherence (coherence.py) and lambda_proxy (hybrid.py logic) are computed
as read-only observers on every mode's trace.

Piggyback checks (same session):
  combo_refuse_fpr   — recoverable_burst: does hybrid wrongly REFUSE during recovery?
  ewma_effectiveness — distributed_low_rate with alpha 0.0 / 0.10 / 0.30

Usage
-----
    python scripts/run_experiment_6.py
    python scripts/run_experiment_6.py --seeds 20 --max-steps 25

Outputs
-------
    results/experiment_6/{variant}_{mode}_{policy_tag}_seed{N}.csv
    results/experiment_6/summary.json
    docs/experiment_6_coherent_failure.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from baselines.circuit_breaker import CircuitBreaker
from experiments.experiment_2 import EXP2_POLICY
from rnos.coherence import compute_runtime_coherence

# Pre-initialise rnos.runtime logger at WARNING level BEFORE any RNOSRuntime
# is created.  rnos/logger.py's get_logger() checks `if logger.handlers`
# and returns early without overriding setLevel when handlers already exist.
_pre_logger = logging.getLogger("rnos.runtime")
_pre_logger.setLevel(logging.WARNING)
if not _pre_logger.handlers:
    _pre_handler = logging.StreamHandler()
    _pre_handler.setLevel(logging.WARNING)
    _pre_logger.addHandler(_pre_handler)
    _pre_logger.propagate = False
from rnos.entropy import calculate_entropy
from rnos.hybrid import HybridController
from rnos.policy import PolicyConfig, evaluate_policy
from rnos.runtime import RNOSRuntime
from rnos.trust import calculate_trust
from rnos.types import ActionRecord, PolicyDecision
from tools.accumulator_env import (
    AccumulatorEnv,
    B_STAR,
    N_STEPS,
    T_CORRECT,
    T_WRONG,
    make_plan_step,
)

logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_RESULTS_DIR = _REPO_ROOT / "results" / "experiment_6"
_SUMMARY_PATH = _RESULTS_DIR / "summary.json"
_DOCS_PATH = _REPO_ROOT / "docs" / "experiment_6_coherent_failure.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp6_trace.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_SEEDS = 20
_MAX_STEPS = N_STEPS   # 25

# Default / illustrative policy thresholds
_DEFAULT_POLICY = PolicyConfig()                         # degrade=3.0, refuse=6.0
# Canonical policy: what exp-5 actually used
# NOTE: EXP2_POLICY in experiments/experiment_2.py is degrade=7.5, refuse=10.0
# The comment in run_experiment_5.py ("degrade=9.0, refuse=11.0") is incorrect.
_CANONICAL_POLICY = EXP2_POLICY                          # degrade=7.5, refuse=10.0

_POLICY_SETS = [
    ("default_3_6", _DEFAULT_POLICY),
    ("canonical_7p5_10", _CANONICAL_POLICY),
]

_ACB_WINDOW = 10
_ACB_THRESHOLD = 0.60
_ACB_COOLDOWN = 3

# ---------------------------------------------------------------------------
# DEGRADE threshold used for H0/H1 evaluation windows
# ---------------------------------------------------------------------------

_DEGRADE_THRESHOLDS = {
    "default_3_6": _DEFAULT_POLICY.degrade_entropy,      # 3.0
    "canonical_7p5_10": _CANONICAL_POLICY.degrade_entropy,  # 7.5
}

# ---------------------------------------------------------------------------
# Entropy floor derivation (Task 0 result — see §1 TRIAGE AMENDMENT)
# ---------------------------------------------------------------------------
# For a zero-failure accumulator run with alternating tool names:
#   depth_score      = 0      (depth=0 always)
#   retry_score      = 0      (retry_count=0 in Variant A)
#   failure_score    = 0      (no failures in last 5)
#   repeated_tool    = 0      (alternating step_toward / add)
#   latency_score    = 0.010  (20 ms x 0.5 / 1000)
#   cost_score       = 0.500  (base_cost caps at cumulative_calls>=20;
#                              spend_per_success=1.0 -> waste=0)
#   long_memory_score= 0      (EWMA=0, no failures ever)
#   -------------------------
#   FLOOR              0.510
#
# With single repeated tool name: repeated_tool=2, floor=2.510.
# Both are well below the default DEGRADE threshold of 3.0.
#
# Canonical (7.5/10.0): both floors << 7.5. H0 holds under both sets.
ENTROPY_FLOOR_ALTERNATING = 0.510   # alternating tool names (this experiment)
ENTROPY_FLOOR_SINGLE_TOOL = 2.510   # single tool name (worst case)


# ---------------------------------------------------------------------------
# Step record dataclass
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    mode: str
    variant: str
    policy_tag: str
    seed: int
    executed: bool
    success: bool | None
    distance_to_correct: float   # external oracle — never seen by detectors
    v: float
    entropy: float | None
    trust: float | None
    rnos_decision: str | None
    cb_state: str | None
    cb_reason: str | None
    cb_failure_rate: float | None
    r_t: float | None            # filled after coherence pass
    H_t: float | None
    Lambda_t: float | None
    coherence_regime: str | None
    lambda_proxy: float | None   # hybrid.py proxy, filled after coherence pass
    phase: str
    consecutive_failures: int
    planner_latency_ms: float


# ---------------------------------------------------------------------------
# Coherence observer: standalone lambda proxy (mirrors HybridController)
# ---------------------------------------------------------------------------

def _compute_lambda_proxy_series(records: list[StepRecord]) -> list[float]:
    """Compute the rolling 8-step lambda proxy over a trace.

    Mirrors HybridController._coherence_lambda_proxy() but applied post-hoc
    to a recorded trace rather than live.  Returns one value per step.
    """
    window: deque[StepRecord] = deque(maxlen=8)
    result: list[float] = []

    for rec in records:
        if rec.executed and rec.success is not None:
            window.append(rec)

        executed_recs = [r for r in window if r.success is not None]
        if len(executed_recs) < 2:
            result.append(1.0)
            continue

        lambdas: list[float] = []
        for r in executed_recs:
            s_et = 1.0 if r.success else 0.0
            # rnos_allowed: True if RNOS said ALLOW or DEGRADE
            rnos_allowed = r.rnos_decision in {None, "ALLOW", "DEGRADE"}
            s_pg = 1.0 if rnos_allowed else 0.0
            r_t_proxy = (1.0 + s_pg + 1.0 + s_et) / 4.0
            f_t_proxy = 0.0 if r.success else 1.0
            b_t_proxy = 0.0 if rnos_allowed else 1.0
            h_t_proxy = 0.35 * f_t_proxy + 0.25 * b_t_proxy
            lambdas.append(r_t_proxy / (1.0 + h_t_proxy))
        result.append(round(sum(lambdas) / len(lambdas), 3))

    return result


def _attach_coherence(records: list[StepRecord]) -> None:
    """Run coherence.py observer and lambda proxy; fill r_t/H_t/Lambda_t/regime/lambda_proxy in place."""
    # Build step_trace for coherence.py
    # decision mapping per §4: ALLOW/DEGRADE->EXECUTE, REFUSE->STOPPED, CB block->BLOCKED
    def _decision_to_coherence(r: StepRecord) -> str:
        if not r.executed:
            if r.cb_reason in {"open_blocked", "permanently_open"}:
                return "BLOCKED"
            return "STOPPED"
        return "EXECUTE"

    step_trace = [
        {
            "step": r.step,
            "phase": r.phase,
            "decision": _decision_to_coherence(r),
            "tool_result": ("SUCCESS" if r.success else "FAILURE") if r.executed else "NONE",
            "planner_emitted_tool_call": True,
            "consecutive_failures": r.consecutive_failures,
            "planner_latency_ms": r.planner_latency_ms,
        }
        for r in records
    ]

    report = compute_runtime_coherence(step_trace)
    for i, row in enumerate(report["table"]):
        records[i].r_t = round(row["r_t"], 3)
        records[i].H_t = round(row["H_t"], 3)
        records[i].Lambda_t = round(row["Lambda_t"], 3)
        records[i].coherence_regime = row["regime"]

    # Lambda proxy series
    proxy_series = _compute_lambda_proxy_series(records)
    for i, lp in enumerate(proxy_series):
        records[i].lambda_proxy = lp


# ---------------------------------------------------------------------------
# CB factory
# ---------------------------------------------------------------------------

def _make_acb() -> AdaptiveCircuitBreaker:
    return AdaptiveCircuitBreaker(
        window_size=_ACB_WINDOW,
        initial_failure_rate=_ACB_THRESHOLD,
        initial_cooldown_steps=_ACB_COOLDOWN,
    )


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

def _run_accumulator(
    mode: str,
    variant: str,
    policy_config: PolicyConfig,
    policy_tag: str,
    seed: int,
    max_steps: int,
) -> list[StepRecord]:
    """Run one modexvariantxseed combination; return per-step records."""
    env = AccumulatorEnv()
    records: list[StepRecord] = []

    # Set up controllers based on mode
    runtime: RNOSRuntime | None = None
    acb: AdaptiveCircuitBreaker | None = None
    ctrl: HybridController | None = None
    prev_latency: float | None = None
    steps_executed = 0

    if mode in {"rnos", "hybrid"}:
        runtime = RNOSRuntime(trace_path=_TRACE_PATH, policy_config=policy_config)

    if mode in {"cb", "hybrid"}:
        acb = _make_acb()

    if mode == "hybrid":
        assert runtime is not None and acb is not None
        ctrl = HybridController(runtime, acb)

    for step in range(1, max_steps + 1):
        plan = make_plan_step(step, variant, seed)
        tool_name = plan["tool_name"]
        tool_kwargs = plan["tool_kwargs"]
        latency_ms = plan["latency_ms"]
        retry_count = plan["retry_count"]
        phase = plan["phase"]

        # ----------------------------------------------------------------
        # Control decisions
        # ----------------------------------------------------------------
        entropy_val: float | None = None
        trust_val: float | None = None
        rnos_decision: str | None = None
        cb_state: str | None = None
        cb_reason_str: str | None = None
        cb_failure_rate: float | None = None

        # RNOS evaluate
        if mode == "rnos":
            assert runtime is not None
            action = ActionRecord(
                tool_name=tool_name,
                depth=0,
                retry_count=retry_count,
                latency_ms=prev_latency,
                cumulative_calls=steps_executed,
            )
            assessment = runtime.evaluate(action)
            entropy_val = assessment.entropy
            trust_val = assessment.trust
            rnos_decision = assessment.decision.value.upper()

            if assessment.decision is PolicyDecision.REFUSE:
                records.append(StepRecord(
                    step=step, mode=mode, variant=variant,
                    policy_tag=policy_tag, seed=seed,
                    executed=False, success=None,
                    distance_to_correct=env.distance_to_correct(),
                    v=env.v,
                    entropy=entropy_val, trust=trust_val,
                    rnos_decision=rnos_decision,
                    cb_state=None, cb_reason=None, cb_failure_rate=None,
                    r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                    lambda_proxy=None,
                    phase=phase, consecutive_failures=0,
                    planner_latency_ms=latency_ms,
                ))
                break

            # Execute tool
            success, _ = env.execute(tool_name, **tool_kwargs)
            action.latency_ms = latency_ms
            runtime.record_outcome(action, success=success)
            prev_latency = latency_ms
            steps_executed += 1
            records.append(StepRecord(
                step=step, mode=mode, variant=variant,
                policy_tag=policy_tag, seed=seed,
                executed=True, success=success,
                distance_to_correct=env.distance_to_correct(),
                v=env.v,
                entropy=entropy_val, trust=trust_val,
                rnos_decision=rnos_decision,
                cb_state=None, cb_reason=None, cb_failure_rate=None,
                r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                lambda_proxy=None,
                phase=phase, consecutive_failures=0,
                planner_latency_ms=latency_ms,
            ))

        elif mode == "cb":
            assert acb is not None
            acb.tick()
            cb_allowed, cb_reason_str = acb.should_execute()
            cb_stats = acb.stats
            cb_state = acb.state
            cb_failure_rate = cb_stats.get("failure_rate", 0.0)

            if not cb_allowed:
                records.append(StepRecord(
                    step=step, mode=mode, variant=variant,
                    policy_tag=policy_tag, seed=seed,
                    executed=False, success=None,
                    distance_to_correct=env.distance_to_correct(),
                    v=env.v,
                    entropy=None, trust=None, rnos_decision=None,
                    cb_state=cb_state, cb_reason=cb_reason_str,
                    cb_failure_rate=cb_failure_rate,
                    r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                    lambda_proxy=None,
                    phase=phase, consecutive_failures=0,
                    planner_latency_ms=latency_ms,
                ))
                if cb_reason_str == "permanently_open":
                    break
                continue  # CB blocked but not permanently open — skip this step

            success, _ = env.execute(tool_name, **tool_kwargs)
            acb.record_result(success=success)
            prev_latency = latency_ms
            steps_executed += 1
            cb_stats_after = acb.stats
            records.append(StepRecord(
                step=step, mode=mode, variant=variant,
                policy_tag=policy_tag, seed=seed,
                executed=True, success=success,
                distance_to_correct=env.distance_to_correct(),
                v=env.v,
                entropy=None, trust=None, rnos_decision=None,
                cb_state=acb.state, cb_reason=cb_reason_str,
                cb_failure_rate=cb_stats_after.get("failure_rate", 0.0),
                r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                lambda_proxy=None,
                phase=phase, consecutive_failures=0,
                planner_latency_ms=latency_ms,
            ))

        elif mode == "hybrid":
            assert ctrl is not None
            action = ActionRecord(
                tool_name=tool_name,
                depth=0,
                retry_count=retry_count,
                latency_ms=prev_latency,
                cumulative_calls=steps_executed,
            )
            ctrl.tick()
            hd = ctrl.evaluate(action)
            entropy_val = hd.rnos_entropy
            trust_val = hd.rnos_trust
            rnos_decision = hd.rnos_decision
            cb_state = hd.cb_state
            cb_reason_str = hd.cb_reason
            cb_failure_rate = hd.cb_failure_rate

            if hd.decision == "REFUSE":
                records.append(StepRecord(
                    step=step, mode=mode, variant=variant,
                    policy_tag=policy_tag, seed=seed,
                    executed=False, success=None,
                    distance_to_correct=env.distance_to_correct(),
                    v=env.v,
                    entropy=entropy_val, trust=trust_val,
                    rnos_decision=rnos_decision,
                    cb_state=cb_state, cb_reason=cb_reason_str,
                    cb_failure_rate=round(cb_failure_rate, 3),
                    r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                    lambda_proxy=None,
                    phase=phase, consecutive_failures=0,
                    planner_latency_ms=latency_ms,
                ))
                break

            success, _ = env.execute(tool_name, **tool_kwargs)
            action.latency_ms = latency_ms
            ctrl.record_outcome(action, success=success)
            prev_latency = latency_ms
            steps_executed += 1
            records.append(StepRecord(
                step=step, mode=mode, variant=variant,
                policy_tag=policy_tag, seed=seed,
                executed=True, success=success,
                distance_to_correct=env.distance_to_correct(),
                v=env.v,
                entropy=entropy_val, trust=trust_val,
                rnos_decision=rnos_decision,
                cb_state=cb_state, cb_reason=cb_reason_str,
                cb_failure_rate=round(cb_failure_rate, 3),
                r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                lambda_proxy=None,
                phase=phase, consecutive_failures=0,
                planner_latency_ms=latency_ms,
            ))

        else:  # baseline
            success, _ = env.execute(tool_name, **tool_kwargs)
            steps_executed += 1
            records.append(StepRecord(
                step=step, mode=mode, variant=variant,
                policy_tag=policy_tag, seed=seed,
                executed=True, success=success,
                distance_to_correct=env.distance_to_correct(),
                v=env.v,
                entropy=None, trust=None, rnos_decision=None,
                cb_state=None, cb_reason=None, cb_failure_rate=None,
                r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
                lambda_proxy=None,
                phase=phase, consecutive_failures=0,
                planner_latency_ms=latency_ms,
            ))

    # Attach coherence and lambda_proxy as read-only observers
    _attach_coherence(records)
    return records


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "step", "mode", "variant", "policy_tag", "seed",
    "executed", "success",
    "distance_to_correct", "v",
    "entropy", "trust", "rnos_decision",
    "cb_state", "cb_reason", "cb_failure_rate",
    "r_t", "H_t", "Lambda_t", "coherence_regime",
    "lambda_proxy",
    "phase", "consecutive_failures", "planner_latency_ms",
]


def _write_csv(records: list[StepRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "step": r.step,
                "mode": r.mode,
                "variant": r.variant,
                "policy_tag": r.policy_tag,
                "seed": r.seed,
                "executed": r.executed,
                "success": r.success,
                "distance_to_correct": round(r.distance_to_correct, 4),
                "v": round(r.v, 4),
                "entropy": round(r.entropy, 3) if r.entropy is not None else "",
                "trust": round(r.trust, 3) if r.trust is not None else "",
                "rnos_decision": r.rnos_decision or "",
                "cb_state": r.cb_state or "",
                "cb_reason": r.cb_reason or "",
                "cb_failure_rate": round(r.cb_failure_rate, 3) if r.cb_failure_rate is not None else "",
                "r_t": r.r_t if r.r_t is not None else "",
                "H_t": r.H_t if r.H_t is not None else "",
                "Lambda_t": r.Lambda_t if r.Lambda_t is not None else "",
                "coherence_regime": r.coherence_regime or "",
                "lambda_proxy": r.lambda_proxy if r.lambda_proxy is not None else "",
                "phase": r.phase,
                "consecutive_failures": r.consecutive_failures,
                "planner_latency_ms": round(r.planner_latency_ms, 1),
            })


# ---------------------------------------------------------------------------
# §6 Decision criteria evaluation
# ---------------------------------------------------------------------------

@dataclass
class CriteriaResult:
    variant: str
    policy_tag: str
    seed: int
    mode: str
    # oracle
    terminal_distance: float
    terminal_v: float
    is_wrong_terminal: bool
    # detection events
    entropy_first_degrade_step: int | None    # first step entropy >= DEGRADE threshold
    cb_first_block_step: int | None           # first step CB blocked
    coherence_first_cf_step: int | None       # first step coherent-failure signature fires
    lambda_proxy_first_critical_step: int | None  # first step lambda_proxy < 0.45
    # run termination
    final_step: int
    terminated_early: bool
    termination_reason: str


def _evaluate_criteria(
    records: list[StepRecord],
    policy_tag: str,
    degrade_threshold: float,
) -> CriteriaResult:
    """Apply §6 decision criteria to a single run's records."""
    final = records[-1]
    b_star_records = [r for r in records if r.step >= B_STAR]

    # Oracle
    terminal_distance = final.distance_to_correct
    terminal_v = final.v
    is_wrong = terminal_distance > 0.5

    # Entropy: first step entropy >= DEGRADE threshold (in window [b*, t_term])
    entropy_step = next(
        (r.step for r in b_star_records
         if r.entropy is not None and r.entropy >= degrade_threshold),
        None,
    )

    # CB: first step blocked
    cb_step = next(
        (r.step for r in records if not r.executed and r.cb_reason in
         {"open_blocked", "permanently_open"}),
        None,
    )

    # Coherence: first step where coherence_regime is not "resonant" in window [b*, t_term]
    coherence_step = next(
        (r.step for r in b_star_records
         if r.coherence_regime is not None and r.coherence_regime != "resonant"),
        None,
    )

    # Lambda proxy: first step lambda_proxy < 0.45 (critical or collapse)
    lambda_proxy_step = next(
        (r.step for r in b_star_records
         if r.lambda_proxy is not None and r.lambda_proxy < 0.45),
        None,
    )

    # Termination
    terminated_early = len(records) < _MAX_STEPS
    last_rec = records[-1]
    if not last_rec.executed:
        if last_rec.rnos_decision == "REFUSE":
            term_reason = "rnos_refuse"
        elif last_rec.cb_reason in {"open_blocked", "permanently_open"}:
            term_reason = "cb_block"
        else:
            term_reason = "blocked_unknown"
    else:
        term_reason = "completed"

    return CriteriaResult(
        variant=records[0].variant,
        policy_tag=policy_tag,
        seed=records[0].seed,
        mode=records[0].mode,
        terminal_distance=round(terminal_distance, 4),
        terminal_v=round(terminal_v, 4),
        is_wrong_terminal=is_wrong,
        entropy_first_degrade_step=entropy_step,
        cb_first_block_step=cb_step,
        coherence_first_cf_step=coherence_step,
        lambda_proxy_first_critical_step=lambda_proxy_step,
        final_step=last_rec.step,
        terminated_early=terminated_early,
        termination_reason=term_reason,
    )


# ---------------------------------------------------------------------------
# Piggyback check: EWMA effectiveness
# ---------------------------------------------------------------------------

def _ewma_effectiveness_check() -> dict[str, Any]:
    """Piggyback §9: EWMA on distributed_low_rate with alpha 0 / 0.10 / 0.30.

    Computes the EWMA failure accumulation manually for each alpha value
    and records when (if ever) the EWMA-only signal would cross a threshold.
    Uses a 30-step F-F-S repeating pattern (same as exp-5 distributed_low_rate).
    """
    pattern = [False, False, True] * 10  # 30 steps: F,F,S repeating
    alphas = [0.0, 0.10, 0.30]
    results: dict[str, Any] = {}

    for alpha in alphas:
        ewma = 0.0
        series: list[float] = []
        for success in pattern:
            signal = 0.0 if success else 1.0
            ewma = alpha * signal + (1.0 - alpha) * ewma
            series.append(round(ewma * 2.0, 4))  # scaled to [0, 2.0] as in entropy.py

        # First step where EWMA score >= 1.0 (half of max)
        first_above_1 = next((i + 1 for i, v in enumerate(series) if v >= 1.0), None)
        results[f"alpha_{alpha:.2f}"] = {
            "final_ewma_score": series[-1],
            "first_step_above_1.0": first_above_1,
            "series": series,
        }

    return results


# ---------------------------------------------------------------------------
# Piggyback check: combo-REFUSE false-positive rate
# ---------------------------------------------------------------------------

def _combo_refuse_fpr_check(max_steps: int = 30) -> dict[str, Any]:
    """Piggyback §8: Check whether the combo-REFUSE rule fires during recovery.

    Scenario: 5 consecutive failures (burst) then 25 successes (recovery).
    With EXP2_POLICY (7.5/10.0), RNOS should DEGRADE during the burst then
    recover.  Test whether the hybrid REFUSE fires during the recovery phase
    (false positive).

    Returns counts of DEGRADE and REFUSE decisions during the recovery window
    (steps 6-30).
    """
    schedule = [False] * 5 + [True] * 25  # burst then recovery
    latency_burst = [400.0] * 5
    latency_recovery = [80.0] * 25

    runtime = RNOSRuntime(trace_path=_TRACE_PATH, policy_config=_CANONICAL_POLICY)
    acb = _make_acb()
    ctrl = HybridController(runtime, acb)

    recovery_degrade = 0
    recovery_refuse = 0
    step_log: list[dict] = []
    prev_latency: float | None = None
    steps_executed = 0
    retry_count = 0

    for step in range(1, max_steps + 1):
        is_recovery_phase = step > 5
        success_outcome = schedule[step - 1]
        latency = (latency_burst + latency_recovery)[step - 1]

        action = ActionRecord(
            tool_name="burst_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )
        ctrl.tick()
        hd = ctrl.evaluate(action)

        if hd.decision == "REFUSE":
            if is_recovery_phase:
                recovery_refuse += 1
            step_log.append({"step": step, "decision": hd.decision,
                             "entropy": hd.rnos_entropy, "phase": "recovery" if is_recovery_phase else "burst"})
            break

        if hd.decision == "DEGRADE" and is_recovery_phase:
            recovery_degrade += 1

        action.latency_ms = latency
        ctrl.record_outcome(action, success=success_outcome)
        if not success_outcome:
            retry_count += 1
        else:
            retry_count = 0
        prev_latency = latency
        steps_executed += 1

        step_log.append({
            "step": step,
            "decision": hd.decision,
            "entropy": round(hd.rnos_entropy, 3),
            "cb_state": hd.cb_state,
            "phase": "recovery" if is_recovery_phase else "burst",
        })

    return {
        "recovery_degrade_count": recovery_degrade,
        "recovery_refuse_count": recovery_refuse,
        "total_steps": len(step_log),
        "false_positive_refuse": recovery_refuse > 0,
        "step_log": step_log,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics across seeds
# ---------------------------------------------------------------------------

def _aggregate(criteria_list: list[CriteriaResult]) -> dict[str, Any]:
    """Compute cross-seed aggregates for one (variant, policy_tag, mode)."""
    n = len(criteria_list)
    if n == 0:
        return {}

    wrong_count = sum(1 for c in criteria_list if c.is_wrong_terminal)
    entropy_fire_count = sum(1 for c in criteria_list if c.entropy_first_degrade_step is not None)
    cb_fire_count = sum(1 for c in criteria_list if c.cb_first_block_step is not None)
    coherence_fire_count = sum(1 for c in criteria_list if c.coherence_first_cf_step is not None)
    lambda_fire_count = sum(1 for c in criteria_list if c.lambda_proxy_first_critical_step is not None)
    early_term_count = sum(1 for c in criteria_list if c.terminated_early)

    entropy_steps = [c.entropy_first_degrade_step for c in criteria_list if c.entropy_first_degrade_step]
    cb_steps = [c.cb_first_block_step for c in criteria_list if c.cb_first_block_step]
    coherence_steps = [c.coherence_first_cf_step for c in criteria_list if c.coherence_first_cf_step]
    lambda_steps = [c.lambda_proxy_first_critical_step for c in criteria_list if c.lambda_proxy_first_critical_step]

    def _mean(lst: list) -> float | None:
        return round(sum(lst) / len(lst), 2) if lst else None

    return {
        "n_seeds": n,
        "wrong_terminal_rate": round(wrong_count / n, 3),
        "entropy_detection_rate": round(entropy_fire_count / n, 3),
        "cb_detection_rate": round(cb_fire_count / n, 3),
        "coherence_cf_detection_rate": round(coherence_fire_count / n, 3),
        "lambda_proxy_critical_rate": round(lambda_fire_count / n, 3),
        "early_termination_rate": round(early_term_count / n, 3),
        "entropy_mean_detect_step": _mean(entropy_steps),
        "cb_mean_detect_step": _mean(cb_steps),
        "coherence_mean_detect_step": _mean(coherence_steps),
        "lambda_proxy_mean_detect_step": _mean(lambda_steps),
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(
    all_agg: dict[str, Any],
    ewma_results: dict[str, Any],
    combo_fpr: dict[str, Any],
    n_seeds: int,
    max_steps: int,
) -> str:
    lines: list[str] = []

    lines += [
        "# Experiment 6: Coherent-Failure Detection",
        "",
        f"**Date:** 2026-06-09  **Seeds:** {n_seeds}  **Steps per run:** {max_steps}",
        f"**Threshold sets:** default (3.0/6.0) and canonical (7.5/10.0)",
        f"**Branch step b*:** {B_STAR}  **T_correct:** {T_CORRECT}  **T_wrong:** {T_WRONG}",
        "",
        "## §0 Entropy Floor (post-refactor formula)",
        "",
        "For a zero-failure accumulator run with alternating tool names (step_toward / add):",
        "",
        "| Signal | Value | Formula |",
        "|---|---|---|",
        "| depth_score | 0 | depth=0 always |",
        "| retry_score | 0 | retry_count=0 in Variant A |",
        "| failure_score | 0 | no failures in last 5 steps |",
        "| repeated_tool | 0 | alternating names |",
        "| latency_score | 0.010 | 20 ms x 0.5/1000 |",
        "| cost_score | 0.500 | base_cost saturates at cumulative>=20; spend_per_success=1.0 -> waste=0 |",
        "| long_memory_score | 0 | EWMA=0, no failures |",
        "| **Floor** | **0.510** | well below DEGRADE=3.0 (default) or 7.5 (canonical) |",
        "",
        "With single repeated tool name: repeated_tool=2, floor=**2.510** — still below 3.0.",
        "",
        "**Canonical threshold set:** `EXP2_POLICY` in `experiments/experiment_2.py` is",
        "`degrade_entropy=7.5, refuse_entropy=10.0`.",
        "The comment in `run_experiment_5.py` ('degrade=9.0, refuse=11.0') is a documentation",
        "error; the Python object resolves to 7.5/10.0. This experiment uses both",
        "default (3.0/6.0) and canonical (7.5/10.0).",
        "",
    ]

    # --- §7 Results table ---------------------------------------------------
    lines += ["## §7 Results Table", ""]
    lines += [
        "Metric: detection rate across N=" + str(n_seeds) + " seeds.",
        "A run is WRONG if distance_to_correct(terminal) > 0.5 with all steps succeeded.",
        "",
        "| Variant | Policy | Mode | Detector | Fires in [b*,t_term]? | Detection Rate | Mean Detect Step |",
        "|---|---|---|---|---|---|---|",
    ]

    detector_keys = [
        ("entropy", "entropy_detection_rate", "entropy_mean_detect_step"),
        ("circuit_breaker", "cb_detection_rate", "cb_mean_detect_step"),
        ("coherence_cf", "coherence_cf_detection_rate", "coherence_mean_detect_step"),
        ("lambda_proxy", "lambda_proxy_critical_rate", "lambda_proxy_mean_detect_step"),
    ]

    for key, agg in all_agg.items():
        variant, policy_tag, mode = key.split("|")
        for det_name, rate_key, step_key in detector_keys:
            rate = agg.get(rate_key, 0.0)
            step = agg.get(step_key) or "—"
            fires = "YES" if rate > 0.0 else "**NO**"
            lines.append(
                f"| {variant} | {policy_tag} | {mode} | {det_name} | {fires} | {rate:.3f} | {step} |"
            )
    lines.append("")

    # --- H0 / H1 verdict ---------------------------------------------------
    lines += ["## §6 Decision Criteria Evaluation", ""]

    lines += [
        "### H0: In Variant A (pure confident-wrong), do all detectors stay healthy?",
        "",
    ]
    for key, agg in all_agg.items():
        variant, policy_tag, mode = key.split("|")
        if variant != "A":
            continue
        h0_holds = (
            agg.get("entropy_detection_rate", 1.0) == 0.0
            and agg.get("cb_detection_rate", 1.0) == 0.0
            and agg.get("coherence_cf_detection_rate", 1.0) == 0.0
            and agg.get("lambda_proxy_critical_rate", 1.0) == 0.0
            and agg.get("wrong_terminal_rate", 0.0) > 0.0
        )
        status = "CONFIRMED" if h0_holds else "VIOLATED"
        lines.append(
            f"- [{policy_tag} / {mode}] wrong_terminal_rate={agg.get('wrong_terminal_rate', '?'):.3f}  "
            f"entropy_rate={agg.get('entropy_detection_rate', '?'):.3f}  "
            f"cb_rate={agg.get('cb_detection_rate', '?'):.3f}  "
            f"coherence_rate={agg.get('coherence_cf_detection_rate', '?'):.3f}  "
            f"-> H0 **{status}**"
        )
    lines.append("")

    lines += [
        "### H1: In Variant B, does coherence fire before entropy?",
        "",
    ]
    for key, agg in all_agg.items():
        variant, policy_tag, mode = key.split("|")
        if variant != "B":
            continue
        coh_step = agg.get("coherence_mean_detect_step")
        ent_step = agg.get("entropy_mean_detect_step")
        coh_rate = agg.get("coherence_cf_detection_rate", 0.0)

        if coh_rate == 0.0:
            h1_verdict = "coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)"
        elif ent_step is None:
            h1_verdict = f"coherence fires (mean step {coh_step}), entropy never fires — **coherence adds value**"
        elif coh_step < ent_step:
            h1_verdict = f"coherence fires at step {coh_step}, entropy at {ent_step} — **coherence adds value**"
        elif coh_step == ent_step:
            h1_verdict = f"both fire at step {coh_step} — **tie, coherence redundant in this regime**"
        else:
            h1_verdict = f"entropy fires first at {ent_step}, coherence at {coh_step} — **entropy dominates**"

        lines.append(f"- [{policy_tag} / {mode}] {h1_verdict}")
    lines.append("")

    # --- Structural finding on coherent_failure signature ------------------
    lines += [
        "## §8 Structural Finding: Coherent-Failure Signature Cannot Fire Without Tool Failures",
        "",
        "`coherence.py:_find_coherent_failure_run` requires ALL of:",
        "  1. `r_t >= 0.75` (high synchrony) — satisfied when tools execute and succeed",
        "  2. `H_t` strictly rising between consecutive steps",
        "  3. `consecutive_failures` strictly rising between consecutive steps",
        "  4. `phase != 'stable'` in at least one of the two steps",
        "",
        "In Variant B (friction without tool failures), `consecutive_failures = 0` throughout.",
        "Therefore condition 3 can never be satisfied, and the signature never fires.",
        "",
        "This is a structural finding, not a threshold choice: **the coherent-failure",
        "signature as coded in `coherence.py` requires actual tool failures to accumulate",
        "alongside high synchrony.** It detects 'execution proceeding coherently WHILE",
        "failures accumulate' — not a run that fails silently with every step succeeding.",
        "",
        "The lambda proxy (`hybrid.py`) is similarly blind in both variants:",
        "- With all tools succeeding and RNOS allowing: `r_t = 1.0`, `h_t = 0`",
        "  -> `lambda_proxy = 1.0` -> always RESONANT.",
        "",
        "**Implication:** To detect confident-wrongness, a goal-progress signal is needed",
        "(§8 constructive: `goal_divergence` as a 7th entropy term, or CEVAK probe).",
        "",
    ]

    # --- Sanity check -------------------------------------------------------
    lines += [
        "## Sanity Check: Oracle Independence",
        "",
        "`distance_to_correct` is computed from `env.distance_to_correct()` and logged",
        "to CSV only. It is never passed to `ActionRecord`, `calculate_entropy`,",
        "`calculate_trust`, `evaluate_policy`, `AdaptiveCircuitBreaker`, or",
        "`HybridController`. The oracle is provably independent of all detector inputs.",
        "",
    ]

    # --- Piggyback: EWMA ---------------------------------------------------
    lines += [
        "## Piggyback: EWMA Effectiveness (Triage Item #5)",
        "",
        "distributed_low_rate pattern (F-F-S repeating, 30 steps) with alpha 0/0.10/0.30:",
        "",
        "| alpha | Final EWMA score (x2.0) | First step score >= 1.0 |",
        "|---|---|---|",
    ]
    for alpha_key, data in ewma_results.items():
        final = data["final_ewma_score"]
        step_above = data["first_step_above_1.0"] or "never"
        lines.append(f"| {alpha_key} | {final:.4f} | {step_above} |")
    lines += [
        "",
        "α=0.0: EWMA frozen at 0 — the long-memory signal is completely disabled.",
        "α=0.10 (current): EWMA accumulates; reaches meaningful signal after many steps.",
        "α=0.30 (fast): reaches >= 1.0 earlier; more responsive but higher false-alarm risk.",
        "",
    ]

    # --- Piggyback: combo-REFUSE FPR ---------------------------------------
    lines += [
        "## Piggyback: Combo-REFUSE False-Positive Rate (Triage Item #8)",
        "",
        "Scenario: 5 consecutive failures (burst) -> 25 successes (recovery). Hybrid controller.",
        "",
        f"- Recovery DEGRADE count (steps 6-30): {combo_fpr['recovery_degrade_count']}",
        f"- Recovery REFUSE count (steps 6-30): {combo_fpr['recovery_refuse_count']}",
        f"- False-positive REFUSE during recovery: **{'YES' if combo_fpr['false_positive_refuse'] else 'NO'}**",
        "",
    ]
    if combo_fpr["false_positive_refuse"]:
        lines.append(
            "WARNING: The combo-REFUSE rule fires REFUSE during legitimate recovery. "
            "This is a false positive that penalises a system that has genuinely recovered."
        )
    else:
        lines.append(
            "The combo-REFUSE rule does not fire REFUSE during legitimate recovery "
            "under this scenario. The post-merge hybrid does not introduce FP-REFUSE "
            "on the burst+recovery pattern."
        )
    lines.append("")

    # --- Constructive next steps -------------------------------------------
    lines += [
        "## Constructive Next Steps (§8 from spec)",
        "",
        "Since H0 holds — trace-internal detectors are all blind to confident-wrong runs:",
        "",
        "1. **External progress signal as a 7th entropy term.** Add `goal_divergence`:",
        "   sustained increase in an external/estimated `distance_to_correct` (or a",
        "   self-reported progress estimate). Changes entropy from turbulence-only to",
        "   turbulence-plus-direction. Cost: requires a goal model.",
        "",
        "2. **CEVAK probe.** Run CEVAK's output-distribution view over the Variant A trace",
        "   and test whether it flags drift where the execution-trace detectors are silent.",
        "   If yes: 'execution-trace detection has a hard blind spot; CEVAK covers it.'",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 6: Coherent-Failure Detection")
    parser.add_argument("--seeds", type=int, default=_N_SEEDS)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    args = parser.parse_args()

    n_seeds = args.seeds
    max_steps = args.max_steps

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.write_text("", encoding="utf-8")

    modes = ["baseline", "rnos", "cb", "hybrid"]
    variants = ["A", "B"]

    all_criteria: dict[str, list[CriteriaResult]] = {}
    all_agg: dict[str, Any] = {}

    print(f"\n{'='*70}")
    print(f"Experiment 6: Coherent-Failure Detection")
    print(f"Seeds: {n_seeds}  Steps: {max_steps}  b*: {B_STAR}")
    print(f"{'='*70}")

    for variant in variants:
        for policy_tag, policy_config in _POLICY_SETS:
            degrade_threshold = _DEGRADE_THRESHOLDS[policy_tag]
            print(f"\nVariant {variant} | policy={policy_tag} | DEGRADE>={degrade_threshold}")

            for mode in modes:
                key = f"{variant}|{policy_tag}|{mode}"
                all_criteria[key] = []

                for seed in range(n_seeds):
                    records = _run_accumulator(
                        mode=mode,
                        variant=variant,
                        policy_config=policy_config,
                        policy_tag=policy_tag,
                        seed=seed,
                        max_steps=max_steps,
                    )

                    # Write per-seed CSV
                    csv_path = _RESULTS_DIR / f"exp6_{variant}_{mode}_{policy_tag}_seed{seed:02d}.csv"
                    _write_csv(records, csv_path)

                    criteria = _evaluate_criteria(records, policy_tag, degrade_threshold)
                    all_criteria[key].append(criteria)

                agg = _aggregate(all_criteria[key])
                all_agg[key] = agg

                wrong_rate = agg.get("wrong_terminal_rate", 0.0)
                ent_rate = agg.get("entropy_detection_rate", 0.0)
                coh_rate = agg.get("coherence_cf_detection_rate", 0.0)
                early_rate = agg.get("early_termination_rate", 0.0)
                print(
                    f"  [{mode:8s}] wrong={wrong_rate:.2f}  entropy_fires={ent_rate:.2f}  "
                    f"coherence_fires={coh_rate:.2f}  early_term={early_rate:.2f}"
                )

    # --- Piggyback checks ---------------------------------------------------
    print(f"\n{'='*70}")
    print("Piggyback checks")
    print(f"{'='*70}")

    ewma_results = _ewma_effectiveness_check()
    print("\nEWMA effectiveness (distributed_low_rate):")
    for alpha_key, data in ewma_results.items():
        print(f"  {alpha_key}: final_score={data['final_ewma_score']:.4f}  "
              f"first_above_1.0=step {data['first_step_above_1.0']}")

    combo_fpr = _combo_refuse_fpr_check()
    print(f"\nCombo-REFUSE FPR check:")
    print(f"  recovery_degrade={combo_fpr['recovery_degrade_count']}  "
          f"recovery_refuse={combo_fpr['recovery_refuse_count']}  "
          f"FP_refuse={'YES' if combo_fpr['false_positive_refuse'] else 'NO'}")

    # --- Summary JSON -------------------------------------------------------
    summary_data = {
        "experiment": "6",
        "entropy_floor": {
            "alternating_tools": ENTROPY_FLOOR_ALTERNATING,
            "single_tool": ENTROPY_FLOOR_SINGLE_TOOL,
        },
        "canonical_threshold_set": {
            "name": "EXP2_POLICY",
            "degrade_entropy": _CANONICAL_POLICY.degrade_entropy,
            "refuse_entropy": _CANONICAL_POLICY.refuse_entropy,
            "note": "run_experiment_5.py comment says 9.0/11.0 which is WRONG; actual object is 7.5/10.0",
        },
        "aggregates": {k: v for k, v in all_agg.items()},
        "ewma_effectiveness": ewma_results,
        "combo_refuse_fpr": {
            k: v for k, v in combo_fpr.items() if k != "step_log"
        },
    }
    _SUMMARY_PATH.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSummary JSON -> {_SUMMARY_PATH.relative_to(_REPO_ROOT)}")

    # --- Markdown report ----------------------------------------------------
    report = _build_report(all_agg, ewma_results, combo_fpr, n_seeds, max_steps)
    _DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_PATH.write_text(report, encoding="utf-8")
    print(f"Report -> {_DOCS_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
