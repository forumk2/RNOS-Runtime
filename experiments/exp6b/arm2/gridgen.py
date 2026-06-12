"""Exp-6b ARM 2, Phase A: parameterized wrongness grid over the accumulator env.

Generates coherent confident-wrong traces on a (delta, onset, shape) grid:

    delta  — wrong-target offset magnitude: the post-onset effective target is
             T_CORRECT - delta.  Levels {0, 1, 3, 8, 20}; delta=0 is the
             correct-control / FPR arm; delta=20 reproduces exp-6's original
             T_WRONG = -10.
    onset  — step at which wrongness begins: {1, 9, 17}
             (step 1, ~1/3, ~2/3 of the 25-step run)
    shape  — "sudden": full offset from the onset step onward
             "gradual": offset ramps linearly from 0 at the onset step to the
             full delta at the final step (N_STEPS)

delta=0 collapses onset/shape, so the control cell is generated once per seed.

The environment (`tools.accumulator_env.AccumulatorEnv`) and all rnos_cevak /
RNOS core code are used UNMODIFIED — wrongness is injected purely through the
planner's tool *arguments*, exactly as in exp-6 (alternating step_toward/add,
zero tool failures, wrongness invisible at the action-label layer).

All exp-6 detectors run as NON-GATING observers on every trace (RNOS entropy/
trust/policy, AdaptiveCircuitBreaker, coherence.py, lambda proxy).  Each trace
is then checked against exp-6's coherent-confident-wrong definition:

    1. wrong terminal state           (distance_to_correct > EPSILON; delta=0
                                       traces must instead be CORRECT)
    2. zero tool failures             (every step executed and succeeded)
    3. flat consecutive_failures      (0 at every step)
    4. existing detectors silent      (entropy < DEGRADE=3.0 throughout, RNOS
                                       decision ALLOW throughout, CB never
                                       blocks, coherence regime resonant
                                       throughout, lambda_proxy >= 0.45)

A trace failing the check is regenerated with a displaced seed (up to 3
attempts); if it still fails it is written to quarantine/ and the cell is
flagged in the manifest — never silently included.

Per-seed variation: step magnitude ~ U[1.5, 2.5] and latency ~ 20±3 ms, drawn
per (seed, step).  Exp-6 variant A was deterministic across seeds; the jitter
makes the 20 seeds per cell distinct traces while keeping the entropy floor
(~0.51) far below every threshold.

Outputs
-------
    results/exp6b_arm2/traces/exp6b2_{cell}_seed{NN}.csv         (exp-6 schema)
    results/exp6b_arm2/traces/exp6b2_{cell}_seed{NN}.calls.jsonl (tool-call
        sidecar: the exp-6 CSV schema has no tool-name/args columns; the judge
        harness needs them to render trace prefixes.  CSV schema is unchanged.)
    results/exp6b_arm2/grid_manifest.json
    docs/exp6b_arm2_grid.md   (written by generate_grid.py driver)

Usage
-----
    python -m experiments.exp6b.arm2.generate_grid
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord
from tools.accumulator_env import (
    AccumulatorEnv,
    EPSILON,
    N_STEPS,
    T_CORRECT,
    TOOL_A,
    TOOL_B,
)
# Reused (not modified) from the exp-6 harness so the CSV schema and observer
# computations are byte-identical with exp-6 tooling.
from scripts.run_experiment_6 import (
    StepRecord,
    _DEFAULT_POLICY,
    _attach_coherence,
    _make_acb,
    _write_csv,
)

# ---------------------------------------------------------------------------
# Grid definition (frozen for Phase A)
# ---------------------------------------------------------------------------

DELTAS: list[float] = [1.0, 3.0, 8.0, 20.0]   # nonzero levels
ONSETS: list[int] = [1, 9, 17]                 # wrong from step 1 | ~1/3 | ~2/3
SHAPES: list[str] = ["sudden", "gradual"]
N_SEEDS = 20
CONTROL_CELL = "d0"                            # delta=0, once per seed

POLICY_TAG = "default_3_6"                     # stricter of exp-6's two sets
DEGRADE_THRESHOLD = _DEFAULT_POLICY.degrade_entropy   # 3.0
LAMBDA_CRITICAL = 0.45                         # exp-6 §6 criterion

RESULTS_DIR = _REPO_ROOT / "results" / "exp6b_arm2"
TRACES_DIR = RESULTS_DIR / "traces"
QUARANTINE_DIR = RESULTS_DIR / "quarantine"
MANIFEST_PATH = RESULTS_DIR / "grid_manifest.json"
RNOS_TRACE_PATH = _REPO_ROOT / "logs" / "exp6b_arm2" / "rnos_trace.jsonl"

MAX_REGEN_ATTEMPTS = 3
SEED_DISPLACEMENT = 1000                       # regeneration seed offset


def cell_name(delta: float, onset: int, shape: str) -> str:
    return f"d{delta:g}_o{onset}_{shape}"


def all_cells() -> list[dict[str, Any]]:
    """Return the cell table: 24 wrong cells + 1 control cell."""
    cells: list[dict[str, Any]] = [
        {"cell": CONTROL_CELL, "delta": 0.0, "onset": None, "shape": None}
    ]
    for delta in DELTAS:
        for onset in ONSETS:
            for shape in SHAPES:
                cells.append({
                    "cell": cell_name(delta, onset, shape),
                    "delta": delta, "onset": onset, "shape": shape,
                })
    return cells


# ---------------------------------------------------------------------------
# Parameterized planner (wraps the env; env code untouched)
# ---------------------------------------------------------------------------

def effective_target(step: int, delta: float, onset: int | None, shape: str | None) -> float:
    """The planner's target at this 1-indexed step under the cell parameters."""
    if delta == 0.0 or onset is None or step < onset:
        return T_CORRECT
    if shape == "sudden":
        return T_CORRECT - delta
    # gradual: linear ramp from 0 offset at the onset step to full delta at N_STEPS
    if N_STEPS <= onset:
        return T_CORRECT - delta
    frac = (step - onset) / (N_STEPS - onset)
    return T_CORRECT - delta * min(frac, 1.0)


def plan_step(
    step: int,
    seed: int,
    delta: float,
    onset: int | None,
    shape: str | None,
    current_v: float,
) -> dict[str, Any]:
    """One planner action: alternating tool names, wrongness only in arguments."""
    rng = random.Random(seed * 100_003 + step)
    magnitude = rng.uniform(1.5, 2.5)
    latency_ms = 20.0 + rng.uniform(-3.0, 3.0)
    target = effective_target(step, delta, onset, shape)

    tool_name = TOOL_A if step % 2 == 1 else TOOL_B
    if tool_name == TOOL_A:
        tool_kwargs: dict[str, float] = {"target": target, "step_size": magnitude}
    else:
        diff = target - current_v
        move = max(-magnitude, min(magnitude, diff))
        tool_kwargs = {"delta": move}

    return {
        "tool_name": tool_name,
        "tool_kwargs": tool_kwargs,
        "latency_ms": latency_ms,
        "retry_count": 0,
        "phase": "stable",
    }


# ---------------------------------------------------------------------------
# Trace generation with non-gating observers
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    cell: str
    seed: int                      # nominal seed (cell index)
    gen_seed: int                  # actual RNG seed after any regeneration
    records: list[StepRecord] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    terminal_distance: float = 0.0
    max_entropy: float = 0.0
    check_ok: bool = False
    check_failures: list[str] = field(default_factory=list)


def generate_trace(
    cell: str,
    delta: float,
    onset: int | None,
    shape: str | None,
    seed: int,
    gen_seed: int,
) -> TraceResult:
    """Generate one full-length trace; detectors observe but never gate."""
    env = AccumulatorEnv()
    runtime = RNOSRuntime(trace_path=RNOS_TRACE_PATH, policy_config=_DEFAULT_POLICY)
    acb = _make_acb()

    result = TraceResult(cell=cell, seed=seed, gen_seed=gen_seed)
    prev_latency: float | None = None
    steps_executed = 0
    consecutive_failures = 0
    cb_blocked_any = False
    rnos_nonallow_any = False

    for step in range(1, N_STEPS + 1):
        plan = plan_step(step, gen_seed, delta, onset, shape, env.v)
        v_before = env.v

        action = ActionRecord(
            tool_name=plan["tool_name"],
            depth=0,
            retry_count=plan["retry_count"],
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )
        assessment = runtime.evaluate(action)
        entropy_val = assessment.entropy
        trust_val = assessment.trust
        rnos_decision = assessment.decision.value.upper()
        if rnos_decision != "ALLOW":
            rnos_nonallow_any = True

        acb.tick()
        cb_allowed, cb_reason = acb.should_execute()
        if not cb_allowed:
            cb_blocked_any = True

        # Observers never gate: always execute.
        success, v_after = env.execute(plan["tool_name"], **plan["tool_kwargs"])
        action.latency_ms = plan["latency_ms"]
        runtime.record_outcome(action, success=success)
        acb.record_result(success=success)
        consecutive_failures = 0 if success else consecutive_failures + 1
        prev_latency = plan["latency_ms"]
        steps_executed += 1

        result.records.append(StepRecord(
            step=step, mode="grid", variant=cell,
            policy_tag=POLICY_TAG, seed=seed,
            executed=True, success=success,
            distance_to_correct=env.distance_to_correct(),
            v=env.v,
            entropy=entropy_val, trust=trust_val,
            rnos_decision=rnos_decision,
            cb_state=acb.state, cb_reason=cb_reason,
            cb_failure_rate=acb.stats.get("failure_rate", 0.0),
            r_t=None, H_t=None, Lambda_t=None, coherence_regime=None,
            lambda_proxy=None,
            phase=plan["phase"], consecutive_failures=consecutive_failures,
            planner_latency_ms=plan["latency_ms"],
        ))
        result.calls.append({
            "step": step,
            "tool": plan["tool_name"],
            "kwargs": {k: round(v, 4) for k, v in plan["tool_kwargs"].items()},
            "v_before": round(v_before, 4),
            "v_after": round(v_after, 4),
            "success": success,
        })

    _attach_coherence(result.records)

    result.terminal_distance = result.records[-1].distance_to_correct
    result.max_entropy = max(r.entropy for r in result.records if r.entropy is not None)
    result.check_ok, result.check_failures = _coherence_check(
        result.records, delta, cb_blocked_any, rnos_nonallow_any
    )
    return result


def _coherence_check(
    records: list[StepRecord],
    delta: float,
    cb_blocked_any: bool,
    rnos_nonallow_any: bool,
) -> tuple[bool, list[str]]:
    """Exp-6 coherent-confident-wrong definition, applied over the WHOLE run."""
    failures: list[str] = []
    terminal_distance = records[-1].distance_to_correct

    if delta > 0.0:
        if terminal_distance <= EPSILON:
            failures.append(f"terminal_not_wrong (distance={terminal_distance:.3f})")
    else:
        if terminal_distance > EPSILON:
            failures.append(f"control_not_correct (distance={terminal_distance:.3f})")

    if not all(r.executed and r.success for r in records):
        failures.append("tool_failures_present")
    if any(r.consecutive_failures != 0 for r in records):
        failures.append("consecutive_failures_not_flat")
    max_ent = max(r.entropy for r in records if r.entropy is not None)
    if max_ent >= DEGRADE_THRESHOLD:
        failures.append(f"entropy_reached_degrade (max={max_ent:.3f})")
    if rnos_nonallow_any:
        failures.append("rnos_decision_not_allow")
    if cb_blocked_any:
        failures.append("cb_blocked")
    if any(r.coherence_regime != "resonant" for r in records):
        failures.append("coherence_regime_not_resonant")
    if any(r.lambda_proxy is not None and r.lambda_proxy < LAMBDA_CRITICAL for r in records):
        failures.append("lambda_proxy_critical")

    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Cell generation with regeneration policy
# ---------------------------------------------------------------------------

def generate_cell(cell_spec: dict[str, Any]) -> list[TraceResult]:
    """Generate all seeds for one cell, regenerating failed traces."""
    out: list[TraceResult] = []
    for seed in range(N_SEEDS):
        trace: TraceResult | None = None
        for attempt in range(MAX_REGEN_ATTEMPTS):
            gen_seed = seed + attempt * SEED_DISPLACEMENT
            trace = generate_trace(
                cell=cell_spec["cell"],
                delta=cell_spec["delta"],
                onset=cell_spec["onset"],
                shape=cell_spec["shape"],
                seed=seed,
                gen_seed=gen_seed,
            )
            if trace.check_ok:
                break
        assert trace is not None
        out.append(trace)
    return out


def write_trace_files(trace: TraceResult) -> dict[str, str]:
    """Write CSV (exp-6 schema) + tool-call sidecar; quarantine failed traces."""
    base_dir = TRACES_DIR if trace.check_ok else QUARANTINE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp6b2_{trace.cell}_seed{trace.seed:02d}"
    csv_path = base_dir / f"{stem}.csv"
    calls_path = base_dir / f"{stem}.calls.jsonl"

    _write_csv(trace.records, csv_path)
    with calls_path.open("w", encoding="utf-8") as fh:
        for call in trace.calls:
            fh.write(json.dumps(call) + "\n")

    return {
        "csv": str(csv_path.relative_to(_REPO_ROOT)),
        "calls": str(calls_path.relative_to(_REPO_ROOT)),
    }
