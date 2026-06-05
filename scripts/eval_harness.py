"""Multi-seed, multi-scenario, multi-mode evaluation harness.

Runs a Cartesian product of:
  seeds      × modes × scenarios × personas
  (0..N-1)  × 3     × 4         × 3

Default: 30 × 3 × 4 × 3 = 1080 runs.

Usage
-----
Dry-run (default, no LM Studio):
    python scripts/eval_harness.py --seeds 30 --tag full

Live (requires LM Studio on http://127.0.0.1:1234):
    python scripts/eval_harness.py --seeds 30 --tag full --live
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Silence RNOS per-step INFO logs — harness owns all metric collection
logging.getLogger("rnos").setLevel(logging.WARNING)

from baselines.circuit_breaker import CircuitBreaker
from rnos.entropy import calculate_entropy
from rnos.policy import PolicyConfig, evaluate_policy
from rnos.trust import calculate_trust
from rnos.types import ActionRecord, PolicyDecision
from tools.scenarios import SCENARIOS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODES = ["rnos", "circuit_breaker", "baseline"]
SCENARIO_NAMES = ["cascade", "flaky", "recovering", "stable"]
PERSONA_NAMES = ["adversarial", "cautious", "mixed"]


# ---------------------------------------------------------------------------
# Single-run core
# ---------------------------------------------------------------------------


def _run_single(
    seed: int,
    mode: str,
    scenario_name: str,
    persona: str,
    *,
    max_steps: int = 20,
    dry_run: bool = True,
    allow_max: float = 3.0,
    degrade_max: float = 6.0,
    tag: str = "",
) -> dict[str, Any]:
    """Execute one (seed, mode, scenario, persona) combination.

    Returns a flat dict suitable for JSONL serialisation.
    All randomness is isolated: module-level random state is NOT touched.
    """
    scenario_cls = SCENARIOS[scenario_name]
    tool = scenario_cls(seed)
    tool_name = tool.name

    policy_config = PolicyConfig(
        degrade_entropy=allow_max,
        refuse_entropy=degrade_max,
    )

    # Call entropy/trust/policy functions directly — no logger, no trace I/O
    rnos_history: list[ActionRecord] = []

    cb: CircuitBreaker | None = None
    if mode == "circuit_breaker":
        cb = CircuitBreaker(failure_threshold=3, initial_cooldown_steps=1, max_cooldown_steps=8, max_total_blocked=10)

    # --- per-run accumulators ------------------------------------------------
    tool_executions = 0
    tool_failures = 0
    blocked_refused = 0
    first_intervention_step: int | None = None
    first_intervention_kind: str | None = None
    final_state = "completed"
    entropy_trace: list[float] = []
    trust_trace: list[float] = []
    decision_trace: list[str] = []

    retry_count = 0
    degrade_remaining: int | None = None
    last_step = 0

    for step in range(1, max_steps + 1):
        last_step = step

        action = ActionRecord(
            tool_name=tool_name,
            payload={"resource": "/status"},
            depth=step - 1,
            retry_count=retry_count,
            latency_ms=0.0,         # dry-run: no planner call
            cumulative_calls=tool_executions,
            metadata={"step": step},
        )

        # --- control decision ------------------------------------------------
        if mode == "rnos":
            entropy = calculate_entropy(rnos_history, action)
            trust = calculate_trust(rnos_history, entropy)
            assessment = evaluate_policy(entropy, trust, policy_config)
            entropy_trace.append(assessment.entropy)
            trust_trace.append(assessment.trust)
            decision_trace.append(assessment.decision.value)

            if assessment.decision is PolicyDecision.REFUSE:
                blocked_refused += 1
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_kind = "refuse"
                final_state = "refused"
                break

            if assessment.decision is PolicyDecision.DEGRADE:
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_kind = "degrade"
                if degrade_remaining is None:
                    degrade_remaining = int(assessment.constraints.get("max_additional_steps", 1))
                elif degrade_remaining == 0:
                    blocked_refused += 1
                    final_state = "degrade_exhausted"
                    break
            elif assessment.decision is PolicyDecision.ALLOW:
                degrade_remaining = None

        elif mode == "circuit_breaker":
            assert cb is not None
            cb.tick()
            allowed, cb_reason = cb.should_execute()

            if not allowed:
                blocked_refused += 1
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_kind = "blocked"
                if cb_reason == "permanently_open":
                    final_state = "permanently_open"
                    break
                entropy_trace.append(0.0)
                trust_trace.append(0.0)
                decision_trace.append("blocked")
                continue

            if cb_reason == "half_open_probe" and first_intervention_step is None:
                first_intervention_step = step
                first_intervention_kind = "half_open"

            entropy_trace.append(0.0)
            trust_trace.append(0.0)
            decision_trace.append(cb_reason)

        else:  # baseline
            entropy_trace.append(0.0)
            trust_trace.append(0.0)
            decision_trace.append("bypass")

        # --- execute tool ----------------------------------------------------
        result = tool.run(resource="/status")
        tool_executions += 1

        if mode == "rnos":
            action.success = result.ok
            rnos_history.append(action)
        elif mode == "circuit_breaker":
            assert cb is not None
            cb.record_result(success=result.ok)

        if not result.ok:
            tool_failures += 1
            retry_count += 1
        else:
            retry_count = 0

        if mode == "rnos" and degrade_remaining is not None:
            degrade_remaining -= 1

    return {
        "seed": seed,
        "mode": mode,
        "scenario": scenario_name,
        "persona": persona,
        "tag": tag,
        "max_steps": max_steps,
        "dry_run": dry_run,
        "allow_max": allow_max,
        "degrade_max": degrade_max,
        "total_steps": last_step,
        "tool_executions": tool_executions,
        "tool_failures": tool_failures,
        "blocked_refused": blocked_refused,
        "first_intervention_step": first_intervention_step,
        "first_intervention_kind": first_intervention_kind,
        "final_state": final_state,
        "entropy_trace": entropy_trace,
        "trust_trace": trust_trace,
        "decision_trace": decision_trace,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _run_single_star(args: tuple) -> dict[str, Any]:
    """Unpacking shim for multiprocessing.Pool.imap."""
    return _run_single(*args[0], **args[1])


# ---------------------------------------------------------------------------
# Harness entry point
# ---------------------------------------------------------------------------


def run_harness(
    *,
    num_seeds: int = 30,
    max_steps: int = 20,
    dry_run: bool = True,
    tag: str = "",
    allow_max: float = 3.0,
    degrade_max: float = 6.0,
    workers: int | None = None,
) -> Path:
    """Run the full Cartesian product and write results to JSONL.

    Returns the path to the output file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / f"eval_{tag}.jsonl" if tag else RESULTS_DIR / "eval_default.jsonl"
    # Truncate on a new run so reruns produce clean output
    out_path.write_text("", encoding="utf-8")

    # Build work list: (positional_args_tuple, keyword_args_dict)
    work: list[tuple[tuple, dict]] = []
    for seed in range(num_seeds):
        for mode in MODES:
            for scenario in SCENARIO_NAMES:
                for persona in PERSONA_NAMES:
                    work.append((
                        (seed, mode, scenario, persona),
                        {
                            "max_steps": max_steps,
                            "dry_run": dry_run,
                            "allow_max": allow_max,
                            "degrade_max": degrade_max,
                            "tag": tag,
                        },
                    ))

    total = len(work)
    print(f"Eval harness: {total} runs  tag={tag!r}  dry_run={dry_run}")
    t0 = time.monotonic()

    n_workers = workers if workers is not None else min(multiprocessing.cpu_count(), 8)
    completed = 0

    with open(out_path, "a", encoding="utf-8") as fh:
        if dry_run or n_workers <= 1:
            # Single-process path keeps output clean and avoids spawn overhead
            # on small run counts.
            for args in work:
                rec = _run_single_star(args)
                fh.write(json.dumps(rec) + "\n")
                completed += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = time.monotonic() - t0
                    print(f"  {completed}/{total}  ({elapsed:.1f}s elapsed)")
        else:
            # Parallel path for live runs (LM Studio latency dominates)
            with multiprocessing.Pool(processes=n_workers) as pool:
                for rec in pool.imap_unordered(_run_single_star, work, chunksize=4):
                    fh.write(json.dumps(rec) + "\n")
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        elapsed = time.monotonic() - t0
                        print(f"  {completed}/{total}  ({elapsed:.1f}s elapsed)")

    elapsed = time.monotonic() - t0
    print(f"Done: {completed} rows written to {out_path}  ({elapsed:.1f}s total)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed multi-scenario RNOS evaluation harness.")
    parser.add_argument("--seeds", type=int, default=30, metavar="N", help="Number of seeds (0..N-1). Default: 30.")
    parser.add_argument("--max-steps", type=int, default=20, metavar="N", help="Max loop steps per run. Default: 20.")
    parser.add_argument("--tag", type=str, default="", metavar="TEXT", help="Output file tag: results/eval_<tag>.jsonl.")
    parser.add_argument("--live", action="store_true", help="Use real LM Studio planner instead of dry-run stub.")
    parser.add_argument("--workers", type=int, default=None, metavar="N", help="Parallel worker processes (live mode only).")
    parser.add_argument("--allow-max", type=float, default=3.0, metavar="F", help="RNOS DEGRADE entropy threshold. Default: 3.0.")
    parser.add_argument("--degrade-max", type=float, default=6.0, metavar="F", help="RNOS REFUSE entropy threshold. Default: 6.0.")
    args = parser.parse_args()

    out = run_harness(
        num_seeds=args.seeds,
        max_steps=args.max_steps,
        dry_run=not args.live,
        tag=args.tag,
        allow_max=args.allow_max,
        degrade_max=args.degrade_max,
        workers=args.workers,
    )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
