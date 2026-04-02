"""Run an LM Studio-driven RNOS intervention scenario."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from agent.parser import parse_action
from agent.planner import Planner, PersonaName
from baselines.circuit_breaker import CircuitBreaker
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import PolicyDecision
from tools.unstable_api import UnstableAPI, UnstableAPITool

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "rnos_trace.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PlannerProtocol(Protocol):
    """Minimal interface satisfied by both Planner and the dry-run stub."""

    def get_next_action(self, history: list[dict[str, Any]]) -> str:
        """Return the next action string."""
        ...


class _DryRunPlanner:
    """Stub that always returns ``CALL unstable_api`` without hitting LM Studio."""

    def get_next_action(self, history: list[dict[str, Any]]) -> str:
        """Return a fixed action string, no LLM call required."""
        return "CALL unstable_api"


def _load_policy_config(config_path: Path) -> PolicyConfig:
    """Parse a YAML config file and return a :class:`PolicyConfig`.

    Reads the ``policy`` subsection; any missing key falls back to the
    :class:`PolicyConfig` defaults.
    """
    with config_path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    policy = raw.get("policy", {})
    return PolicyConfig(
        refuse_entropy=float(policy.get("refuse_entropy", 6.0)),
        refuse_trust=float(policy.get("refuse_trust", 0.2)),
        degrade_entropy=float(policy.get("degrade_entropy", 3.0)),
        degrade_trust=float(policy.get("degrade_trust", 0.45)),
    )


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_agent(
    *,
    max_steps: int,
    seed: int,
    no_rnos: bool = False,
    circuit_breaker: bool = False,
    cb_threshold: int = 3,
    cb_cooldown: int = 1,
    cb_max_cooldown: int = 8,
    cb_max_blocked: int = 10,
    dry_run: bool = False,
    persona: PersonaName = "adversarial",
    config_path: Path | None = None,
) -> dict[str, object]:
    """Execute an RNOS-controlled, circuit-breaker-controlled, or bypassed tool loop.

    Args:
        max_steps: Maximum number of loop iterations.
        seed: Random seed for the :class:`UnstableAPI` failure simulation.
        no_rnos: When True, skip all RNOS evaluation (decision printed as
            ``BYPASS``). The loop runs until ``max_steps`` is exhausted.
        circuit_breaker: When True, use exponential backoff circuit breaker
            instead of RNOS. Mutually exclusive with ``no_rnos``.
        cb_threshold: Consecutive failures before the circuit breaker trips.
        cb_cooldown: Initial cooldown steps before the first probe.
        cb_max_cooldown: Maximum cooldown ceiling after exponential growth.
        cb_max_blocked: Total blocked steps before permanently opening.
        dry_run: Replace :class:`Planner` with a stub; no LM Studio required.
        persona: System-prompt persona forwarded to :class:`Planner`.
        config_path: Optional path to a YAML file with ``policy`` thresholds.
    """

    random.seed(seed)

    # --- planner selection ---------------------------------------------------
    planner: _PlannerProtocol
    if dry_run:
        planner = _DryRunPlanner()
    else:
        planner = Planner(persona=persona)

    # --- policy config (RNOS only) -------------------------------------------
    policy_config: PolicyConfig | None = None
    if config_path is not None:
        policy_config = _load_policy_config(config_path)

    # --- runtime & tools -----------------------------------------------------
    rnos = RNOSRuntime(trace_path=TRACE_PATH, policy_config=policy_config)
    unstable_api = UnstableAPI()
    tool = UnstableAPITool(api=unstable_api)

    # --- circuit breaker setup -----------------------------------------------
    cb: CircuitBreaker | None = None
    total_blocked_steps = 0
    max_cooldown_reached = False
    if circuit_breaker:
        cb = CircuitBreaker(
            failure_threshold=cb_threshold,
            initial_cooldown_steps=cb_cooldown,
            max_cooldown_steps=cb_max_cooldown,
            max_total_blocked=cb_max_blocked,
        )

    # --- loop state ----------------------------------------------------------
    history: list[dict[str, object]] = []
    retry_count = 0
    refused = False
    total_failures = 0
    steps_executed = 0
    final_entropy = 0.0
    final_trust = 0.0

    # Degrade budget (RNOS mode only):
    # Invariant — degrade_remaining is decremented immediately after each
    # degraded tool execution.  A value of 0 means the budget is exhausted;
    # this is detected in the DEGRADE branch before the NEXT tool is allowed
    # to run.  None means we are not currently in a degrade window.
    degrade_remaining: int | None = None

    # --- header --------------------------------------------------------------
    if dry_run:
        print("[DRY RUN] LM Studio not called — planner returns 'CALL unstable_api' always")
    print("=== LM Studio RNOS Loop ===")
    if circuit_breaker:
        mode_label = "circuit_breaker"
        print(
            f"mode={mode_label} seed={seed} max_steps={max_steps} persona={persona} "
            f"cb_threshold={cb_threshold} cb_cooldown={cb_cooldown} "
            f"cb_max_cooldown={cb_max_cooldown} cb_max_blocked={cb_max_blocked}"
        )
    else:
        mode_label = "baseline (--no-rnos)" if no_rnos else "rnos"
        print(f"mode={mode_label} seed={seed} max_steps={max_steps} persona={persona}")
    if config_path:
        print(f"config={config_path}")

    # =========================================================================
    for step in range(1, max_steps + 1):

        # --- planner call (measure wall-clock time for latency signal) -------
        t0 = time.monotonic()
        llm_output = planner.get_next_action(history)
        planner_latency_ms = (time.monotonic() - t0) * 1000.0

        # --- build action record ---------------------------------------------
        action = parse_action(llm_output)
        action.depth = step - 1
        action.retry_count = retry_count
        action.latency_ms = planner_latency_ms
        action.cumulative_calls = steps_executed
        action.payload = {"resource": "/status"}
        action.metadata["step"] = step

        # --- control decision ------------------------------------------------
        executed_in_degrade = False  # RNOS only
        cb_reason: str = "closed"   # circuit breaker only; set below

        if circuit_breaker:
            # -----------------------------------------------------------------
            # Circuit breaker path — no RNOS calls
            # -----------------------------------------------------------------
            assert cb is not None
            cb.tick()
            allowed, cb_reason = cb.should_execute()

            if not allowed:
                cb_stats = cb.stats
                decision_str = (
                    "BREAKER_STOPPED" if cb_reason == "permanently_open"
                    else "BREAKER_BLOCKED"
                )
                print(
                    f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
                    f"decision={decision_str}"
                )
                print(
                    f"           breaker_state={cb.state} "
                    f"cooldown_remaining={cb_stats['cooldown_remaining']} "
                    f"consecutive_failures={cb_stats['consecutive_failures']} "
                    f"total_blocked={cb_stats['total_blocked']}"
                )
                total_blocked_steps += 1
                history.append(
                    {
                        "step": step,
                        "llm_output": llm_output,
                        "tool": action.tool_name,
                        "decision": decision_str.lower(),
                    }
                )
                if cb_reason == "permanently_open":
                    break
                continue

            # Allowed — print step header now (tool result printed below)
            decision_str = (
                "BREAKER_HALF_OPEN" if cb_reason == "half_open_probe"
                else "BREAKER_CLOSED"
            )
            print(
                f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
                f"decision={decision_str} "
                f"retry_count={retry_count}"
            )

        elif no_rnos:
            print(
                f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
                f"entropy=N/A trust=N/A decision=BYPASS "
                f"retry_count={retry_count}"
            )
        else:
            assessment = rnos.evaluate(action)
            final_entropy = assessment.entropy
            final_trust = assessment.trust

            print(
                f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
                f"entropy={assessment.entropy:.3f} "
                f"trust={assessment.trust:.3f} "
                f"decision={assessment.decision.value.upper()} "
                f"retry_count={retry_count}"
            )

            if assessment.decision is PolicyDecision.REFUSE:
                refused = True
                print("           stop=RNOS refused execution")
                break

            if assessment.decision is PolicyDecision.DEGRADE:
                if degrade_remaining is None:
                    # First entry into degrade mode: initialise budget from constraints.
                    degrade_remaining = int(
                        assessment.constraints.get("max_additional_steps", 1)
                    )
                elif degrade_remaining == 0:
                    # Budget exhausted from the previous degraded step.
                    # Stop before executing again.
                    print("           stop=DEGRADE budget exhausted")
                    break
                # else: degrade_remaining > 0, continue in degrade window
                executed_in_degrade = True
                action.payload["_rnos_constraints"] = assessment.constraints
                print(
                    "           degraded_mode=True "
                    f"constraints={json.dumps(assessment.constraints, sort_keys=True)}"
                )
            elif assessment.decision is PolicyDecision.ALLOW:
                # Trust recovered: clear the degrade budget so re-entry initialises fresh.
                degrade_remaining = None

        # --- unknown-tool guard ----------------------------------------------
        if action.tool_name != "unstable_api":
            print("           tool_result=SKIPPED (planner requested unknown tool)")
            if not no_rnos and not circuit_breaker:
                rnos.record_outcome(action, success=False)
            total_failures += 1
            steps_executed += 1
            history.append(
                {
                    "step": step,
                    "llm_output": llm_output,
                    "tool": action.tool_name,
                    "result": "unknown_tool",
                }
            )
            break

        # --- tool execution --------------------------------------------------
        tool_t0 = time.monotonic()
        result = tool.run(**action.payload)
        tool_latency_ms = (time.monotonic() - tool_t0) * 1000.0
        steps_executed += 1

        if circuit_breaker:
            assert cb is not None
            cb.record_result(success=result.success)
            cb_stats = cb.stats
            if cb_stats["current_cooldown_limit"] >= cb_max_cooldown:
                max_cooldown_reached = True
        elif not no_rnos:
            rnos.record_outcome(action, success=result.success)

        if not result.success:
            total_failures += 1

        print(
            "           tool_result="
            f"{'SUCCESS' if result.success else 'FAILURE'} ({result.message})"
        )
        print(
            f"           phase={result.result_data.get('phase')} "
            f"call_count={result.result_data.get('call_count')} "
            f"failure_streak={result.result_data.get('failure_streak')}"
        )
        print(f"           result_data={json.dumps(result.result_data, sort_keys=True)}")
        print(f"           planner_latency_ms={planner_latency_ms:.1f} tool_latency_ms={tool_latency_ms:.1f}")

        if circuit_breaker:
            assert cb is not None
            cb_stats = cb.stats
            print(
                f"           breaker_state={cb.state} "
                f"cooldown_remaining={cb_stats['cooldown_remaining']} "
                f"consecutive_failures={cb_stats['consecutive_failures']} "
                f"total_blocked={cb_stats['total_blocked']}"
            )

        # Immediately consume one degrade credit after a degraded step executes
        # (RNOS only).
        if executed_in_degrade and degrade_remaining is not None:
            degrade_remaining -= 1
            print(f"           remaining_degraded_retries={degrade_remaining}")

        history.append(
            {
                "step": step,
                "llm_output": llm_output,
                "tool": action.tool_name,
                "decision": (
                    cb_reason if circuit_breaker
                    else ("bypass" if no_rnos else assessment.decision.value)
                ),
                "ok": result.success,
                "phase": result.result_data.get("phase"),
                "retry_count": retry_count,
            }
        )

        retry_count = 0 if result.success else retry_count + 1

    # =========================================================================

    # --- summary -------------------------------------------------------------
    if no_rnos:
        print("\n[BASELINE] RNOS was disabled for this run.")

    if circuit_breaker:
        assert cb is not None
        summary: dict[str, object] = {
            "mode": "circuit_breaker",
            "total_steps_executed": steps_executed,
            "total_tool_failures": total_failures,
            "total_blocked_steps": total_blocked_steps,
            "max_cooldown_reached": max_cooldown_reached,
            "final_breaker_state": cb.state,
            "seed": seed,
            "max_steps": max_steps,
            "cb_threshold": cb_threshold,
            "cb_cooldown": cb_cooldown,
            "cb_max_cooldown": cb_max_cooldown,
            "cb_max_blocked": cb_max_blocked,
        }

        print("\nSummary")
        print(f"  mode={summary['mode']}")
        print(f"  total_steps_executed={steps_executed}")
        print(f"  total_tool_failures={total_failures}")
        print(f"  total_blocked_steps={total_blocked_steps}")
        print(f"  max_cooldown_reached={max_cooldown_reached}")
        print(f"  final_breaker_state={cb.state}")
    else:
        summary = {
            "mode": "baseline" if no_rnos else "rnos",
            "total_steps_executed": steps_executed,
            "total_tool_failures": total_failures,
            "refused": refused,
            "final_entropy": final_entropy,
            "final_trust": final_trust,
            "seed": seed,
            "max_steps": max_steps,
        }

        print("\nSummary")
        print(f"  mode={summary['mode']}")
        print(f"  total_steps_executed={steps_executed}")
        print(f"  total_tool_failures={total_failures}")
        print(f"  refused={refused}")
        print(f"  final_entropy={final_entropy:.3f}")
        print(f"  final_trust={final_trust:.3f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the RNOS agent loop script."""
    parser = argparse.ArgumentParser(description="Run the LM Studio RNOS agent loop.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--no-rnos",
        action="store_true",
        help="Disable RNOS evaluation entirely (baseline comparison mode).",
    )
    parser.add_argument(
        "--circuit-breaker",
        action="store_true",
        help=(
            "Use exponential backoff circuit breaker instead of RNOS. "
            "Mutually exclusive with --no-rnos."
        ),
    )
    parser.add_argument(
        "--cb-threshold",
        type=int,
        default=3,
        metavar="N",
        help="Consecutive failures before the circuit breaker trips (default: 3).",
    )
    parser.add_argument(
        "--cb-cooldown",
        type=int,
        default=1,
        metavar="N",
        help="Initial cooldown steps before first HALF-OPEN probe (default: 1).",
    )
    parser.add_argument(
        "--cb-max-cooldown",
        type=int,
        default=8,
        metavar="N",
        help="Maximum cooldown ceiling after exponential growth (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replace the LM Studio planner with a stub (no LM Studio required).",
    )
    parser.add_argument(
        "--persona",
        choices=["adversarial", "cautious", "mixed"],
        default="adversarial",
        help="System-prompt strategy for the planner.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to a YAML file with 'policy' threshold overrides.",
    )
    args = parser.parse_args()

    if args.circuit_breaker and args.no_rnos:
        print("Error: --circuit-breaker and --no-rnos are mutually exclusive.")
        sys.exit(1)

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    summary = run_agent(
        max_steps=args.max_steps,
        seed=args.seed,
        no_rnos=args.no_rnos,
        circuit_breaker=args.circuit_breaker,
        cb_threshold=args.cb_threshold,
        cb_cooldown=args.cb_cooldown,
        cb_max_cooldown=args.cb_max_cooldown,
        dry_run=args.dry_run,
        persona=args.persona,
        config_path=args.config,
    )

    print("\n=== Summary JSON ===")
    print(json.dumps(summary, indent=2))
    print(f"Trace log written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
