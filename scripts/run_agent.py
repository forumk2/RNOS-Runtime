"""Run an LM Studio-driven RNOS intervention scenario."""

from __future__ import annotations

import argparse
import datetime
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
from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from baselines.circuit_breaker import CircuitBreaker
from rnos.coherence import compute_runtime_coherence, format_runtime_coherence_report
from rnos.hybrid import HybridController
from rnos.logger import write_trace
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import PolicyDecision
from tools.unstable_api import UnstableAPI, UnstableAPITool

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "rnos_trace.jsonl"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "runs.jsonl"
_VALID_PHASES = {"stable", "unstable", "collapse"}


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


def _resolve_phase(
    observed_phase: str | None,
    *,
    last_phase: str,
) -> tuple[str, str]:
    """Return ``(phase, source)`` for the step trace."""
    if observed_phase in _VALID_PHASES:
        return observed_phase, "observed"
    return last_phase, "carried_forward"


def _append_execution_step(
    step_trace: list[dict[str, object]],
    *,
    trace_path: Path,
    step: int,
    tool: str,
    phase: str,
    phase_source: str,
    decision: str,
    decision_raw: str,
    tool_result: str,
    tool_result_raw: str,
    consecutive_failures: int,
    cooldown_remaining: int,
    planner_latency_ms: float,
    planner_emitted_tool_call: bool,
) -> None:
    """Append a canonical execution-step record and mirror it to JSONL."""
    record = {
        "stage": "execution_step",
        "step": step,
        "tool": tool,
        "phase": phase,
        "phase_source": phase_source,
        "decision": decision,
        "decision_raw": decision_raw,
        "tool_result": tool_result,
        "tool_result_raw": tool_result_raw,
        "consecutive_failures": consecutive_failures,
        "cooldown_remaining": cooldown_remaining,
        "planner_latency_ms": planner_latency_ms,
        "planner_emitted_tool_call": planner_emitted_tool_call,
    }
    step_trace.append(record)
    write_trace(record, path=trace_path)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_agent(
    *,
    max_steps: int,
    seed: int,
    no_rnos: bool = False,
    circuit_breaker: bool = False,
    hybrid: bool = False,
    cb_threshold: int = 3,
    cb_cooldown: int = 1,
    cb_max_cooldown: int = 8,
    cb_max_blocked: int = 10,
    acb_window: int = 10,
    acb_threshold: float = 0.6,
    acb_cooldown: int = 3,
    dry_run: bool = False,
    persona: PersonaName = "adversarial",
    config_path: Path | None = None,
    tag: str = "",
) -> dict[str, object]:
    """Execute an RNOS-controlled, circuit-breaker-controlled, or bypassed tool loop.

    Args:
        max_steps: Maximum number of loop iterations.
        seed: Random seed for the :class:`UnstableAPI` failure simulation.
        no_rnos: When True, skip all RNOS evaluation (decision printed as
            ``BYPASS``). The loop runs until ``max_steps`` is exhausted.
        circuit_breaker: When True, use exponential backoff circuit breaker
            instead of RNOS. Mutually exclusive with ``no_rnos`` and ``hybrid``.
        hybrid: When True, compose RNOS + AdaptiveCircuitBreaker with
            safety-first merge. Mutually exclusive with ``circuit_breaker``
            and ``no_rnos``.
        cb_threshold: Consecutive failures before the circuit breaker trips.
        cb_cooldown: Initial cooldown steps before the first probe.
        cb_max_cooldown: Maximum cooldown ceiling after exponential growth.
        cb_max_blocked: Total blocked steps before permanently opening.
        acb_window: Sliding window size for the adaptive CB used in hybrid mode.
        acb_threshold: Failure-rate threshold for the adaptive CB (0.0–1.0).
        acb_cooldown: Initial cooldown steps for the adaptive CB.
        dry_run: Replace :class:`Planner` with a stub; no LM Studio required.
        persona: System-prompt persona forwarded to :class:`Planner`.
        config_path: Optional path to a YAML file with ``policy`` thresholds.
        tag: Free-text label stored in the summary for later filtering.
    """

    t_run_start = time.monotonic()
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
    hybrid_ctrl: HybridController | None = None
    total_blocked_steps = 0
    max_cooldown_reached = False
    if circuit_breaker:
        cb = CircuitBreaker(
            failure_threshold=cb_threshold,
            initial_cooldown_steps=cb_cooldown,
            max_cooldown_steps=cb_max_cooldown,
            max_total_blocked=cb_max_blocked,
        )
    elif hybrid:
        _acb = AdaptiveCircuitBreaker(
            window_size=acb_window,
            initial_failure_rate=acb_threshold,
            initial_cooldown_steps=acb_cooldown,
        )
        hybrid_ctrl = HybridController(rnos, _acb)

    # --- loop state ----------------------------------------------------------
    history: list[dict[str, object]] = []
    retry_count = 0
    refused = False
    total_failures = 0
    steps_executed = 0
    final_entropy = 0.0
    final_trust = 0.0
    last_step = 0
    last_phase = "stable"
    step_trace: list[dict[str, object]] = []

    # Latency accumulators
    planner_latency_total_ms = 0.0
    tool_latency_total_ms = 0.0

    # Intervention tracking (first non-ALLOW / non-CLOSED event)
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None

    # Stop reason (set when loop breaks before max_steps)
    stop_reason = "completed"

    # Degrade budget (RNOS mode only).
    # Invariant — degrade_remaining is decremented immediately after each
    # degraded tool execution. A value of 0 means budget exhausted; detected
    # before the NEXT tool is allowed to run. None = not in a degrade window.
    degrade_remaining: int | None = None

    # --- header --------------------------------------------------------------
    if dry_run:
        print("[DRY RUN] LM Studio not called -- planner returns 'CALL unstable_api' always")
    print("=== LM Studio RNOS Loop ===")
    if circuit_breaker:
        mode_label = "circuit_breaker"
        print(
            f"mode={mode_label} seed={seed} max_steps={max_steps} persona={persona} "
            f"cb_threshold={cb_threshold} cb_cooldown={cb_cooldown} "
            f"cb_max_cooldown={cb_max_cooldown} cb_max_blocked={cb_max_blocked}"
        )
    elif hybrid:
        mode_label = "hybrid"
        print(
            f"mode={mode_label} seed={seed} max_steps={max_steps} persona={persona} "
            f"acb_window={acb_window} acb_threshold={acb_threshold} "
            f"acb_cooldown={acb_cooldown}"
        )
    else:
        mode_label = "baseline (--no-rnos)" if no_rnos else "rnos"
        print(f"mode={mode_label} seed={seed} max_steps={max_steps} persona={persona}")
    if config_path:
        print(f"config={config_path}")

    # =========================================================================
    for step in range(1, max_steps + 1):
        last_step = step

        # --- planner call (measure wall-clock time for latency signal) -------
        t0 = time.monotonic()
        llm_output = planner.get_next_action(history)
        planner_latency_ms = (time.monotonic() - t0) * 1000.0
        planner_latency_total_ms += planner_latency_ms

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
        assessment = None           # RNOS only; avoids NameError in history append
        hybrid_decision = None      # hybrid mode only

        if hybrid:
            # -----------------------------------------------------------------
            # Hybrid path -- RNOS + AdaptiveCircuitBreaker, safety-first merge
            # -----------------------------------------------------------------
            assert hybrid_ctrl is not None
            hybrid_ctrl.tick()
            hybrid_decision = hybrid_ctrl.evaluate(action)

            final_entropy = hybrid_decision.rnos_entropy
            final_trust = hybrid_decision.rnos_trust

            print(
                f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
                f"entropy={hybrid_decision.rnos_entropy:.3f} "
                f"trust={hybrid_decision.rnos_trust:.3f} "
                f"rnos={hybrid_decision.rnos_decision} "
                f"cb_state={hybrid_decision.cb_state} "
                f"cb_failure_rate={hybrid_decision.cb_failure_rate:.3f} "
                f"hybrid={hybrid_decision.decision} "
                f"trigger={hybrid_decision.trigger_source}"
            )

            if hybrid_decision.decision == "REFUSE":
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_type = "refuse"
                stop_reason = "refused"
                refused = True
                phase, phase_source = _resolve_phase(None, last_phase=last_phase)
                _append_execution_step(
                    step_trace,
                    trace_path=TRACE_PATH,
                    step=step,
                    tool=action.tool_name,
                    phase=phase,
                    phase_source=phase_source,
                    decision="STOPPED",
                    decision_raw=f"HYBRID_REFUSE ({hybrid_decision.trigger_source})",
                    tool_result="BLOCKED",
                    tool_result_raw="BLOCKED",
                    consecutive_failures=retry_count,
                    cooldown_remaining=0,
                    planner_latency_ms=planner_latency_ms,
                    planner_emitted_tool_call=bool(action.tool_name),
                )
                print("           stop=HYBRID refused execution")
                break

            if hybrid_decision.decision == "DEGRADE":
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_type = "degrade"
                if degrade_remaining is None:
                    rnos_constraints = hybrid_decision.rnos_assessment.constraints
                    degrade_remaining = int(rnos_constraints.get("max_additional_steps", 1))
                elif degrade_remaining == 0:
                    stop_reason = "degrade_exhausted"
                    phase, phase_source = _resolve_phase(None, last_phase=last_phase)
                    _append_execution_step(
                        step_trace,
                        trace_path=TRACE_PATH,
                        step=step,
                        tool=action.tool_name,
                        phase=phase,
                        phase_source=phase_source,
                        decision="STOPPED",
                        decision_raw="HYBRID_DEGRADE_EXHAUSTED",
                        tool_result="BLOCKED",
                        tool_result_raw="BLOCKED",
                        consecutive_failures=retry_count,
                        cooldown_remaining=0,
                        planner_latency_ms=planner_latency_ms,
                        planner_emitted_tool_call=bool(action.tool_name),
                    )
                    print("           stop=HYBRID DEGRADE budget exhausted")
                    break
                executed_in_degrade = True
                action.payload["_rnos_constraints"] = hybrid_decision.rnos_assessment.constraints
            elif hybrid_decision.decision == "ALLOW":
                degrade_remaining = None

        elif circuit_breaker:
            # -----------------------------------------------------------------
            # Circuit breaker path -- no RNOS calls
            # -----------------------------------------------------------------
            assert cb is not None
            cb.tick()
            allowed, cb_reason = cb.should_execute()

            if not allowed:
                cb_stats = cb.stats
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_type = "blocked"
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
                phase, phase_source = _resolve_phase(None, last_phase=last_phase)
                _append_execution_step(
                    step_trace,
                    trace_path=TRACE_PATH,
                    step=step,
                    tool=action.tool_name,
                    phase=phase,
                    phase_source=phase_source,
                    decision="STOPPED" if cb_reason == "permanently_open" else "BLOCKED",
                    decision_raw=decision_str,
                    tool_result="BLOCKED",
                    tool_result_raw="BLOCKED",
                    consecutive_failures=int(cb_stats["consecutive_failures"]),
                    cooldown_remaining=int(cb_stats["cooldown_remaining"]),
                    planner_latency_ms=planner_latency_ms,
                    planner_emitted_tool_call=bool(action.tool_name),
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
                    stop_reason = "permanently_open"
                    break
                continue

            # Allowed -- track first HALF_OPEN and print step header
            if cb_reason == "half_open_probe" and first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "half_open"

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
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_type = "refuse"
                stop_reason = "refused"
                refused = True
                phase, phase_source = _resolve_phase(None, last_phase=last_phase)
                _append_execution_step(
                    step_trace,
                    trace_path=TRACE_PATH,
                    step=step,
                    tool=action.tool_name,
                    phase=phase,
                    phase_source=phase_source,
                    decision="STOPPED",
                    decision_raw=assessment.decision.value.upper(),
                    tool_result="BLOCKED",
                    tool_result_raw="BLOCKED",
                    consecutive_failures=retry_count,
                    cooldown_remaining=0,
                    planner_latency_ms=planner_latency_ms,
                    planner_emitted_tool_call=bool(action.tool_name),
                )
                print("           stop=RNOS refused execution")
                break

            if assessment.decision is PolicyDecision.DEGRADE:
                if first_intervention_step is None:
                    first_intervention_step = step
                    first_intervention_type = "degrade"
                if degrade_remaining is None:
                    degrade_remaining = int(
                        assessment.constraints.get("max_additional_steps", 1)
                    )
                elif degrade_remaining == 0:
                    stop_reason = "degrade_exhausted"
                    phase, phase_source = _resolve_phase(None, last_phase=last_phase)
                    _append_execution_step(
                        step_trace,
                        trace_path=TRACE_PATH,
                        step=step,
                        tool=action.tool_name,
                        phase=phase,
                        phase_source=phase_source,
                        decision="STOPPED",
                        decision_raw="DEGRADE_EXHAUSTED",
                        tool_result="BLOCKED",
                        tool_result_raw="BLOCKED",
                        consecutive_failures=retry_count,
                        cooldown_remaining=0,
                        planner_latency_ms=planner_latency_ms,
                        planner_emitted_tool_call=bool(action.tool_name),
                    )
                    print("           stop=DEGRADE budget exhausted")
                    break
                executed_in_degrade = True
                action.payload["_rnos_constraints"] = assessment.constraints
                print(
                    "           degraded_mode=True "
                    f"constraints={json.dumps(assessment.constraints, sort_keys=True)}"
                )
            elif assessment.decision is PolicyDecision.ALLOW:
                degrade_remaining = None

        # --- unknown-tool guard ----------------------------------------------
        if action.tool_name != "unstable_api":
            print("           tool_result=SKIPPED (planner requested unknown tool)")
            if hybrid:
                assert hybrid_ctrl is not None
                hybrid_ctrl.record_outcome(action, success=False)
            elif not no_rnos and not circuit_breaker:
                rnos.record_outcome(action, success=False)
            total_failures += 1
            steps_executed += 1
            phase, phase_source = _resolve_phase(None, last_phase=last_phase)
            if circuit_breaker:
                assert cb is not None
                consecutive_failures = int(cb.stats["consecutive_failures"])
                cooldown_remaining = int(cb.stats["cooldown_remaining"])
                decision_raw = decision_str
            elif hybrid:
                consecutive_failures = retry_count + 1
                cooldown_remaining = 0
                decision_raw = f"HYBRID_{hybrid_decision.decision}" if hybrid_decision else "HYBRID_ALLOW"
            else:
                consecutive_failures = retry_count + 1
                cooldown_remaining = 0
                decision_raw = "BYPASS" if no_rnos else assessment.decision.value.upper()
            _append_execution_step(
                step_trace,
                trace_path=TRACE_PATH,
                step=step,
                tool=action.tool_name,
                phase=phase,
                phase_source=phase_source,
                decision="EXECUTE",
                decision_raw=decision_raw,
                tool_result="FAILURE",
                tool_result_raw="UNKNOWN_TOOL",
                consecutive_failures=consecutive_failures,
                cooldown_remaining=cooldown_remaining,
                planner_latency_ms=planner_latency_ms,
                planner_emitted_tool_call=bool(action.tool_name),
            )
            history.append(
                {
                    "step": step,
                    "llm_output": llm_output,
                    "tool": action.tool_name,
                    "result": "unknown_tool",
                }
            )
            stop_reason = "unknown_tool"
            break

        # --- tool execution --------------------------------------------------
        tool_t0 = time.monotonic()
        result = tool.run(**action.payload)
        tool_latency_ms = (time.monotonic() - tool_t0) * 1000.0
        tool_latency_total_ms += tool_latency_ms
        steps_executed += 1

        if circuit_breaker:
            assert cb is not None
            cb.record_result(success=result.success)
            cb_stats = cb.stats
            if cb_stats["current_cooldown_limit"] >= cb_max_cooldown:
                max_cooldown_reached = True
        elif hybrid:
            assert hybrid_ctrl is not None
            hybrid_ctrl.record_outcome(action, success=result.success)
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
        print(
            f"           planner_latency_ms={planner_latency_ms:.1f} "
            f"tool_latency_ms={tool_latency_ms:.1f}"
        )

        if circuit_breaker:
            assert cb is not None
            cb_stats = cb.stats
            print(
                f"           breaker_state={cb.state} "
                f"cooldown_remaining={cb_stats['cooldown_remaining']} "
                f"consecutive_failures={cb_stats['consecutive_failures']} "
                f"total_blocked={cb_stats['total_blocked']}"
            )
            consecutive_failures = int(cb_stats["consecutive_failures"])
            cooldown_remaining = int(cb_stats["cooldown_remaining"])
            decision_raw = decision_str
        elif hybrid:
            assert hybrid_decision is not None
            print(
                f"           hybrid_state={hybrid_decision.cb_state} "
                f"cb_failure_rate={hybrid_decision.cb_failure_rate:.3f} "
                f"trigger={hybrid_decision.trigger_source}"
            )
            consecutive_failures = retry_count + 1 if not result.success else 0
            cooldown_remaining = 0
            decision_raw = f"HYBRID_{hybrid_decision.decision}"
        else:
            consecutive_failures = 0 if result.success else retry_count + 1
            cooldown_remaining = 0
            decision_raw = "BYPASS" if no_rnos else assessment.decision.value.upper()

        observed_phase = result.result_data.get("phase")
        phase, phase_source = _resolve_phase(observed_phase, last_phase=last_phase)
        last_phase = phase
        _append_execution_step(
            step_trace,
            trace_path=TRACE_PATH,
            step=step,
            tool=action.tool_name,
            phase=phase,
            phase_source=phase_source,
            decision="EXECUTE",
            decision_raw=decision_raw,
            tool_result="SUCCESS" if result.success else "FAILURE",
            tool_result_raw="SUCCESS" if result.success else "FAILURE",
            consecutive_failures=consecutive_failures,
            cooldown_remaining=cooldown_remaining,
            planner_latency_ms=planner_latency_ms,
            planner_emitted_tool_call=bool(action.tool_name),
        )

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
                    else (
                        hybrid_decision.decision.lower() if hybrid and hybrid_decision
                        else ("bypass" if no_rnos else assessment.decision.value)
                    )
                ),
                "ok": result.success,
                "phase": result.result_data.get("phase"),
                "retry_count": retry_count,
            }
        )

        retry_count = 0 if result.success else retry_count + 1

    # =========================================================================

    duration_seconds = time.monotonic() - t_run_start

    # --- summary -------------------------------------------------------------
    if no_rnos:
        print("\n[BASELINE] RNOS was disabled for this run.")

    coherence_report = compute_runtime_coherence(step_trace)

    if hybrid:
        assert hybrid_ctrl is not None
        _hcb = hybrid_ctrl.cb
        summary: dict[str, object] = {
            "mode": "hybrid",
            "total_loop_steps": last_step,
            "total_steps_executed": steps_executed,
            "total_tool_failures": total_failures,
            "refused": refused,
            "final_entropy": final_entropy,
            "final_trust": final_trust,
            "final_state": stop_reason,
            "first_intervention_step": first_intervention_step,
            "first_intervention_type": first_intervention_type,
            "acb_window": acb_window,
            "acb_threshold": acb_threshold,
            "duration_seconds": round(duration_seconds, 3),
            "planner_latency_total_ms": round(planner_latency_total_ms, 1),
            "tool_latency_total_ms": round(tool_latency_total_ms, 1),
            "seed": seed,
            "max_steps": max_steps,
            "execution_trace": step_trace,
            "runtime_coherence": coherence_report,
        }
        print("\nSummary")
        print(f"  mode={summary['mode']}")
        print(f"  total_loop_steps={last_step}")
        print(f"  total_steps_executed={steps_executed}")
        print(f"  total_tool_failures={total_failures}")
        print(f"  refused={refused}")
        print(f"  final_entropy={final_entropy:.3f}")
        print(f"  final_trust={final_trust:.3f}")
        print(f"  final_state={stop_reason}")
        print(f"  first_intervention_step={first_intervention_step}")
        print(f"  first_intervention_type={first_intervention_type}")
        print(f"  duration_seconds={duration_seconds:.3f}")
    elif circuit_breaker:
        assert cb is not None
        summary = {
            "mode": "circuit_breaker",
            "total_loop_steps": last_step,
            "total_steps_executed": steps_executed,
            "total_tool_failures": total_failures,
            "total_blocked_steps": total_blocked_steps,
            "max_cooldown_reached": max_cooldown_reached,
            "final_breaker_state": cb.state,
            "final_state": stop_reason,
            "first_intervention_step": first_intervention_step,
            "first_intervention_type": first_intervention_type,
            "duration_seconds": round(duration_seconds, 3),
            "planner_latency_total_ms": round(planner_latency_total_ms, 1),
            "tool_latency_total_ms": round(tool_latency_total_ms, 1),
            "seed": seed,
            "max_steps": max_steps,
            "cb_threshold": cb_threshold,
            "cb_cooldown": cb_cooldown,
            "cb_max_cooldown": cb_max_cooldown,
            "cb_max_blocked": cb_max_blocked,
            "execution_trace": step_trace,
            "runtime_coherence": coherence_report,
        }

        print("\nSummary")
        print(f"  mode={summary['mode']}")
        print(f"  total_loop_steps={last_step}")
        print(f"  total_steps_executed={steps_executed}")
        print(f"  total_tool_failures={total_failures}")
        print(f"  total_blocked_steps={total_blocked_steps}")
        print(f"  max_cooldown_reached={max_cooldown_reached}")
        print(f"  final_breaker_state={cb.state}")
        print(f"  final_state={stop_reason}")
        print(f"  first_intervention_step={first_intervention_step}")
        print(f"  first_intervention_type={first_intervention_type}")
        print(f"  duration_seconds={duration_seconds:.3f}")
        print(f"  planner_latency_total_ms={planner_latency_total_ms:.1f}")
        print(f"  tool_latency_total_ms={tool_latency_total_ms:.1f}")
    else:
        summary = {
            "mode": "baseline" if no_rnos else "rnos",
            "total_loop_steps": last_step,
            "total_steps_executed": steps_executed,
            "total_tool_failures": total_failures,
            "refused": refused,
            "final_entropy": final_entropy,
            "final_trust": final_trust,
            "final_state": stop_reason,
            "first_intervention_step": first_intervention_step,
            "first_intervention_type": first_intervention_type,
            "duration_seconds": round(duration_seconds, 3),
            "planner_latency_total_ms": round(planner_latency_total_ms, 1),
            "tool_latency_total_ms": round(tool_latency_total_ms, 1),
            "seed": seed,
            "max_steps": max_steps,
            "execution_trace": step_trace,
            "runtime_coherence": coherence_report,
        }

        print("\nSummary")
        print(f"  mode={summary['mode']}")
        print(f"  total_loop_steps={last_step}")
        print(f"  total_steps_executed={steps_executed}")
        print(f"  total_tool_failures={total_failures}")
        print(f"  refused={refused}")
        print(f"  final_entropy={final_entropy:.3f}")
        print(f"  final_trust={final_trust:.3f}")
        print(f"  final_state={stop_reason}")
        print(f"  first_intervention_step={first_intervention_step}")
        print(f"  first_intervention_type={first_intervention_type}")
        print(f"  duration_seconds={duration_seconds:.3f}")
        print(f"  planner_latency_total_ms={planner_latency_total_ms:.1f}")
        print(f"  tool_latency_total_ms={tool_latency_total_ms:.1f}")

    print("\nRuntime Coherence Metrics v0.1")
    print(format_runtime_coherence_report(coherence_report))

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
        "--hybrid",
        action="store_true",
        help=(
            "Compose RNOS + AdaptiveCircuitBreaker with safety-first merge. "
            "Mutually exclusive with --circuit-breaker and --no-rnos."
        ),
    )
    parser.add_argument(
        "--acb-window",
        type=int,
        default=10,
        metavar="N",
        help="Sliding window size for the hybrid AdaptiveCircuitBreaker (default: 10).",
    )
    parser.add_argument(
        "--acb-threshold",
        type=float,
        default=0.6,
        metavar="F",
        help="Failure-rate threshold for the hybrid AdaptiveCircuitBreaker (default: 0.6).",
    )
    parser.add_argument(
        "--acb-cooldown",
        type=int,
        default=3,
        metavar="N",
        help="Initial cooldown steps for the hybrid AdaptiveCircuitBreaker (default: 3).",
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
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        metavar="TEXT",
        help="Free-text label stored with the run for later filtering.",
    )
    args = parser.parse_args()

    if args.circuit_breaker and args.no_rnos:
        print("Error: --circuit-breaker and --no-rnos are mutually exclusive.")
        sys.exit(1)
    if args.hybrid and (args.circuit_breaker or args.no_rnos):
        print("Error: --hybrid is mutually exclusive with --circuit-breaker and --no-rnos.")
        sys.exit(1)

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    summary = run_agent(
        max_steps=args.max_steps,
        seed=args.seed,
        no_rnos=args.no_rnos,
        circuit_breaker=args.circuit_breaker,
        hybrid=args.hybrid,
        cb_threshold=args.cb_threshold,
        cb_cooldown=args.cb_cooldown,
        cb_max_cooldown=args.cb_max_cooldown,
        cb_max_blocked=10,
        acb_window=args.acb_window,
        acb_threshold=args.acb_threshold,
        acb_cooldown=args.acb_cooldown,
        dry_run=args.dry_run,
        persona=args.persona,
        config_path=args.config,
        tag=args.tag,
    )

    print("\n=== Summary JSON ===")
    print(json.dumps(summary, indent=2))

    # --- append run record to results/runs.jsonl ----------------------------
    summary["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    summary["persona"] = args.persona
    summary["config"] = str(args.config) if args.config else None
    summary["dry_run"] = args.dry_run
    summary["tag"] = args.tag

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print(f"Trace log written to {TRACE_PATH}")
    print(f"Results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
