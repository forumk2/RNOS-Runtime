from dataclasses import dataclass, replace
import logging
from pathlib import Path

from .agent import AgentState, DeterministicScenarioAgent, ScenarioSpec, constrain_plan
from .drift_model import DriftAssessment, assess_drift
from .risk_model import RiskAssessment, assess_tool_risk
from .rnos_bridge import GateDecision, RNOSBridge, RNOSContext
from .tool_executor import ToolExecutionResult, ToolExecutor
from . import executor, planner, rnos_adapter, validator
from .ast_change_vector import compute_change_vector, summarize_change
from .ast_diff import classify_change, compute_progress
from .ast_similarity import ast_similarity_score
from .cevak import compute_cevak
from .intent_signal import classify_intent, compute_intent_score
from .types import ExecutionResult, RNOSDecision, Task
from .utils import cleanup_workspace


logger = logging.getLogger(__name__)


def _log_result(step: str, execution: ExecutionResult, validation: ExecutionResult, decision: RNOSDecision) -> None:
    logger.info("runner.step=%s", step)
    logger.info(
        "runner.execution success=%s output=%s error=%s",
        execution.success,
        execution.output,
        execution.error,
    )
    logger.info(
        "runner.validation success=%s output=%s error=%s",
        validation.success,
        validation.output,
        validation.error,
    )
    logger.info(
        "runner.rnos action=%s reason=%s instability=%.2f",
        decision.action,
        decision.reason,
        decision.entropy_score,
    )


def _read_artifact_source(result: ExecutionResult) -> str | None:
    if not result.artifact_path:
        return None

    try:
        return Path(result.artifact_path).read_text(encoding="utf-8")
    except OSError:
        return None


def _with_artifact_metadata(
    validation: ExecutionResult,
    execution: ExecutionResult,
) -> ExecutionResult:
    return replace(
        validation,
        artifact_path=validation.artifact_path or execution.artifact_path,
        ast_fingerprint=validation.ast_fingerprint or execution.ast_fingerprint,
        ast_similarity_to_previous=execution.ast_similarity_to_previous,
        ast_tokens=validation.ast_tokens or execution.ast_tokens,
        ast_progress_score=execution.ast_progress_score,
        ast_change_type=execution.ast_change_type,
        ast_features=validation.ast_features or execution.ast_features,
        ast_change_vector=execution.ast_change_vector,
        ast_change_summary=execution.ast_change_summary,
        intent_score=execution.intent_score,
        intent_class=execution.intent_class,
        cevak=execution.cevak,
    )


def run_task(task: Task) -> list[ExecutionResult]:
    cleanup_workspace()
    plan = planner.plan_task(task)
    history: list[ExecutionResult] = []
    previous_source: str | None = None
    previous_tokens: list[str] | None = None
    previous_features: dict[str, int] | None = None

    logger.info("runner.plan")
    for step in plan.steps:
        logger.info("  %s", step)

    for step in plan.steps:
        execution = executor.execute_step(step)
        current_source = _read_artifact_source(execution)
        if previous_source is not None and current_source is not None:
            progress_score = None
            change_type = None
            change_vector = None
            change_summary = None
            similarity_score = ast_similarity_score(previous_source, current_source)
            intent_score = None
            intent_class = None
            if previous_tokens is not None and execution.ast_tokens is not None:
                progress_score = compute_progress(previous_tokens, execution.ast_tokens)
                change_type = classify_change(progress_score)
            if previous_features is not None and execution.ast_features is not None:
                change_vector = compute_change_vector(
                    previous_features,
                    execution.ast_features,
                )
                change_summary = summarize_change(change_vector)
            if progress_score is not None and change_vector is not None:
                intent_score = compute_intent_score(
                    similarity_score,
                    progress_score,
                    change_vector,
                )
                intent_class = classify_intent(
                    intent_score,
                    progress_score,
                    similarity_score,
                    change_vector,
                )

            execution = replace(
                execution,
                ast_similarity_to_previous=similarity_score,
                ast_progress_score=progress_score,
                ast_change_type=change_type,
                ast_change_vector=change_vector,
                ast_change_summary=change_summary,
                intent_score=intent_score,
                intent_class=intent_class,
            )

        validation = validator.validate()
        validation = _with_artifact_metadata(validation, execution)
        validation = replace(validation, cevak=compute_cevak(validation, history))
        history.append(validation)

        if current_source is not None:
            previous_source = current_source
        if execution.ast_tokens is not None:
            previous_tokens = execution.ast_tokens
        if execution.ast_features is not None:
            previous_features = execution.ast_features

        decision = rnos_adapter.evaluate_state(history)
        _log_result(step, execution, validation, decision)

        if decision.action == "refuse":
            logger.error("RNOS REFUSAL TRIGGERED")
            break

    return history


@dataclass(frozen=True)
class StepTrace:
    step: int
    planned_tool: str
    executed_tool: str
    decision: str
    entropy: float
    tool_risk: float
    drift_score: float
    validation_success: bool | None
    refused: bool = False
    constrained: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ModeRunResult:
    mode: str
    attempts: int
    wasted: int
    refusal_step: int | None
    peak_entropy: float
    risk_escalations: int
    destructive_actions_prevented: int
    drift_detection_step: int | None
    trace: tuple[StepTrace, ...]


@dataclass(frozen=True)
class ScenarioComparison:
    scenario: ScenarioSpec
    naive: ModeRunResult
    rnos: ModeRunResult


def run_agent_gate_scenario(scenario: ScenarioSpec, *, mode: str) -> ModeRunResult:
    """Run one deterministic scenario in either naive or RNOS-gated mode."""

    if mode not in {"naive", "rnos"}:
        raise ValueError(f"unsupported mode: {mode}")

    state = AgentState(scenario_name=scenario.name, objective=scenario.objective)
    agent = DeterministicScenarioAgent(scenario)
    gate = RNOSBridge()
    tool_executor = ToolExecutor()
    gate_validator = validator.GateValidator()

    risk_scores: list[float] = []
    recent_errors: list[str] = []
    trace: list[StepTrace] = []
    wasted = 0
    refusal_step: int | None = None
    peak_entropy = 0.0
    risk_escalations = 0
    destructive_actions_prevented = 0
    drift_detection_step: int | None = None
    max_steps = scenario.naive_max_steps if mode == "naive" else scenario.rnos_max_steps

    for step in range(1, max_steps + 1):
        plan = agent.plan(state)
        if plan is None:
            break

        risk = assess_tool_risk(plan, risk_scores)
        drift = assess_drift(plan, state)
        if drift_detection_step is None and drift.score >= 4.5:
            drift_detection_step = step

        decision = _evaluate_gate(gate, state, risk, drift)
        peak_entropy = max(peak_entropy, decision.entropy)
        if risk.escalation:
            risk_escalations += 1

        if mode == "rnos" and decision.action == "REFUSE":
            refusal_step = step
            if risk.destructive:
                destructive_actions_prevented += 1
            trace.append(
                StepTrace(
                    step=step,
                    planned_tool=plan.tool,
                    executed_tool="-",
                    decision=decision.action,
                    entropy=decision.entropy,
                    tool_risk=risk.score,
                    drift_score=drift.score,
                    validation_success=None,
                    refused=True,
                    reason=", ".join(decision.reasons),
                )
            )
            break

        constrained = mode == "rnos" and decision.action == "DEGRADE"
        executable_plan = constrain_plan(plan) if constrained else plan
        result = tool_executor.execute(executable_plan)
        validation = gate_validator.validate(result, recent_errors)
        _update_state(state, plan, result, validation, decision)

        if not bool(validation["success"]):
            wasted += 1
            recent_errors.append(str(validation["error"]))

        risk_scores.append(risk.score)
        trace.append(
            StepTrace(
                step=step,
                planned_tool=plan.tool,
                executed_tool=executable_plan.tool,
                decision="ALLOW" if mode == "naive" else decision.action,
                entropy=decision.entropy,
                tool_risk=risk.score,
                drift_score=drift.score,
                validation_success=bool(validation["success"]),
                constrained=constrained,
                reason=", ".join(decision.reasons),
            )
        )

    return ModeRunResult(
        mode=mode,
        attempts=state.attempts,
        wasted=wasted,
        refusal_step=refusal_step,
        peak_entropy=round(peak_entropy, 3),
        risk_escalations=risk_escalations,
        destructive_actions_prevented=destructive_actions_prevented,
        drift_detection_step=drift_detection_step,
        trace=tuple(trace),
    )


def compare_agent_gate_scenario(scenario: ScenarioSpec) -> ScenarioComparison:
    return ScenarioComparison(
        scenario=scenario,
        naive=run_agent_gate_scenario(scenario, mode="naive"),
        rnos=run_agent_gate_scenario(scenario, mode="rnos"),
    )


def run_agent_gate_benchmark(scenarios: list[ScenarioSpec]) -> list[ScenarioComparison]:
    return [compare_agent_gate_scenario(scenario) for scenario in scenarios]


def format_agent_gate_results(comparisons: list[ScenarioComparison]) -> str:
    blocks: list[str] = []
    for comparison in comparisons:
        blocks.append(_format_scenario_result(comparison))
    return "\n\n".join(blocks)


def _evaluate_gate(
    gate: RNOSBridge,
    state: AgentState,
    risk: RiskAssessment,
    drift: DriftAssessment,
) -> GateDecision:
    context = RNOSContext(
        entropy=0.0,
        retry_count=state.retry_count,
        drift_score=drift.score,
        tool_risk=risk.score,
        validation_failures=state.validation_failures,
        destructive_action=risk.destructive,
        risk_escalation=risk.escalation,
    )
    return gate.evaluate(context)


def _update_state(
    state: AgentState,
    plan,
    result: ToolExecutionResult,
    validation: dict[str, object],
    decision: GateDecision,
) -> None:
    success = bool(validation["success"])
    state.attempts += 1
    state.retry_count = 0 if success else state.retry_count + 1
    if not success:
        state.validation_failures += 1
    if plan.tool == "edit_file" and plan.target and not success:
        state.failed_edit_targets.append(plan.target)
    if plan.target:
        state.targets.append(plan.target)
    state.plan_texts.append(plan.text)
    state.decisions.append(decision.action)
    if result.metadata.get("constrained_from"):
        state.degraded = True


def _format_scenario_result(comparison: ScenarioComparison) -> str:
    lines = [
        f"Scenario: {comparison.scenario.name}",
        comparison.scenario.description,
        "",
        "Mode   Attempts  Wasted  Refusal Step  Peak Entropy",
        "-----  --------  ------  ------------  ------------",
    ]
    for result in (comparison.naive, comparison.rnos):
        refusal = f"Step {result.refusal_step}" if result.refusal_step is not None else "-"
        mode_label = "RNOS" if result.mode == "rnos" else result.mode.title()
        lines.append(
            f"{mode_label:<5}  "
            f"{result.attempts:<8}  "
            f"{result.wasted:<6}  "
            f"{refusal:<12}  "
            f"{result.peak_entropy:<12.2f}"
        )

    rnos = comparison.rnos
    drift_step = f"Step {rnos.drift_detection_step}" if rnos.drift_detection_step else "-"
    gate_events = _format_gate_events(rnos)
    lines.extend(
        [
            "",
            f"Tool Risk Escalations: {rnos.risk_escalations}",
            f"Destructive Actions Prevented: {rnos.destructive_actions_prevented}",
            f"Drift Detection Step: {drift_step}",
            f"RNOS Gate Events: {gate_events}",
        ]
    )
    return "\n".join(lines)


def _format_gate_events(result: ModeRunResult) -> str:
    events = [
        f"Step {trace.step} {trace.decision}"
        for trace in result.trace
        if trace.decision in {"DEGRADE", "REFUSE"}
    ]
    return ", ".join(events) if events else "-"
