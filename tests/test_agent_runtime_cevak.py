from agent_runtime.cevak import compute_cevak
from agent_runtime.rnos_adapter import evaluate_state
from agent_runtime.types import ExecutionResult


def test_cevak_detects_repeated_failure_overreach() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            failure_type="SyntaxError",
        )
    ]
    result = ExecutionResult(
        success=False,
        output="validated workspace/step_02.py",
        error="SyntaxError: invalid syntax",
        ast_similarity_to_previous=0.8,
        ast_progress_score=0.0,
        failure_type="SyntaxError",
    )

    cevak = compute_cevak(result, history)

    assert cevak["confidence"] > 0.8
    assert cevak["evidence"] == 0.0
    assert cevak["drift_type"] == "overreach"


def test_cevak_detects_echo_chamber_agreement() -> None:
    history = [
        ExecutionResult(success=True, output="validated workspace/step_01.py"),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="RuntimeError: crashed",
            ast_progress_score=0.6,
            failure_type="RuntimeError",
        ),
    ]
    result = ExecutionResult(
        success=False,
        output="validated workspace/step_03.py",
        error="RuntimeError: crashed",
        ast_similarity_to_previous=0.9,
        ast_progress_score=0.6,
        failure_type="RuntimeError",
    )

    cevak = compute_cevak(result, history)

    assert cevak["agreement"] == 0.9
    assert cevak["variance"] < 0.2
    assert cevak["drift_type"] == "echo_chamber"


def test_cevak_detects_incoherent_attempts() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            ast_progress_score=0.0,
            failure_type="SyntaxError",
        )
    ]
    result = ExecutionResult(
        success=False,
        output="validated workspace/step_02.py",
        error="RuntimeError: crashed",
        ast_similarity_to_previous=0.1,
        ast_progress_score=1.0,
        failure_type="RuntimeError",
    )

    cevak = compute_cevak(result, history)

    assert cevak["consistency"] == 0.0
    assert cevak["drift_type"] == "incoherent"


def test_cevak_keeps_stable_success_stable() -> None:
    history = [
        ExecutionResult(success=True, output="validated workspace/step_01.py"),
        ExecutionResult(success=True, output="validated workspace/step_02.py"),
    ]
    result = ExecutionResult(
        success=True,
        output="validated workspace/step_03.py",
        ast_similarity_to_previous=1.0,
        ast_progress_score=0.0,
    )

    cevak = compute_cevak(result, history)

    assert cevak["evidence"] == 1.0
    assert cevak["drift_type"] == "stable"


def test_rnos_refuses_overconfident_stagnation() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            intent_class="no_intent",
            cevak={"drift_type": "overreach"},
            failure_type="SyntaxError",
        )
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert decision.reason == "overconfident stagnation"


def test_rnos_refuses_self_reinforcing_failure_loop() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="RuntimeError: crashed",
            failure_type="RuntimeError",
        ),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="RuntimeError: crashed",
            ast_similarity_to_previous=0.9,
            intent_class="weak_intent",
            cevak={"drift_type": "echo_chamber"},
            failure_type="RuntimeError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert decision.reason == "self-reinforcing failure loop"


def test_rnos_retries_incoherent_attempts() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="RuntimeError: crashed",
            intent_class="weak_intent",
            cevak={"drift_type": "incoherent"},
            failure_type="RuntimeError",
        )
    ]

    decision = evaluate_state(history)

    assert decision.action == "retry"
    assert decision.reason == "unstable attempts, allow exploration"
