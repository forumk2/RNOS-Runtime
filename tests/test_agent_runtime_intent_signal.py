from agent_runtime.intent_signal import (
    classify_intent,
    compute_intent_score,
    compute_structural_score,
)
from agent_runtime.rnos_adapter import evaluate_state
from agent_runtime.types import ExecutionResult


def _classify(
    similarity: float,
    progress_score: float,
    change_vector: dict[str, int],
) -> str:
    intent_score = compute_intent_score(similarity, progress_score, change_vector)
    return classify_intent(intent_score, progress_score, similarity, change_vector)


def test_intent_classifies_identical_retry_as_no_intent() -> None:
    intent_class = _classify(
        similarity=0.95,
        progress_score=0.0,
        change_vector={"if": 0, "for": 0, "while": 0, "try": 0, "function_def": 0},
    )

    assert intent_class == "no_intent"


def test_intent_classifies_variable_rename_as_cosmetic() -> None:
    intent_class = _classify(
        similarity=0.65,
        progress_score=0.05,
        change_vector={"if": 0, "for": 0, "while": 0, "try": 0, "function_def": 0},
    )

    assert intent_class == "cosmetic_intent"


def test_intent_classifies_print_patch_as_weak() -> None:
    intent_class = _classify(
        similarity=0.7,
        progress_score=0.25,
        change_vector={"if": 0, "for": 0, "while": 0, "try": 0, "function_def": 0},
    )

    assert intent_class == "weak_intent"


def test_intent_classifies_structural_change_as_exploratory() -> None:
    change_vector = {"if": 0, "for": 1, "while": 0, "try": 0, "function_def": 0}

    assert compute_structural_score(change_vector) > 0
    assert _classify(0.55, 0.45, change_vector) == "exploratory_intent"


def test_intent_classifies_major_rewrite_as_strong() -> None:
    intent_class = _classify(
        similarity=0.3,
        progress_score=0.7,
        change_vector={"if": 0, "for": 0, "while": 0, "try": 0, "function_def": 0},
    )

    assert intent_class == "strong_intent"


def test_rnos_refuses_no_intent_failure() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            intent_score=0.01,
            intent_class="no_intent",
            failure_type="SyntaxError",
        )
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert decision.reason == "no meaningful attempt"


def test_rnos_retries_exploratory_intent_failure() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            ast_similarity_to_previous=0.9,
            ast_progress_score=0.45,
            ast_change_vector={
                "if": 0,
                "for": 1,
                "while": 0,
                "try": 0,
                "function_def": 0,
                "call": 0,
                "assign": 0,
                "return": 0,
            },
            intent_score=0.52,
            intent_class="exploratory_intent",
            failure_type="SyntaxError",
        )
    ]

    decision = evaluate_state(history)

    assert decision.action == "retry"
    assert decision.reason == "valid structural exploration"
