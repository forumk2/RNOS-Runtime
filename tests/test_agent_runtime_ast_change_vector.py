from agent_runtime.ast_change_vector import (
    compute_change_vector,
    extract_features,
    summarize_change,
)
from agent_runtime.rnos_adapter import evaluate_state
from agent_runtime.types import ExecutionResult


def _vector(source_a: str, source_b: str) -> dict[str, int]:
    return compute_change_vector(extract_features(source_a), extract_features(source_b))


def test_change_vector_is_zero_for_same_code() -> None:
    source = """
def foo(x):
    return x
"""

    vector = _vector(source, source)

    assert all(delta == 0 for delta in vector.values())
    assert summarize_change(vector) == "no structural change"


def test_change_vector_detects_added_print_call() -> None:
    source_a = """
value = 1
"""
    source_b = """
value = 1
print(value)
"""

    vector = _vector(source_a, source_b)

    assert vector["call"] == 1
    assert summarize_change(vector) == "minor logic change"


def test_change_vector_detects_added_loop() -> None:
    source_a = """
print("ready")
"""
    source_b = """
for item in range(3):
    print(item)
"""

    vector = _vector(source_a, source_b)

    assert vector["for"] == 1
    assert summarize_change(vector) == "control flow change"


def test_change_vector_detects_function_wrap() -> None:
    source_a = """
print("ready")
"""
    source_b = """
def run():
    print("ready")
"""

    vector = _vector(source_a, source_b)

    assert vector["function_def"] == 1
    assert summarize_change(vector) == "function structure change"


def test_change_vector_detects_loop_to_function_rewrite() -> None:
    source_a = """
for item in range(3):
    print(item)
"""
    source_b = """
def run(items):
    for item in items:
        return item
"""

    vector = _vector(source_a, source_b)

    assert vector["function_def"] == 1
    assert vector["return"] == 1
    assert summarize_change(vector) == "major structural change"


def test_rnos_refuses_cosmetic_retry_loop() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            failure_type="SyntaxError",
        ),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="SyntaxError: invalid syntax",
            ast_similarity_to_previous=0.8,
            ast_progress_score=0.15,
            ast_change_summary="minor logic change",
            ast_change_vector={
                "if": 0,
                "for": 0,
                "while": 0,
                "try": 0,
                "function_def": 0,
                "call": 1,
                "assign": 0,
                "return": 0,
            },
            failure_type="SyntaxError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert decision.reason == "cosmetic retry loop"


def test_rnos_allows_failed_control_flow_change() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            failure_type="SyntaxError",
        ),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="SyntaxError: invalid syntax",
            ast_similarity_to_previous=0.9,
            ast_progress_score=0.45,
            ast_change_summary="control flow change",
            ast_change_vector={
                "if": 1,
                "for": 0,
                "while": 0,
                "try": 0,
                "function_def": 0,
                "call": 0,
                "assign": 0,
                "return": 0,
            },
            failure_type="SyntaxError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "retry"
    assert decision.reason == "meaningful structural attempt"
