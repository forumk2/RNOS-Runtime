from agent_runtime.ast_similarity import ast_similarity_score
from agent_runtime.rnos_adapter import evaluate_state
from agent_runtime.types import ExecutionResult


def test_ast_similarity_ignores_names_and_literal_values() -> None:
    source_a = """
def foo(x):
    return x + 1
"""
    source_b = """
def bar(y):
    return y + 2
"""

    assert ast_similarity_score(source_a, source_b) >= 0.85


def test_ast_similarity_distinguishes_different_structure() -> None:
    source_a = """
def foo(x):
    return x + 1
"""
    source_b = """
for i in range(10):
    print(i)
"""

    assert ast_similarity_score(source_a, source_b) < 0.75


def test_rnos_refuses_repeated_failed_structural_retry() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            artifact_path="workspace/step_01.py",
            ast_fingerprint="fingerprint-a",
            failure_type="SyntaxError",
        ),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="SyntaxError: invalid syntax",
            artifact_path="workspace/step_02.py",
            ast_fingerprint="fingerprint-b",
            ast_similarity_to_previous=0.92,
            failure_type="SyntaxError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert "structural retry loop" in decision.reason
