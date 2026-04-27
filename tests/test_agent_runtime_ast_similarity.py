from agent_runtime.ast_diff import classify_change, compute_progress, flatten_ast
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


def test_ast_progress_is_zero_for_identical_structure() -> None:
    source = """
def foo(x):
    return x + 1
"""

    progress = compute_progress(flatten_ast(source), flatten_ast(source))

    assert progress == 0.0
    assert classify_change(progress) == "no_change"


def test_ast_progress_treats_variable_rename_as_fake_change() -> None:
    source_a = """
def foo(x):
    y = x + 1
    return y
"""
    source_b = """
def bar(value):
    result = value + 2
    return result
"""

    progress = compute_progress(flatten_ast(source_a), flatten_ast(source_b))

    assert progress < 0.1
    assert classify_change(progress) == "no_change"


def test_ast_progress_detects_small_patch() -> None:
    source_a = """
def foo(x):
    print(x)
"""
    source_b = """
def foo(x):
    print(x)
    return x
"""

    progress = compute_progress(flatten_ast(source_a), flatten_ast(source_b))

    assert 0.1 <= progress < 0.3
    assert classify_change(progress) == "minor_patch"


def test_ast_progress_detects_rewrite() -> None:
    source_a = """
for i in range(10):
    print(i)
"""
    source_b = """
def foo(x):
    return x + 1
"""

    progress = compute_progress(flatten_ast(source_a), flatten_ast(source_b))

    assert progress > 0.5
    assert classify_change(progress) in {"moderate_change", "major_change"}


def test_rnos_refuses_repeated_failed_no_progress_retry() -> None:
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
            ast_progress_score=0.0,
            ast_change_type="no_change",
            failure_type="SyntaxError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "refuse"
    assert decision.reason == "no structural progress"


def test_rnos_allows_meaningful_failed_attempt_to_retry() -> None:
    history = [
        ExecutionResult(
            success=False,
            output="validated workspace/step_01.py",
            error="SyntaxError: invalid syntax",
            ast_similarity_to_previous=None,
            failure_type="SyntaxError",
        ),
        ExecutionResult(
            success=False,
            output="validated workspace/step_02.py",
            error="RuntimeError: crashed",
            ast_similarity_to_previous=0.35,
            ast_progress_score=0.65,
            ast_change_type="major_change",
            failure_type="RuntimeError",
        ),
    ]

    decision = evaluate_state(history)

    assert decision.action == "retry"
    assert decision.reason == "meaningful attempt, allow exploration"
