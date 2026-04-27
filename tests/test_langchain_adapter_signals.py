import subprocess
import sys

from langchain_demo.langchain_adapter import (
    convert_to_execution_result,
    extract_python_code,
)


def _output(content: str, error: str | None = None) -> dict:
    payload = {"messages": [{"role": "assistant", "content": content}]}
    if error:
        payload["error"] = error
    return payload


def test_extract_fenced_python_code() -> None:
    text = """Here is the fix:

```python
def square(x):
    return x * x
```
"""

    code = extract_python_code(text)

    assert code is not None
    assert "def square" in code


def test_ast_mode_detects_structural_similarity() -> None:
    first = convert_to_execution_result(
        _output(
            """```python
def foo(x):
    return x + 1
```"""
        ),
        0,
    )
    second = convert_to_execution_result(
        _output(
            """```python
def bar(y):
    return y + 2
```"""
        ),
        1,
        previous_result=first,
    )

    assert second.signal_mode == "python_ast"
    assert second.ast_similarity_to_previous is not None
    assert second.ast_similarity_to_previous >= 0.85
    assert second.ast_progress_score is not None
    assert second.ast_progress_score <= 0.2


def test_ast_mode_detects_meaningful_change() -> None:
    first = convert_to_execution_result(
        _output(
            """```python
print(run(
```""",
            error="SyntaxError: '(' was never closed",
        ),
        0,
    )
    second = convert_to_execution_result(
        _output(
            """```python
def run():
    return 1
```"""
        ),
        1,
        previous_result=first,
    )

    assert second.signal_mode == "python_ast"
    assert second.ast_progress_score is not None
    assert second.ast_progress_score > 0.2
    assert second.ast_change_summary != "no structural change"


def test_text_fallback_detects_repeated_text() -> None:
    first = convert_to_execution_result(_output("Searching Atlantis weather..."), 0)
    second = convert_to_execution_result(
        _output("Searching Atlantis weather..."),
        1,
        previous_result=first,
    )

    assert second.signal_mode == "text_fallback"
    assert second.ast_similarity_to_previous == 1.0
    assert second.ast_progress_score == 0.0
    assert second.intent_class in {"no_intent", "cosmetic_intent"}


def test_langchain_benchmark_still_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "langchain_demo/langchain_benchmark.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SCENARIO: syntax_error_fix" in completed.stdout
    assert "RNOS stopped execution" in completed.stdout
    assert "SCENARIO: simple_success" in completed.stdout
