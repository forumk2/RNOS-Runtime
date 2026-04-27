import difflib
import hashlib
import re

from agent_runtime.ast_change_vector import (
    compute_change_vector,
    extract_features,
    summarize_change,
)
from agent_runtime.ast_diff import classify_change, compute_progress, flatten_ast
from agent_runtime.ast_similarity import ast_fingerprint, ast_similarity_score
from agent_runtime.intent_signal import classify_intent, compute_intent_score
from agent_runtime.types import ExecutionResult


ZERO_CHANGE_VECTOR = {
    "if": 0,
    "for": 0,
    "while": 0,
    "try": 0,
    "function_def": 0,
    "call": 0,
    "assign": 0,
    "return": 0,
}

PYTHON_LINE_PREFIXES = (
    "def ",
    "class ",
    "import ",
    "from ",
    "print(",
    "return ",
    "for ",
    "while ",
    "if ",
    "try:",
)


def _extract_text(output) -> str:
    if isinstance(output, Exception):
        return str(output)
    if isinstance(output, dict):
        messages = output.get("messages")
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", last_message))
            return str(last_message)
        return str(output)
    return str(output)


def _extract_error(output) -> str | None:
    if isinstance(output, Exception):
        return str(output)
    if isinstance(output, dict) and output.get("error"):
        return str(output["error"])
    return None


def _looks_like_python(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(PYTHON_LINE_PREFIXES):
            return True
        if "=" in line and "==" not in line:
            return True
    return False


def extract_python_code(text: str) -> str | None:
    fenced_blocks = re.findall(r"```([a-zA-Z0-9_-]*)\s*\n(.*?)```", text, re.DOTALL)
    candidates = []
    for language, block in fenced_blocks:
        code = block.strip()
        if not code:
            continue
        language = language.lower()
        if language in {"python", "py"} or _looks_like_python(code):
            candidates.append(code)

    if candidates:
        return max(candidates, key=len)

    stripped = text.strip()
    if stripped and _looks_like_python(stripped):
        return stripped

    return None


def _text_fingerprint(text: str) -> str:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return "text:" + hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def _text_change_summary(progress_score: float) -> str:
    if progress_score < 0.1:
        return "no textual change"
    if progress_score < 0.3:
        return "minor textual change"
    if progress_score < 0.6:
        return "moderate textual change"
    return "major textual change"


def _previous_text(previous_result: ExecutionResult | None) -> str | None:
    if previous_result is None:
        return None
    return previous_result.source_text or previous_result.output


def _previous_code(previous_result: ExecutionResult | None) -> str | None:
    if previous_result is None:
        return None
    if previous_result.extracted_code:
        return previous_result.extracted_code
    previous_text = _previous_text(previous_result)
    if previous_text:
        return extract_python_code(previous_text)
    return None


def _compute_intent(
    similarity: float | None,
    progress_score: float | None,
    change_vector: dict[str, int] | None,
) -> tuple[float | None, str | None]:
    if similarity is None or progress_score is None:
        return None, None

    vector = change_vector or ZERO_CHANGE_VECTOR
    intent_score = compute_intent_score(similarity, progress_score, vector)
    intent_class = classify_intent(intent_score, progress_score, similarity, vector)
    return intent_score, intent_class


def _build_python_ast_result(
    *,
    success: bool,
    output: str,
    error: str | None,
    source_text: str,
    code: str,
    previous_result: ExecutionResult | None,
) -> ExecutionResult:
    tokens = flatten_ast(code)
    features = extract_features(code)
    previous_code = _previous_code(previous_result)
    previous_text = _previous_text(previous_result)

    similarity = None
    progress_score = None
    change_type = None
    change_vector = None
    change_summary = None

    if previous_code:
        previous_tokens = flatten_ast(previous_code)
        previous_features = extract_features(previous_code)
        similarity = ast_similarity_score(previous_code, code)
        progress_score = compute_progress(previous_tokens, tokens)
        change_type = classify_change(progress_score)
        change_vector = compute_change_vector(previous_features, features)
        change_summary = summarize_change(change_vector)
    elif previous_text:
        similarity = difflib.SequenceMatcher(None, previous_text, source_text).ratio()
        progress_score = 1.0 - similarity
        change_type = classify_change(progress_score)
        change_vector = ZERO_CHANGE_VECTOR.copy()
        change_summary = _text_change_summary(progress_score)

    intent_score, intent_class = _compute_intent(
        similarity,
        progress_score,
        change_vector,
    )

    return ExecutionResult(
        success=success,
        output=output,
        error=error,
        ast_fingerprint=ast_fingerprint(code),
        ast_similarity_to_previous=similarity,
        ast_tokens=tokens,
        ast_progress_score=progress_score,
        ast_change_type=change_type,
        ast_features=features,
        ast_change_vector=change_vector,
        ast_change_summary=change_summary,
        intent_score=intent_score,
        intent_class=intent_class,
        source_text=source_text,
        extracted_code=code,
        signal_mode="python_ast",
        failure_type="LangChainError" if error else None,
    )


def _build_text_result(
    *,
    success: bool,
    output: str,
    error: str | None,
    source_text: str,
    previous_result: ExecutionResult | None,
) -> ExecutionResult:
    previous_text = _previous_text(previous_result)
    signal_mode = "text_fallback" if source_text.strip() else "none"
    similarity = None
    progress_score = None
    change_type = None
    change_summary = None

    if previous_text is not None:
        similarity = difflib.SequenceMatcher(None, previous_text, source_text).ratio()
        progress_score = 1.0 - similarity
        change_type = classify_change(progress_score)
        change_summary = _text_change_summary(progress_score)

    change_vector = ZERO_CHANGE_VECTOR.copy() if signal_mode == "text_fallback" else None
    intent_score, intent_class = _compute_intent(
        similarity,
        progress_score,
        change_vector,
    )

    return ExecutionResult(
        success=success,
        output=output,
        error=error,
        ast_fingerprint=_text_fingerprint(source_text) if signal_mode else None,
        ast_similarity_to_previous=similarity,
        ast_progress_score=progress_score,
        ast_change_type=change_type,
        ast_change_vector=change_vector,
        ast_change_summary=change_summary,
        intent_score=intent_score,
        intent_class=intent_class,
        source_text=source_text,
        extracted_code=None,
        signal_mode=signal_mode,
        failure_type="LangChainError" if error else None,
    )


def convert_to_execution_result(
    output,
    step_index: int,
    previous_result: ExecutionResult | None = None,
) -> ExecutionResult:
    error = _extract_error(output)
    text = _extract_text(output)
    success = error is None
    code = extract_python_code(text)

    if code:
        return _build_python_ast_result(
            success=success,
            output=text,
            error=error,
            source_text=text,
            code=code,
            previous_result=previous_result,
        )

    return _build_text_result(
        success=success,
        output=text,
        error=error,
        source_text=text,
        previous_result=previous_result,
    )
