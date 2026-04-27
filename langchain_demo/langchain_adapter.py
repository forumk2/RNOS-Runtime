import hashlib

from agent_runtime.types import ExecutionResult


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


def convert_to_execution_result(output, step_index: int) -> ExecutionResult:
    error = _extract_error(output)
    text = _extract_text(output)
    success = error is None
    fingerprint_source = "langchain-agent-loop"

    return ExecutionResult(
        success=success,
        output=text,
        error=error,
        ast_fingerprint=hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
        ast_similarity_to_previous=1.0 if step_index > 0 else None,
        ast_progress_score=0.0 if step_index > 0 else None,
        ast_change_type="no_change" if step_index > 0 else None,
        ast_change_vector={
            "if": 0,
            "for": 0,
            "while": 0,
            "try": 0,
            "function_def": 0,
            "call": 0,
            "assign": 0,
            "return": 0,
        }
        if step_index > 0
        else None,
        ast_change_summary="no structural change" if step_index > 0 else None,
        intent_score=0.0 if step_index > 0 else None,
        intent_class="no_intent" if step_index > 0 else None,
        failure_type="LangChainError" if error else None,
    )
