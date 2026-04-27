import logging
import re

from .types import ExecutionResult, RNOSDecision


WINDOW_SIZE = 3
logger = logging.getLogger(__name__)


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _recent(history: list[ExecutionResult]) -> list[ExecutionResult]:
    return history[-WINDOW_SIZE:]


def _failures(history: list[ExecutionResult]) -> list[ExecutionResult]:
    return [result for result in _recent(history) if not result.success]


def _normalized_error(error: str | None) -> str:
    if not error:
        return "Unknown"

    normalized = error.strip()
    normalized = re.sub(r"[A-Za-z]:\\[^\s)]+", "<path>", normalized)
    normalized = re.sub(r"\bstep_\d+\.py\b", "step.py", normalized)
    normalized = re.sub(r"\bline \d+\b", "line N", normalized)
    return normalized


def extract_error_type(error: str | None) -> str:
    if not error:
        return "Unknown"

    lower_error = error.lower()
    if "syntaxerror" in lower_error or "invalid syntax" in lower_error:
        return "SyntaxError"
    if "was never closed" in lower_error or "unexpected eof" in lower_error:
        return "SyntaxError"
    if "runtimeerror" in lower_error:
        return "RuntimeError"
    if "filenotfound" in lower_error or "missingfile" in lower_error:
        return "MissingFile"
    if "no such file" in lower_error or "not found" in lower_error:
        return "MissingFile"

    return "Unknown"


def compute_failure_rate(history: list[ExecutionResult]) -> float:
    window = _recent(history)
    if not window:
        return 0.0
    return sum(1 for result in window if not result.success) / WINDOW_SIZE


def compute_diversity(history: list[ExecutionResult]) -> float:
    failures = _failures(history)
    if not failures:
        return 0.0

    error_types = {
        result.failure_type or extract_error_type(result.error) for result in failures
    }
    return len(error_types) / len(failures)


def compute_error_similarity(history: list[ExecutionResult]) -> float:
    failures = _failures(history)
    if not failures:
        return 0.0

    recent_errors = [_normalized_error(result.error) for result in failures[-3:]]
    repeated_count = sum(
        1 for error in recent_errors if recent_errors.count(error) > 1
    )
    return repeated_count / len(failures)


def compute_ast_retry_similarity(history: list[ExecutionResult]) -> float:
    window = _recent(history)
    scores = []
    for index, result in enumerate(window):
        if index == 0:
            continue
        previous = window[index - 1]
        if result.success or previous.success:
            continue
        if result.ast_similarity_to_previous is not None:
            scores.append(result.ast_similarity_to_previous)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def compute_similarity(history: list[ExecutionResult]) -> float:
    return max(
        compute_error_similarity(history),
        compute_ast_retry_similarity(history),
    )


def compute_progress_score(history: list[ExecutionResult]) -> float:
    for result in reversed(_recent(history)):
        if result.ast_progress_score is not None:
            return result.ast_progress_score
    return 0.0


def latest_change_type(history: list[ExecutionResult]) -> str:
    for result in reversed(_recent(history)):
        if result.ast_change_type:
            return result.ast_change_type
    return "unknown"


def latest_change_vector(history: list[ExecutionResult]) -> dict[str, int]:
    for result in reversed(_recent(history)):
        if result.ast_change_vector is not None:
            return result.ast_change_vector
    return {}


def latest_change_summary(history: list[ExecutionResult]) -> str:
    for result in reversed(_recent(history)):
        if result.ast_change_summary:
            return result.ast_change_summary
    return "unknown"


def _log_change_vector(change_vector: dict[str, int]) -> None:
    keys = ("if", "for", "while", "try", "function_def", "call", "assign", "return")
    logger.info(
        "[RNOS CHANGE VECTOR]\n%s",
        "\n".join(f"{key}={change_vector.get(key, 0):+d}" for key in keys),
    )


def compute_trend(history: list[ExecutionResult]) -> float:
    window = _recent(history)
    if len(window) < 2:
        return 0.0

    half_size = len(window) // 2
    first_half = window[:half_size]
    last_half = window[-half_size:]

    first_failures = sum(1 for result in first_half if not result.success) / len(first_half)
    last_failures = sum(1 for result in last_half if not result.success) / len(last_half)
    return _clamp(last_failures - first_failures)


def compute_instability(history: list[ExecutionResult]) -> float:
    failure_rate = compute_failure_rate(history)
    diversity_score = compute_diversity(history)
    similarity_score = compute_similarity(history)
    trend_score = compute_trend(history)

    return _clamp(
        (0.35 * failure_rate)
        + (0.20 * diversity_score)
        + (0.30 * similarity_score)
        + (0.15 * trend_score)
    )


def evaluate_state(history: list[ExecutionResult]) -> RNOSDecision:
    failure_rate = compute_failure_rate(history)
    diversity_score = compute_diversity(history)
    error_similarity = compute_error_similarity(history)
    ast_similarity = compute_ast_retry_similarity(history)
    similarity_score = compute_similarity(history)
    progress_score = compute_progress_score(history)
    change_type = latest_change_type(history)
    change_vector = latest_change_vector(history)
    change_summary = latest_change_summary(history)
    trend_score = compute_trend(history)
    instability_score = compute_instability(history)
    recent_failures = len(_failures(history))
    latest_failure = bool(history and not history[-1].success)

    logger.info(
        "[RNOS METRICS]\n"
        "failure_rate=%.2f\n"
        "diversity=%.2f\n"
        "error_similarity=%.2f\n"
        "ast_similarity=%.2f\n"
        "ast_progress=%.2f\n"
        "change_type=%s\n"
        "similarity=%.2f\n"
        "trend=%.2f\n"
        "instability=%.2f",
        failure_rate,
        diversity_score,
        error_similarity,
        ast_similarity,
        progress_score,
        change_type,
        similarity_score,
        trend_score,
        instability_score,
    )
    _log_change_vector(change_vector)
    logger.info("[RNOS CHANGE SUMMARY]\n%s", change_summary)

    if (
        latest_failure
        and similarity_score > 0.85
        and progress_score < 0.1
        and change_summary in {"no structural change", "unknown"}
        and recent_failures >= 2
    ):
        action = "refuse"
        reason = "no structural progress"
    elif (
        latest_failure
        and similarity_score > 0.75
        and progress_score < 0.2
        and change_summary == "minor logic change"
        and recent_failures >= 2
    ):
        action = "refuse"
        reason = "cosmetic retry loop"
    elif latest_failure and change_summary in {
        "control flow change",
        "function structure change",
        "major structural change",
    }:
        action = "retry"
        reason = "meaningful structural attempt"
    elif instability_score > 0.75:
        action = "refuse"
        reason = "system unstable (high entropy)"
    elif latest_failure and progress_score > 0.4:
        action = "retry"
        reason = "meaningful attempt, allow exploration"
    elif error_similarity > 0.6 and recent_failures >= 2:
        action = "refuse"
        reason = "stuck retry loop"
    elif latest_failure and instability_score >= 0.4:
        action = "retry"
        reason = "degrading system"
    else:
        action = "continue"
        reason = "stable"

    logger.info("[RNOS DECISION]\naction=%s\nreason=%s", action, reason)

    return RNOSDecision(
        action=action,
        reason=reason,
        entropy_score=instability_score,
    )
