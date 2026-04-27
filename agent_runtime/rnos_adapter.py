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

    error_types = {extract_error_type(result.error) for result in failures}
    return len(error_types) / len(failures)


def compute_similarity(history: list[ExecutionResult]) -> float:
    failures = _failures(history)
    if not failures:
        return 0.0

    recent_errors = [_normalized_error(result.error) for result in failures[-3:]]
    repeated_count = sum(
        1 for error in recent_errors if recent_errors.count(error) > 1
    )
    return repeated_count / len(failures)


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
    similarity_score = compute_similarity(history)
    trend_score = compute_trend(history)
    instability_score = compute_instability(history)

    logger.info(
        "[RNOS METRICS]\n"
        "failure_rate=%.2f\n"
        "diversity=%.2f\n"
        "similarity=%.2f\n"
        "trend=%.2f\n"
        "instability=%.2f",
        failure_rate,
        diversity_score,
        similarity_score,
        trend_score,
        instability_score,
    )

    if instability_score >= 0.75:
        action = "refuse"
        reason = "system unstable (high entropy)"
    elif similarity_score > 0.6:
        action = "refuse"
        reason = "stuck retry loop"
    elif instability_score >= 0.4:
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
