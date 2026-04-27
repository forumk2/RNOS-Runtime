from typing import Dict, List

from .types import ExecutionResult


WINDOW_SIZE = 3


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _recent_with_current(
    result: ExecutionResult,
    history: List[ExecutionResult],
) -> list[ExecutionResult]:
    return (history + [result])[-WINDOW_SIZE:]


def _attempt_similarity(
    result: ExecutionResult,
    history: List[ExecutionResult],
) -> float:
    previous = history[-1] if history else None
    if not previous or result.success or previous.success:
        return 0.0
    return _clamp(result.ast_similarity_to_previous or 0.0)


def _progress_fluctuation(recent: list[ExecutionResult]) -> float:
    scores = [
        result.ast_progress_score
        for result in recent
        if result.ast_progress_score is not None
    ]
    if len(scores) < 2:
        return 0.0
    return _clamp(max(scores) - min(scores))


def _compute_consistency(
    result: ExecutionResult,
    recent: list[ExecutionResult],
    similarity: float,
) -> float:
    recent_failures = sum(1 for item in recent if not item.success)
    persistent_failure = not result.success and recent_failures >= 2
    stuck_penalty = similarity if persistent_failure else 0.0
    fluctuation_penalty = _progress_fluctuation(recent)
    return _clamp(1.0 - max(stuck_penalty, fluctuation_penalty))


def _classify_drift(
    overreach: float,
    echo_chamber: float,
    incoherence: float,
) -> str:
    if overreach > 0.6:
        return "overreach"
    if echo_chamber > 0.6:
        return "echo_chamber"
    if incoherence > 0.6:
        return "incoherent"
    return "stable"


def compute_cevak(
    result: ExecutionResult,
    history: List[ExecutionResult],
) -> Dict[str, float | str]:
    recent = _recent_with_current(result, history)
    progress_score = _clamp(result.ast_progress_score or 0.0)
    similarity = _attempt_similarity(result, history)

    consistency = _compute_consistency(result, recent, similarity)
    evidence = sum(1 for item in recent if item.success) / WINDOW_SIZE
    variance = _clamp(1.0 - similarity)
    agreement = similarity
    confidence = _clamp((0.5 * (1.0 - progress_score)) + (0.5 * similarity))

    overreach = confidence * (1.0 - evidence)
    echo_chamber = agreement * (1.0 - variance)
    incoherence = 1.0 - consistency
    drift_score = max(overreach, echo_chamber, incoherence)
    drift_type = _classify_drift(overreach, echo_chamber, incoherence)

    return {
        "consistency": consistency,
        "evidence": evidence,
        "variance": variance,
        "agreement": agreement,
        "confidence": confidence,
        "drift_score": drift_score,
        "drift_type": drift_type,
    }
