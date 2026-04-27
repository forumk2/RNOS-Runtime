from .types import ExecutionResult, RNOSDecision


WINDOW_SIZE = 3


def evaluate_state(history: list[ExecutionResult]) -> RNOSDecision:
    window = history[-WINDOW_SIZE:]
    if not window:
        return RNOSDecision(action="continue", reason="stable", entropy_score=0.0)

    failures = sum(1 for item in window if not item.success)
    entropy_score = failures / len(window)

    if failures >= 3:
        return RNOSDecision(
            action="refuse",
            reason="retry loop detected",
            entropy_score=entropy_score,
        )

    if not history[-1].success:
        return RNOSDecision(
            action="retry",
            reason="high entropy failure",
            entropy_score=entropy_score,
        )

    return RNOSDecision(
        action="continue",
        reason="stable",
        entropy_score=entropy_score,
    )
