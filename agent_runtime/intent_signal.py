from typing import Dict


STRUCTURAL_KEYS = ("if", "for", "while", "try", "function_def")


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def compute_structural_score(change_vector: Dict[str, int] | None) -> float:
    if not change_vector:
        return 0.0

    num_structural_changes = sum(
        abs(change_vector.get(key, 0)) for key in STRUCTURAL_KEYS
    )
    return min(1.0, num_structural_changes / 2)


def compute_intent_score(
    similarity: float,
    progress_score: float,
    change_vector: Dict[str, int] | None,
) -> float:
    structural_score = compute_structural_score(change_vector)
    return _clamp(
        (0.4 * progress_score)
        + (0.3 * structural_score)
        + (0.3 * (1 - similarity))
    )


def classify_intent(
    intent_score: float,
    progress_score: float,
    similarity: float,
    change_vector: Dict[str, int] | None,
) -> str:
    structural_score = compute_structural_score(change_vector)

    if progress_score < 0.1 and similarity > 0.85:
        return "no_intent"
    if progress_score < 0.2 and structural_score == 0:
        return "cosmetic_intent"
    if progress_score > 0.4 and structural_score > 0:
        return "exploratory_intent"
    if progress_score > 0.6:
        return "strong_intent"
    return "weak_intent"
