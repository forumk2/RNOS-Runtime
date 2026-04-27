"""Drift scoring for planned agent actions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .agent import AgentPlan, AgentState


@dataclass(frozen=True)
class DriftAssessment:
    score: float
    plan_similarity: float
    target_shift: float
    repeated_failed_edits: int
    reasons: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _target_shift(plan: AgentPlan, state: AgentState) -> float:
    if not plan.target or not state.targets:
        return 0.0
    if plan.target in state.targets[-2:]:
        return 0.0
    previous_roots = {target.replace("\\", "/").split("/")[0] for target in state.targets if target}
    root = plan.target.replace("\\", "/").split("/")[0]
    return 4.0 if root not in previous_roots else 2.0


def assess_drift(plan: AgentPlan, state: AgentState) -> DriftAssessment:
    """Compute a bounded 0-10 drift score from planner history."""

    reasons: list[str] = []
    similarity = 1.0
    if state.plan_texts:
        similarity = _similarity(state.plan_texts[-1], plan.text)

    score = max(0.0, (1.0 - similarity) * 3.5)
    if similarity < 0.25 and state.attempts > 0:
        reasons.append("plan_similarity_drop")

    target_shift = _target_shift(plan, state)
    if target_shift:
        score += target_shift
        reasons.append("file_target_shift")

    repeated_failed_edits = 0
    if plan.tool == "edit_file" and plan.target:
        repeated_failed_edits = state.failed_edit_targets.count(plan.target)
        if repeated_failed_edits:
            score += min(repeated_failed_edits * 1.7, 3.5)
            reasons.append("repeated_failed_edits")

    if "nonsense" in plan.description.lower() or "unrelated" in plan.description.lower():
        score += 2.0
        reasons.append("semantic_drift")

    return DriftAssessment(
        score=round(max(0.0, min(10.0, score)), 3),
        plan_similarity=round(similarity, 3),
        target_shift=target_shift,
        repeated_failed_edits=repeated_failed_edits,
        reasons=tuple(reasons or ["stable"]),
    )
