from dataclasses import dataclass
from typing import List, Literal, Optional


DecisionAction = Literal["continue", "retry", "refuse"]


@dataclass(frozen=True)
class Task:
    prompt: str


@dataclass(frozen=True)
class Plan:
    steps: List[str]


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    ast_fingerprint: Optional[str] = None
    ast_similarity_to_previous: Optional[float] = None
    ast_tokens: Optional[List[str]] = None
    ast_progress_score: Optional[float] = None
    ast_change_type: Optional[str] = None
    failure_type: Optional[str] = None


@dataclass(frozen=True)
class RNOSDecision:
    action: DecisionAction
    reason: str
    entropy_score: float
