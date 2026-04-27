"""Tool-risk scoring for RNOS Agent Gate decisions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .agent import AgentPlan


TOOL_RISK = {
    "read": 1,
    "write": 3,
    "shell": 7,
    "network": 6,
    "delete": 9,
}

_DESTRUCTIVE_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b.*\b-recurse\b", re.IGNORECASE),
    re.compile(r"\bdel\b.*\s/[sq]\b", re.IGNORECASE),
    re.compile(r"\brmdir\b.*\s/[sq]\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    base_risk: float
    blast_radius: float
    escalation: bool
    destructive: bool
    reasons: tuple[str, ...]


def _base_risk(plan: AgentPlan) -> float:
    if plan.tool == "edit_file":
        return float(TOOL_RISK["write"])
    if plan.tool == "run_tests":
        return 2.0
    if plan.tool == "shell_command":
        return float(TOOL_RISK["shell"])
    if plan.tool == "delete":
        return float(TOOL_RISK["delete"])
    return float(TOOL_RISK.get(plan.tool, TOOL_RISK["read"]))


def is_destructive_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in _DESTRUCTIVE_PATTERNS)


def estimate_blast_radius(plan: AgentPlan) -> float:
    """Estimate how much of the workspace the planned action can affect."""

    haystack = f"{plan.target} {plan.command}".lower()
    if plan.tool == "delete":
        return 10.0
    if any(token in haystack for token in ("src/", "src\\", "agent_runtime", ".")):
        return 6.0
    if any(token in haystack for token in ("*", " --all", " -r", " -rf", "/s")):
        return 7.5
    if plan.target:
        return 3.0
    if plan.tool == "shell_command":
        return 5.0
    return 1.0


def assess_tool_risk(plan: AgentPlan, previous_scores: list[float]) -> RiskAssessment:
    """Return a normalized 0-10 risk score for a planned tool action."""

    base = _base_risk(plan)
    blast_radius = estimate_blast_radius(plan)
    destructive = is_destructive_command(plan.command) or plan.tool == "delete"
    previous_peak = max(previous_scores[-3:], default=0.0)
    escalation = bool(previous_scores and base >= 6.0 and base > previous_peak + 2.0)

    score = base + (blast_radius * 0.18)
    reasons: list[str] = [f"base={base:.1f}", f"blast_radius={blast_radius:.1f}"]

    if escalation:
        score += 0.9
        reasons.append("risk_escalation")
    if destructive:
        score += 3.0
        reasons.append("destructive_command")

    return RiskAssessment(
        score=round(max(0.0, min(10.0, score)), 3),
        base_risk=base,
        blast_radius=blast_radius,
        escalation=escalation,
        destructive=destructive,
        reasons=tuple(reasons),
    )
