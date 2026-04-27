"""Readable CLI timeline formatting for RNOS events."""

from __future__ import annotations

from collections import Counter
from typing import Any


def format_timeline(events: list[dict[str, Any]]) -> str:
    lines = ["RNOS Event Timeline", "-------------------"]
    previous_drift = 0.0

    if not events:
        return "\n".join(lines + ["(no events)"])

    for event in events:
        step = int(event.get("step", 0))
        decision = str(event.get("decision", "ALLOW"))
        action = str(event.get("action", "unknown"))
        target = str(event.get("target", "") or "")
        entropy = float(event.get("entropy", 0.0))
        drift = float(event.get("drift_score", 0.0))
        risk = float(event.get("tool_risk", 0.0))
        reason = str(event.get("reason", "") or "healthy execution")

        drift_marker = " ^" if drift > previous_drift + 0.001 else ""
        entropy_marker = " !" if entropy >= 7.0 else ""
        target_text = f"({target})" if target else ""

        lines.append("")
        lines.append(f"[Step {step}] {decision:<7} {action}{target_text}")
        lines.append(
            f"         entropy={entropy:.2f}{entropy_marker} "
            f"drift={drift:.2f}{drift_marker} "
            f"risk={risk:.2f}"
        )
        lines.append(
            f"         failures={int(event.get('validation_failures', 0))} "
            f"files={int(event.get('files_modified', 0))} "
            f"lines={int(event.get('lines_changed', 0))}"
        )
        failure_type = str(event.get("failure_type", "") or "")
        improvement = event.get("improvement")
        if failure_type:
            lines.append(f"         failure_type={failure_type} improvement={improvement}")
        if decision in {"DEGRADE", "RECOVER", "REFUSE"} or reason != "healthy execution":
            lines.append(f"         reason: {reason}")

        previous_drift = drift

    lines.extend(["", format_decision_summary(events)])
    return "\n".join(lines)


def format_decision_summary(events: list[dict[str, Any]]) -> str:
    decision_events = [
        event
        for event in events
        if str(event.get("type", "")) not in {"recovery_event", "tuning_event", "outcome_event"}
    ]
    counts = Counter(str(event.get("decision", "ALLOW")) for event in decision_events)
    return (
        "RNOS Decision Summary\n"
        "---------------------\n"
        f"ALLOW: {counts.get('ALLOW', 0)}\n"
        f"DEGRADE: {counts.get('DEGRADE', 0)}\n"
        f"RECOVER: {counts.get('RECOVER', 0)}\n"
        f"REFUSE: {counts.get('REFUSE', 0)}"
    )
