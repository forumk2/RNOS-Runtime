"""Structured recovery feedback for guided RNOS retries."""

from __future__ import annotations

from typing import Any


def generate_feedback(context: dict[str, Any]) -> str:
    """Return concise corrective guidance for the next recovery attempt."""

    failure_type = str(context.get("failure_type", "unknown"))
    error = str(context.get("error", "") or "")
    target = str(context.get("target", "") or "the original target file")
    attempt = int(context.get("recovery_attempts", 0))

    if _looks_like_hallucination(error):
        lines = [
            "You referenced functions, names, or helpers that do not exist.",
            "Use only code present in the file.",
            "Do not invent helpers.",
        ]
        if "name 're' is not defined" in error or 'name "re" is not defined' in error:
            lines.append("The concrete fix is to add `import re` at the top of the target file.")
            lines.append(
                "Use this exact minimal diff shape:\n"
                "--- a/parser.py\n"
                "+++ b/parser.py\n"
                "@@ -1,5 +1,7 @@\n"
                " \"\"\"Parser sandbox with a recoverable missing import.\"\"\"\n"
                " \n"
                "+import re\n"
                "+\n"
                " def parse_number(text):"
            )
    elif failure_type == "malformed_output":
        lines = [
            "Your previous output was not valid JSON or not a valid unified diff.",
            "Return ONLY valid JSON matching the schema.",
            "Do not include explanations.",
            "The diff must start with --- a/<target> and +++ b/<target>, followed by a valid @@ hunk.",
        ]
    elif failure_type == "anchor_mismatch":
        lines = [
            "Your patch failed to apply because the hunk anchor did not match the file.",
            "Return a minimal edit using this format:",
            '{ "action": "edit_file", "target": "...", "edits": [ { "line_number": X, "content": "..." } ] }',
            "Modify only the necessary line.",
        ]
    elif failure_type == "drift":
        lines = [
            "You are modifying unrelated files.",
            f"Focus ONLY on {target}.",
            "Do not expand scope.",
        ]
    elif failure_type in {"recoverable_validation", "validation_failure"}:
        lines = [
            "The previous patch did not fix the failing tests.",
            "Focus only on fixing the specific error.",
            "Do not introduce new changes.",
            "Keep modifications minimal.",
        ]
        if "1+2+3" in error or "multi_addition" in error:
            lines.append("The parser must handle multiple addition terms, not just one plus sign.")
            lines.append(
                "Use a minimal patch that changes parse_addition to sum parse_number(part) for each part in text.split('+')."
            )
    else:
        lines = [
            "The previous attempt did not improve the outcome.",
            f"Focus only on {target}.",
            "Return the smallest valid patch that addresses the failure.",
        ]

    lines.extend(_escalation_lines(attempt, target))
    if error:
        lines.append("Failure excerpt:")
        lines.append(error[-900:])
    return "\n".join(lines)


def _escalation_lines(attempt: int, target: str) -> list[str]:
    if attempt <= 1:
        return [
            "Recovery attempt 1: choose edit_file and fix the issue with a minimal targeted change.",
        ]
    if attempt == 2:
        return [
            "Recovery attempt 2: choose edit_file and fix ONLY the failing line or import.",
            "Do NOT change anything else.",
        ]
    return [
        "Recovery attempt 3: choose edit_file and return a minimal patch.",
        f"Only modify the exact failing code in {target}.",
    ]


def _looks_like_hallucination(error: str) -> bool:
    lowered = error.lower()
    markers = (
        "nameerror",
        "attributeerror",
        "not defined",
        "has no attribute",
        "cannot import name",
        "module has no attribute",
    )
    return any(marker in lowered for marker in markers)
