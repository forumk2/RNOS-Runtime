"""Parse planner output into structured actions."""

from __future__ import annotations

from rnos.types import ActionRecord


def parse_action(output: str) -> ActionRecord:
    """Parse a planner completion into an action record."""

    if "CALL unstable_api" in output:
        return ActionRecord(
            tool_name="unstable_api",
            metadata={"raw_text": output.strip()},
        )

    return ActionRecord(
        tool_name="unknown",
        metadata={"raw_text": output.strip(), "parser_mode": "unknown"},
    )
