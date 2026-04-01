"""Parse planner output into structured actions."""

from __future__ import annotations

import json

from rnos.types import ActionRecord


def parse_action(raw_text: str, *, depth: int = 0, retry_count: int = 0) -> ActionRecord:
    """Parse a JSON tool request or fall back to a calculator action."""

    text = raw_text.strip()
    try:
        data = json.loads(text)
        tool_name = data["tool"]
        payload = data.get("payload", {})
        return ActionRecord(
            tool_name=tool_name,
            payload=payload,
            depth=depth,
            retry_count=retry_count,
            metadata={"raw_text": raw_text},
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return ActionRecord(
            tool_name="calculator",
            payload={"expression": text},
            depth=depth,
            retry_count=retry_count,
            metadata={"raw_text": raw_text, "parser_mode": "fallback"},
        )
