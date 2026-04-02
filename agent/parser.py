"""Parse planner output into structured actions."""

from __future__ import annotations

import json
import re

from rnos.types import ActionRecord

# Matches: CALL <tool_name> [optional JSON payload]
_CALL_RE = re.compile(r"CALL\s+(\w+)(.*)", re.DOTALL)


def parse_action(output: str) -> ActionRecord:
    """Parse a planner completion into an action record.

    Handles the following forms:
    - ``CALL <tool_name>`` — bare tool invocation
    - ``CALL <tool_name> <json_object>`` — tool invocation with a JSON payload
    - Any other text — falls through to ``tool_name="unknown"``

    The optional JSON payload is merged into :attr:`ActionRecord.payload`.
    Malformed JSON is silently dropped and recorded in ``metadata``.
    """

    m = _CALL_RE.search(output)
    if m is None:
        return ActionRecord(
            tool_name="unknown",
            metadata={"raw_text": output.strip(), "parser_mode": "unknown"},
        )

    tool_name = m.group(1)
    payload_text = m.group(2).strip()

    payload: dict[str, object] = {}
    metadata: dict[str, object] = {"raw_text": output.strip()}

    if payload_text:
        try:
            parsed = json.loads(payload_text)
            if isinstance(parsed, dict):
                payload = parsed
            else:
                metadata["payload_parse_error"] = "JSON value was not an object"
        except json.JSONDecodeError as exc:
            metadata["payload_parse_error"] = str(exc)

    return ActionRecord(tool_name=tool_name, payload=payload, metadata=metadata)
