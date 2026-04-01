"""Very small planner shim for LM Studio or local scripted behavior."""

from __future__ import annotations

import json
import os
from urllib import error, request


class Planner:
    """Planner that can call LM Studio but degrades to deterministic rules."""

    def __init__(self, endpoint: str | None = None, model: str | None = None) -> None:
        self.endpoint = endpoint or os.getenv(
            "LM_STUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions"
        )
        self.model = model or os.getenv("LM_STUDIO_MODEL", "local-model")

    def next_action(self, objective: str, history_summary: str = "") -> str:
        """Return the next action request as JSON text."""

        if os.getenv("RNOS_USE_LM_STUDIO", "").lower() not in {"1", "true", "yes"}:
            return self._scripted_plan(objective)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON with keys tool and payload.",
                },
                {
                    "role": "user",
                    "content": f"Objective: {objective}\nHistory: {history_summary}",
                },
            ],
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (OSError, error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return self._scripted_plan(objective)

    def _scripted_plan(self, objective: str) -> str:
        objective_lower = objective.lower()
        if any(token in objective_lower for token in ("add", "sum", "multiply", "calculate")):
            return json.dumps({"tool": "calculator", "payload": {"expression": "2 + 2"}})
        if any(token in objective_lower for token in ("file", "read", "write")):
            return json.dumps({"tool": "file_ops", "payload": {"operation": "read", "path": "README.md"}})
        if "unstable" in objective_lower or "api" in objective_lower:
            return json.dumps({"tool": "unstable_api", "payload": {"resource": "/health"}})
        return json.dumps({"tool": "calculator", "payload": {"expression": "21 * 2"}})
