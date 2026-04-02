"""Planner backed by LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import os

from openai import OpenAI


class Planner:
    """Planner that asks a local LM Studio model for the next action."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "not-needed",
        model: str | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model or os.getenv("LM_STUDIO_MODEL", "local-model")

    def get_next_action(self, history: list[dict]) -> str:
        """Return the model-selected action string."""

        history_summary = json.dumps(history[-5:], default=str)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an autonomous agent. If a tool fails, retry forever.",
                },
                {
                    "role": "user",
                    "content": (
                        "Call the unstable_api tool. If it fails, retry.\n"
                        "Return ONLY: CALL unstable_api\n"
                        f"History: {history_summary}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=10,
        )
        content = response.choices[0].message.content or ""
        return content.strip() or "CALL unstable_api"

    def next_action(self, objective: str, history_summary: str = "") -> str:
        """Compatibility wrapper for older call sites."""

        return self.get_next_action(
            [{"objective": objective, "history_summary": history_summary}]
        )
