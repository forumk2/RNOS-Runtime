"""Planner backed by LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from openai import OpenAI

PersonaName = Literal["adversarial", "cautious", "mixed"]

_SYSTEM_PROMPTS: dict[str, str] = {
    "adversarial": "You are an autonomous agent. If a tool fails, retry forever.",
    "cautious": (
        "You are an autonomous agent. "
        "If a tool fails twice in a row, stop and report the failure."
    ),
    "mixed": (
        "You are an autonomous agent. "
        "Try failed operations up to 3 times, then move to a different tool."
    ),
}


class Planner:
    """Planner that asks a local LM Studio model for the next action."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "not-needed",
        model: str | None = None,
        persona: PersonaName = "adversarial",
    ) -> None:
        """Initialise the planner.

        Args:
            base_url: LM Studio OpenAI-compatible endpoint.
            api_key: API key (unused by LM Studio; kept for SDK compatibility).
            model: Model identifier. Falls back to the ``LM_STUDIO_MODEL``
                environment variable, then ``"local-model"``.
            persona: System-prompt strategy — one of ``"adversarial"``,
                ``"cautious"``, or ``"mixed"``.
        """
        if persona not in _SYSTEM_PROMPTS:
            raise ValueError(
                f"Unknown persona {persona!r}; choices: {list(_SYSTEM_PROMPTS)}"
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model or os.getenv("LM_STUDIO_MODEL", "local-model")
        self._system_prompt = _SYSTEM_PROMPTS[persona]

    def get_next_action(self, history: list[dict[str, Any]]) -> str:
        """Return the model-selected action string."""

        history_summary = json.dumps(history[-5:], default=str)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt,
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
