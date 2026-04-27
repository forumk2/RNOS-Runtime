"""Structured event stream for RNOS decision traces."""

from __future__ import annotations

import time
from typing import Any


class EventStream:
    """In-memory stream of ordered RNOS execution events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        if "timestamp" not in event:
            event = {**event, "timestamp": time.time()}
        self.events.append(event)

    def get_events(self) -> list[dict[str, Any]]:
        return list(self.events)
