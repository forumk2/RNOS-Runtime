"""Structured event stream for RNOS decision traces."""

from __future__ import annotations

import os
import time
from typing import Any

from .live.live_publisher import LivePublisher
from .live.session import LiveSession


class EventStream:
    """In-memory stream of ordered RNOS execution events."""

    def __init__(self, live: bool = False, session: LiveSession | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.live = live
        self.session = session if session is not None else (LiveSession() if live else None)
        self.publisher = LivePublisher() if live else None

    def emit(self, event: dict[str, Any]) -> None:
        event = self._with_default_metadata(event)
        if "timestamp" not in event:
            event = {**event, "timestamp": time.time()}
        if self.live and self.session is not None:
            event = self._with_live_metadata(event)
        self.events.append(event)
        if self.live and self.publisher is not None:
            self.publisher.publish(event)

    def get_events(self) -> list[dict[str, Any]]:
        return list(self.events)

    def _with_live_metadata(self, event: dict[str, Any]) -> dict[str, Any]:
        scenario = str(event.get("scenario", "unknown"))
        mode = str(event.get("mode", self.session.mode if self.session else "live"))
        run_id = str(event.get("run_id") or f"live:{self.session.session_id}:{scenario}:{mode}")
        return {
            **event,
            "session_id": self.session.session_id,
            "run_id": run_id,
            "stream_mode": "live",
            "source": self.session.source,
        }

    @staticmethod
    def _with_default_metadata(event: dict[str, Any]) -> dict[str, Any]:
        model = os.getenv("RNOS_EVENT_MODEL") or os.getenv("RNOS_LM_MODEL")
        if model and "model" not in event:
            event = {**event, "model": model}
        return event
