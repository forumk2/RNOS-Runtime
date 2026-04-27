from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


class LivePublisher:
    def __init__(self, url: str = "http://127.0.0.1:8765/events", *, timeout: float = 0.2) -> None:
        self.url = url
        self.timeout = timeout
        self._disabled = False
        self._warned = False

    def publish(self, event: dict[str, Any]) -> None:
        if self._disabled:
            return

        payload = json.dumps(event).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            self._disabled = True
            if not self._warned:
                logger.warning("RNOS live publishing disabled: %s", exc)
                self._warned = True

