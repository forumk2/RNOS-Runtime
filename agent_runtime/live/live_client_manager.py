from __future__ import annotations

import json
from typing import Any


class LiveClientManager:
    def __init__(self) -> None:
        self.clients: set[Any] = set()

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: Any) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        disconnected: list[Any] = []
        message = json.dumps(event)

        for websocket in list(self.clients):
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)

