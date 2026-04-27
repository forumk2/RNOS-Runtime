from __future__ import annotations

import argparse
from collections import deque
from typing import Any

from .live_client_manager import LiveClientManager


try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn
except ImportError as exc:  # pragma: no cover - exercised only when optional deps are absent.
    FastAPI = None  # type: ignore[assignment]
    WebSocket = None  # type: ignore[assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


manager = LiveClientManager()
recent_events: deque[dict[str, Any]] = deque(maxlen=1000)


if FastAPI is not None:
    app = FastAPI(title="RNOS Runtime Live Events")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/events")
    async def ingest_event(event: dict[str, Any]) -> dict[str, bool]:
        recent_events.append(event)
        await manager.broadcast({"type": "event", "event": event})
        return {"ok": True}

    @app.websocket("/ws/events")
    async def events_socket(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            await websocket.send_json({"type": "snapshot", "events": list(recent_events)})
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)
else:
    app = None


def main() -> int:
    if _IMPORT_ERROR is not None or uvicorn is None:
        raise SystemExit(
            "RNOS live server requires optional dependencies. "
            "Install with: pip install fastapi uvicorn"
        )

    parser = argparse.ArgumentParser(description="Run the local RNOS live event server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    uvicorn.run("agent_runtime.live.live_server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
