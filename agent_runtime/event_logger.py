"""JSON persistence for RNOS event streams."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def save_json(events: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    """Save events as pretty-printed JSON and return the written path."""

    target = Path(path) if path is not None else _default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _default_path() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    return Path("rnos_logs") / f"run_{stamp}.json"
