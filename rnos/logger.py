"""Logging utilities for RNOS traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str = "rnos") -> logging.Logger:
    """Return a console logger with a predictable format."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def write_trace(event: dict[str, Any], path: str | Path = "logs/rnos_trace.jsonl") -> None:
    """Append a JSON line trace event."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
