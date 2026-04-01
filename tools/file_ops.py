"""Basic file operations constrained to the current workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ToolResult


class FileOpsTool:
    name = "file_ops"

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def run(self, operation: str, path: str, content: str = "", **_: Any) -> ToolResult:
        target = (self.root / path).resolve()
        if self.root not in target.parents and target != self.root:
            return ToolResult(ok=False, message="Refused path outside workspace", data={"path": str(target)})

        if operation == "read":
            if not target.exists():
                return ToolResult(ok=False, message="File not found", data={"path": str(target)})
            return ToolResult(ok=True, message="Read succeeded", data={"content": target.read_text(encoding="utf-8")})

        if operation == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(ok=True, message="Write succeeded", data={"path": str(target)})

        return ToolResult(ok=False, message=f"Unknown operation: {operation}")
