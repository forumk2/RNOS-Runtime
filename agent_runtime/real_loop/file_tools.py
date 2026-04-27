"""Filesystem tools constrained to ./sandbox_repo."""

from __future__ import annotations

from pathlib import Path

from .patcher import PatchError, PatchReport, apply_unified_diff


class SandboxViolation(ValueError):
    """Raised when a file operation would leave the sandbox."""


class FileTools:
    """Read, write, and patch text files inside a sandbox repository."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        if self.repo_root.name != "sandbox_repo":
            raise SandboxViolation("real loop file tools require ./sandbox_repo")

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        data = target.read_bytes()
        self._reject_binary(data, path)
        return data.decode("utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        if target.exists():
            self._reject_binary(target.read_bytes(), path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def apply_patch(self, diff: str, *, dry_run: bool = False) -> PatchReport:
        try:
            return apply_unified_diff(self.repo_root, diff, dry_run=dry_run)
        except PatchError as exc:
            raise SandboxViolation(str(exc)) from exc

    def _resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise SandboxViolation(f"absolute path rejected: {path}")
        target = (self.repo_root / path).resolve()
        if target != self.repo_root and self.repo_root not in target.parents:
            raise SandboxViolation(f"path traversal rejected: {path}")
        return target

    @staticmethod
    def _reject_binary(data: bytes, path: str) -> None:
        if b"\x00" in data:
            raise SandboxViolation(f"binary file rejected: {path}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SandboxViolation(f"non-utf8 file rejected: {path}") from exc
