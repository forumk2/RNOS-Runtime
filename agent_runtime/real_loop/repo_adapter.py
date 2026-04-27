"""Repository adapter for the real Agent Gate sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .file_tools import FileTools


@dataclass(frozen=True)
class RepoSnapshot:
    files: dict[str, bytes]


class RepoAdapter:
    """Owns setup, snapshots, modification tracking, and rollback."""

    def __init__(self, repo_path: str | Path = "sandbox_repo") -> None:
        self.root = Path(repo_path).resolve()
        self._assert_sandbox_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tools = FileTools(self.root)

    def reset(self, files: dict[str, str]) -> RepoSnapshot:
        self._clear()
        for path, content in files.items():
            self.tools.write_file(path, content)
        return self.snapshot()

    def snapshot(self) -> RepoSnapshot:
        files: dict[str, bytes] = {}
        if not self.root.exists():
            return RepoSnapshot(files=files)

        for path in sorted(self.root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                files[rel] = path.read_bytes()
        return RepoSnapshot(files=files)

    def modified_files(self, before: RepoSnapshot) -> tuple[str, ...]:
        after = self.snapshot().files
        changed = set(before.files) ^ set(after)
        for path, content in after.items():
            if before.files.get(path) != content:
                changed.add(path)
        return tuple(sorted(changed))

    def rollback(self, snapshot: RepoSnapshot) -> None:
        self._clear()
        for rel_path, content in snapshot.files.items():
            target = (self.root / rel_path).resolve()
            if self.root not in target.parents and target != self.root:
                raise ValueError(f"rollback path escapes sandbox_repo: {rel_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def _clear(self) -> None:
        self._assert_sandbox_root()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            return

        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _assert_sandbox_root(self) -> None:
        cwd = Path.cwd().resolve()
        if self.root.name != "sandbox_repo":
            raise ValueError("real loop repository must be named sandbox_repo")
        if self.root.parent != cwd:
            raise ValueError("real loop repository must be ./sandbox_repo under the current working directory")
