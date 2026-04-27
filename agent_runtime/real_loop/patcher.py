"""Unified-diff patch analysis and application for sandbox repositories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


class PatchError(ValueError):
    """Raised when a patch is invalid or unsafe."""


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FilePatch:
    old_path: str
    new_path: str
    hunks: tuple[Hunk, ...]


@dataclass(frozen=True)
class PatchReport:
    files_modified: tuple[str, ...] = field(default_factory=tuple)
    lines_added: int = 0
    lines_removed: int = 0
    lines_changed: int = 0
    large_edit: bool = False
    multi_file_edit: bool = False
    risky_edit: bool = False
    destructive: bool = False


_HUNK_RE = re.compile(r"@@ -(?P<old>\d+)(?:,(?P<old_count>\d+))? \+(?P<new>\d+)(?:,(?P<new_count>\d+))? @@")


def analyze_patch(repo_root: Path, diff: str) -> PatchReport:
    parsed = _parse_unified_diff(diff)
    return _report_for(repo_root, parsed)


def apply_unified_diff(repo_root: Path, diff: str, *, dry_run: bool = False) -> PatchReport:
    parsed = _parse_unified_diff(diff)
    report = _report_for(repo_root, parsed)
    if dry_run:
        return report

    for file_patch in parsed:
        _apply_file_patch(repo_root, file_patch)

    return report


def _parse_unified_diff(diff: str) -> list[FilePatch]:
    lines = diff.splitlines(keepends=True)
    patches: list[FilePatch] = []
    index = 0

    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue

        old_path = _clean_header_path(lines[index][4:].strip())
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise PatchError("unified diff missing +++ header")

        new_path = _clean_header_path(lines[index][4:].strip())
        index += 1
        hunks: list[Hunk] = []

        while index < len(lines):
            if lines[index].startswith("--- "):
                break
            if not lines[index].startswith("@@"):
                index += 1
                continue

            match = _HUNK_RE.match(lines[index])
            if not match:
                raise PatchError(f"invalid hunk header: {lines[index].strip()}")

            index += 1
            hunk_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                if lines[index].startswith("\\ No newline"):
                    index += 1
                    continue
                if not lines[index].startswith((" ", "+", "-")):
                    raise PatchError(f"invalid hunk line: {lines[index].strip()}")
                hunk_lines.append(lines[index])
                index += 1

            hunks.append(
                Hunk(
                    old_start=int(match.group("old")),
                    old_count=int(match.group("old_count") or "1"),
                    new_start=int(match.group("new")),
                    new_count=int(match.group("new_count") or "1"),
                    lines=tuple(hunk_lines),
                )
            )

        patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=tuple(hunks)))

    if not patches:
        raise PatchError("no file patches found")
    return patches


def _apply_file_patch(repo_root: Path, file_patch: FilePatch) -> None:
    if file_patch.new_path == "/dev/null":
        target = _resolve_sandbox_path(repo_root, file_patch.old_path)
        if target.exists():
            target.unlink()
        return

    target = _resolve_sandbox_path(repo_root, file_patch.new_path)
    original = []
    if target.exists():
        original = target.read_text(encoding="utf-8").splitlines(keepends=True)

    patched: list[str] = []
    source_index = 0

    for hunk in file_patch.hunks:
        preferred_start = max(hunk.old_start - 1, 0)
        hunk_start = _resolve_hunk_start(original, hunk, preferred_start, source_index, target)
        patched.extend(original[source_index:hunk_start])
        source_index = hunk_start

        for line in hunk.lines:
            prefix = line[0]
            content = line[1:]
            if prefix == " ":
                _assert_context(original, source_index, content, target)
                patched.append(original[source_index])
                source_index += 1
            elif prefix == "-":
                _assert_context(original, source_index, content, target)
                source_index += 1
            elif prefix == "+":
                patched.append(content)

    patched.extend(original[source_index:])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(patched), encoding="utf-8")


def _resolve_hunk_start(
    original: list[str],
    hunk: Hunk,
    preferred_start: int,
    minimum_start: int,
    target: Path,
) -> int:
    if _hunk_matches(original, preferred_start, hunk):
        return preferred_start

    for candidate in range(minimum_start, len(original) + 1):
        if candidate == preferred_start:
            continue
        if _hunk_matches(original, candidate, hunk):
            return candidate

    raise PatchError(f"patch context mismatch for {target.name}")


def _hunk_matches(original: list[str], start: int, hunk: Hunk) -> bool:
    index = start
    for line in hunk.lines:
        if line.startswith("+"):
            continue
        if index >= len(original) or original[index] != line[1:]:
            return False
        index += 1
    return True


def _report_for(repo_root: Path, patches: list[FilePatch]) -> PatchReport:
    files: list[str] = []
    added = 0
    removed = 0
    destructive = False

    for file_patch in patches:
        display_path = file_patch.old_path if file_patch.new_path == "/dev/null" else file_patch.new_path
        target = _resolve_sandbox_path(repo_root, display_path)
        _reject_binary(target)
        files.append(display_path)

        if file_patch.new_path == "/dev/null":
            destructive = True

        for hunk in file_patch.hunks:
            for line in hunk.lines:
                if line.startswith("+"):
                    added += 1
                elif line.startswith("-"):
                    removed += 1

    changed = added + removed
    large_edit = changed > 30
    multi_file = len(set(files)) > 1
    destructive = destructive or removed > 40
    risky = large_edit or multi_file or destructive
    return PatchReport(
        files_modified=tuple(sorted(set(files))),
        lines_added=added,
        lines_removed=removed,
        lines_changed=changed,
        large_edit=large_edit,
        multi_file_edit=multi_file,
        risky_edit=risky,
        destructive=destructive,
    )


def _assert_context(original: list[str], index: int, expected: str, target: Path) -> None:
    if index >= len(original) or original[index] != expected:
        raise PatchError(f"patch context mismatch for {target.name}")


def _clean_header_path(raw: str) -> str:
    path = raw.split("\t", 1)[0].strip()
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _resolve_sandbox_path(repo_root: Path, user_path: str) -> Path:
    if user_path == "/dev/null":
        raise PatchError("/dev/null is not a writable sandbox path")
    root = repo_root.resolve()
    candidate = (root / user_path).resolve()
    if root.name != "sandbox_repo":
        raise PatchError("real loop writes are restricted to ./sandbox_repo")
    if candidate != root and root not in candidate.parents:
        raise PatchError(f"path escapes sandbox_repo: {user_path}")
    return candidate


def _reject_binary(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if b"\x00" in data:
        raise PatchError(f"binary file rejected: {path.name}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"non-utf8 file rejected: {path.name}") from exc
