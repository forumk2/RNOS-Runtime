from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    path: str
    start_line: int
    end_line: int
    commit_sha: str | None
    content: str
    chunk_type: str


def chunk_file(
    source: str, path: str, suffix: str, commit_sha: str | None
) -> list[Chunk]:
    if suffix == ".py":
        return _chunk_python(source, path, commit_sha)
    if suffix == ".md":
        return _chunk_markdown(source, path, commit_sha)
    return _chunk_text(source, path, commit_sha)


# ---------------------------------------------------------------------------
# Python: tree-sitter, one chunk per top-level def/class
# ---------------------------------------------------------------------------

def _chunk_python(source: str, path: str, commit_sha: str | None) -> list[Chunk]:
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python as tspython

        lang = Language(tspython.language())
        parser = Parser(lang)
    except Exception:
        return _chunk_text(source, path, commit_sha)

    tree = parser.parse(source.encode())
    root = tree.root_node
    chunks: list[Chunk] = []

    # Module docstring
    if (
        root.child_count > 0
        and root.children[0].type == "expression_statement"
        and root.children[0].child_count > 0
        and root.children[0].children[0].type == "string"
    ):
        node = root.children[0]
        chunks.append(
            Chunk(
                path=path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                commit_sha=commit_sha,
                content=source[node.start_byte : node.end_byte],
                chunk_type="module_docstring",
            )
        )

    for node in root.children:
        if node.type in (
            "function_definition",
            "class_definition",
            "decorated_definition",
        ):
            chunks.append(
                Chunk(
                    path=path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    commit_sha=commit_sha,
                    content=source[node.start_byte : node.end_byte],
                    chunk_type=node.type,
                )
            )

    return chunks or _chunk_text(source, path, commit_sha)


# ---------------------------------------------------------------------------
# Markdown: split on ## boundaries, carry H1 context
# ---------------------------------------------------------------------------

def _chunk_markdown(source: str, path: str, commit_sha: str | None) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    h1 = ""
    section_h2 = ""
    section_start = 1
    section_lines: list[str] = []

    def _flush(end_line: int) -> None:
        content = "\n".join(section_lines).strip()
        if not content:
            return
        prefix = (f"# {h1}\n" if h1 else "") + (
            f"## {section_h2}\n\n" if section_h2 else ""
        )
        chunks.append(
            Chunk(
                path=path,
                start_line=section_start,
                end_line=end_line,
                commit_sha=commit_sha,
                content=prefix + content,
                chunk_type="md_section",
            )
        )

    for i, line in enumerate(lines, 1):
        if line.startswith("# ") and not line.startswith("## "):
            h1 = line[2:].strip()
            section_lines.append(line)
        elif line.startswith("## "):
            _flush(i - 1)
            section_h2 = line[3:].strip()
            section_start = i
            section_lines = [line]
        else:
            section_lines.append(line)

    _flush(len(lines))

    if not chunks:
        return [
            Chunk(
                path=path,
                start_line=1,
                end_line=max(len(lines), 1),
                commit_sha=commit_sha,
                content=source,
                chunk_type="md_full",
            )
        ]
    return chunks


# ---------------------------------------------------------------------------
# Text/JSON/log: whole file if small, else split on blank lines
# ---------------------------------------------------------------------------

def _chunk_text(source: str, path: str, commit_sha: str | None) -> list[Chunk]:
    if len(source) < 1500:
        lines = source.splitlines()
        return [
            Chunk(
                path=path,
                start_line=1,
                end_line=max(len(lines), 1),
                commit_sha=commit_sha,
                content=source,
                chunk_type="file",
            )
        ]

    lines = source.splitlines()
    chunks: list[Chunk] = []
    block_start = 0
    block_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.strip() == "":
            if block_lines:
                chunks.append(
                    Chunk(
                        path=path,
                        start_line=block_start + 1,
                        end_line=block_start + len(block_lines),
                        commit_sha=commit_sha,
                        content="\n".join(block_lines),
                        chunk_type="paragraph",
                    )
                )
                block_lines = []
            block_start = i + 1
        else:
            if not block_lines:
                block_start = i
            block_lines.append(line)

    if block_lines:
        chunks.append(
            Chunk(
                path=path,
                start_line=block_start + 1,
                end_line=block_start + len(block_lines),
                commit_sha=commit_sha,
                content="\n".join(block_lines),
                chunk_type="paragraph",
            )
        )

    if not chunks:
        return [
            Chunk(
                path=path,
                start_line=1,
                end_line=max(len(lines), 1),
                commit_sha=commit_sha,
                content=source,
                chunk_type="file",
            )
        ]
    return chunks
