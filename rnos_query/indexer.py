from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import git
import sqlite_vec

from .chunker import Chunk, chunk_file
from .config import Config
from .db import get_connection, init_schema
from .embedder import embed_documents

_BATCH_SIZE = 32


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _head_sha(repo_root: Path) -> str | None:
    try:
        repo = git.Repo(repo_root, search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except Exception:
        return None


def _is_excluded(path: Path, root: Path, excludes: list[str]) -> bool:
    parts = set(path.relative_to(root).parts)
    for excl in excludes:
        if excl.rstrip("/") in parts:
            return True
    return False


def _iter_files(root: Path, cfg: Config) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for pattern in cfg.indexing.include:
        for p in root.glob(pattern):
            if p.is_file() and p not in seen:
                if not _is_excluded(p, root, cfg.indexing.exclude):
                    seen.add(p)
                    result.append(p)
    return result


def run_index(root: Path, cfg: Config) -> None:
    conn = get_connection()
    init_schema(conn)

    commit_sha = _head_sha(root)
    files = _iter_files(root, cfg)

    pending: list[tuple[Chunk, str]] = []
    skipped = 0

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = file_path.relative_to(root).as_posix()
        for chunk in chunk_file(source, rel, file_path.suffix, commit_sha):
            chash = _content_hash(chunk.content)
            exists = conn.execute(
                "SELECT 1 FROM chunks WHERE path=? AND start_line=? AND content_hash=?",
                (chunk.path, chunk.start_line, chash),
            ).fetchone()
            if exists:
                skipped += 1
            else:
                pending.append((chunk, chash))

    inserted = 0
    total = len(pending)
    for i in range(0, total, _BATCH_SIZE):
        batch = pending[i : i + _BATCH_SIZE]
        texts = [c.content for c, _ in batch]
        try:
            embeddings = embed_documents(texts, cfg)
        except Exception as exc:
            print(f"\nError embedding batch {i // _BATCH_SIZE + 1}: {exc}", file=sys.stderr)
            print("Is LM Studio running with the embedding model loaded?", file=sys.stderr)
            raise SystemExit(1)

        for (chunk, chash), emb in zip(batch, embeddings):
            cur = conn.execute(
                """
                INSERT INTO chunks (path, start_line, end_line, commit_sha, content, chunk_type, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.commit_sha,
                    chunk.content,
                    chunk.chunk_type,
                    chash,
                ),
            )
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(emb)),
            )
            inserted += 1

        conn.commit()
        done = min(i + _BATCH_SIZE, total)
        print(f"  Embedded {done}/{total} chunks...", end="\r", flush=True)

    print(f"\nIndexed: {inserted} new chunks, {skipped} unchanged.")
