from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import sqlite_vec

from .config import Config
from .db import get_connection
from .embedder import embed_query

# Matches bare filenames and relative paths that end with a known source extension,
# e.g. "policy.py", "rnos/policy.py", "README.md".
_PATH_HINT_RE = re.compile(
    r'\b[\w][\w./\-]*\.(py|md|json|yaml|yml|txt|toml|log)\b'
)


@dataclass(slots=True)
class SearchResult:
    chunk_id: int
    path: str
    start_line: int
    end_line: int
    commit_sha: str | None
    content: str
    chunk_type: str
    score: float
    """Cosine distance in [0, 2]; -1.0 means pinned by path hint (not vector-ranked)."""


def _extract_path_hints(query: str) -> list[str]:
    """Return any filename or path strings found in the query text."""
    return [m.group() for m in _PATH_HINT_RE.finditer(query)]


def _chunks_by_path(conn: sqlite3.Connection, hint: str) -> list[SearchResult]:
    """Return all indexed chunks whose path contains hint, ordered by start_line.

    Score is set to -1.0 to indicate these were pinned by name, not vector-ranked.
    """
    rows = conn.execute(
        """
        SELECT id, path, start_line, end_line, commit_sha, content, chunk_type
        FROM chunks
        WHERE path LIKE ?
        ORDER BY path, start_line
        """,
        [f"%{hint}%"],
    ).fetchall()
    return [
        SearchResult(
            chunk_id=r[0],
            path=r[1],
            start_line=r[2],
            end_line=r[3],
            commit_sha=r[4],
            content=r[5],
            chunk_type=r[6],
            score=-1.0,
        )
        for r in rows
    ]


def search(
    query: str,
    cfg: Config,
    top_k: int | None = None,
    path_filter: str | None = None,
) -> list[SearchResult]:
    """Return chunks for the query, with hybrid path-hint merging.

    Retrieval runs in two phases:

    1. Vector KNN: top-k nearest neighbours by embedding distance.
    2. Path pinning: if the query text mentions a filename (e.g. "policy.py")
       or path_filter is supplied, all chunks from that path are fetched from
       the DB and prepended to the result list ahead of the vector hits.

    Chunks that appear in both phases keep their vector score. Pinned-only
    chunks carry score=-1.0 as a sentinel. The synthesizer sees pinned chunks
    first, so they are included in context before the token budget runs out.

    Args:
        query: Natural-language query string.
        cfg: Runtime configuration.
        top_k: Override for result count. Defaults to cfg.retrieval.top_k.
        path_filter: Explicit path substring filter (e.g. "rnos/policy.py").
    """
    conn = get_connection()
    k = top_k if top_k is not None else cfg.retrieval.top_k
    q_bytes = sqlite_vec.serialize_float32(embed_query(query, cfg))

    rows = conn.execute(
        """
        SELECT c.id, c.path, c.start_line, c.end_line,
               c.commit_sha, c.content, c.chunk_type, v.distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        [q_bytes, k],
    ).fetchall()

    vector_results = [
        SearchResult(
            chunk_id=r[0],
            path=r[1],
            start_line=r[2],
            end_line=r[3],
            commit_sha=r[4],
            content=r[5],
            chunk_type=r[6],
            score=r[7],
        )
        for r in rows
    ]

    # Collect path hints: explicit filter first, then auto-detected from query.
    hints: list[str] = []
    if path_filter:
        hints.append(path_filter)
    for h in _extract_path_hints(query):
        if h not in hints:
            hints.append(h)

    if not hints:
        return vector_results

    # Fetch pinned chunks for each hint, deduplicated across hints.
    pinned: list[SearchResult] = []
    pinned_ids: set[int] = set()
    for hint in hints:
        for chunk in _chunks_by_path(conn, hint):
            if chunk.chunk_id not in pinned_ids:
                pinned.append(chunk)
                pinned_ids.add(chunk.chunk_id)

    # For any pinned chunk that also appeared in vector results, keep the
    # vector score so debug output shows real distances.
    vector_by_id = {r.chunk_id: r for r in vector_results}
    resolved_pinned = [
        vector_by_id.get(p.chunk_id, p) for p in pinned
    ]

    # Pinned chunks first, then vector results not already covered.
    remainder = [r for r in vector_results if r.chunk_id not in pinned_ids]
    return resolved_pinned + remainder
