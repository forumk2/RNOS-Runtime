from __future__ import annotations

from dataclasses import dataclass

import sqlite_vec

from .config import Config
from .db import get_connection
from .embedder import embed_query


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


def search(
    query: str, cfg: Config, top_k: int | None = None
) -> list[SearchResult]:
    """Return top-K chunks closest to the query vector.

    Args:
        query: Natural-language query string.
        cfg: Runtime configuration.
        top_k: Override for the number of results. Defaults to cfg.retrieval.top_k.
    """
    conn = get_connection()
    q_bytes = sqlite_vec.serialize_float32(embed_query(query, cfg))
    k = top_k if top_k is not None else cfg.retrieval.top_k

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

    return [
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
