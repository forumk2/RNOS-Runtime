from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

DB_PATH = Path(".rnos-query/index.db")
EMBEDDING_DIM = 768  # nomic-embed-text-v1.5


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            path         TEXT    NOT NULL,
            start_line   INTEGER NOT NULL,
            end_line     INTEGER NOT NULL,
            commit_sha   TEXT,
            content      TEXT    NOT NULL,
            chunk_type   TEXT    NOT NULL,
            content_hash TEXT    NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id  INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_lookup
            ON chunks (path, start_line, content_hash)
    """)
    conn.commit()
