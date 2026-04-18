from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    chat_model: str = "qwen/qwen3-coder-30b"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    chat_context: int = 4096


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 6
    max_chunk_tokens: int = 400


@dataclass(slots=True)
class IndexingConfig:
    include: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.md", "**/*.txt", "**/*.log", "**/*.json"]
    )
    exclude: list[str] = field(
        default_factory=lambda: [".git", "__pycache__", ".rnos-query", "node_modules"]
    )


@dataclass(slots=True)
class Config:
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)


def load_config(path: Path | None = None) -> Config:
    if path is None:
        path = Path("rnos-query.toml")
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    lm = LMStudioConfig(**raw.get("lm_studio", {}))
    retrieval = RetrievalConfig(**raw.get("retrieval", {}))
    indexing = IndexingConfig(**raw.get("indexing", {}))
    return Config(lm_studio=lm, retrieval=retrieval, indexing=indexing)
