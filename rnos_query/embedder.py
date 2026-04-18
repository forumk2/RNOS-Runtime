from __future__ import annotations

from openai import OpenAI

from .config import Config


def _client(cfg: Config) -> OpenAI:
    return OpenAI(base_url=cfg.lm_studio.base_url, api_key="lm-studio")


def embed_documents(texts: list[str], cfg: Config) -> list[list[float]]:
    """Embed a batch of document strings with the search_document prefix."""
    prefixed = ["search_document: " + t for t in texts]
    response = _client(cfg).embeddings.create(
        model=cfg.lm_studio.embedding_model,
        input=prefixed,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str, cfg: Config) -> list[float]:
    """Embed a single query string with the search_query prefix."""
    response = _client(cfg).embeddings.create(
        model=cfg.lm_studio.embedding_model,
        input=["search_query: " + text],
    )
    return response.data[0].embedding
