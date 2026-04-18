from __future__ import annotations

import tiktoken
from openai import OpenAI

from .config import Config
from .searcher import SearchResult

_SYSTEM_PROMPT = (
    "You answer questions about the RNOS-Runtime codebase using only "
    "the provided chunks. If the chunks don't contain the answer, say so. "
    "Cite every claim using the format [path:start-end]. Be concise."
)

_SYSTEM_TOKENS = 300
_QUERY_TOKENS = 150
_OUTPUT_RESERVE = 800

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _format_chunk(r: SearchResult) -> str:
    return f"[{r.path}:{r.start_line}-{r.end_line}]\n{r.content}"


def _select_chunks(results: list[SearchResult], chat_context: int) -> list[SearchResult]:
    budget = chat_context - _SYSTEM_TOKENS - _QUERY_TOKENS - _OUTPUT_RESERVE
    selected: list[SearchResult] = []
    used = 0
    for r in results:
        tokens = _count_tokens(_format_chunk(r))
        if used + tokens > budget:
            break
        selected.append(r)
        used += tokens
    return selected


def synthesize(query: str, results: list[SearchResult], cfg: Config) -> str:
    selected = _select_chunks(results, cfg.lm_studio.chat_context)
    chunks_text = "\n\n".join(_format_chunk(r) for r in selected)
    user_message = f"<chunks>\n{chunks_text}\n</chunks>\n\nQuestion: {query}"

    client = OpenAI(base_url=cfg.lm_studio.base_url, api_key="lm-studio")
    response = client.chat.completions.create(
        model=cfg.lm_studio.chat_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=_OUTPUT_RESERVE,
    )
    return response.choices[0].message.content or ""
