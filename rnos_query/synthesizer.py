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

_EXPLORE_SYSTEM_PROMPT = """\
You are analyzing the RNOS-Runtime codebase. Use the provided context \
as grounding. Structure your response in three sections:

GROUNDED -- claims stated directly in the retrieved context. Each claim \
must have a citation in [path:start-end] format.

INFERRED -- claims that follow from the grounded facts via explicit \
reasoning. State the reasoning briefly ("Because X [citation], Y \
follows"). Citations are permitted here; they support the premises, \
not the inference.

PROPOSED -- extensions, hypotheses, or integration ideas that go past \
the evidence. No citations in this section. Citations imply grounding; \
proposals are not grounded.

Rules:
- Do not mix categories. A claim belongs to exactly one section.
- If a query has a clean grounded answer and no useful proposals to \
make, write "No proposals for this query; the grounded and inferred \
sections are sufficient." in the PROPOSED section. Do not generate \
speculation to fill space.
- If a section is empty, write "None." Do not omit the section header.
- Be concise. Prefer fewer precise claims over many vague ones.\
"""

_SYSTEM_TOKENS = 300
_QUERY_TOKENS = 150
_OUTPUT_RESERVE = 800
_EXPLORE_OUTPUT_RESERVE = 2000

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _format_chunk(r: SearchResult) -> str:
    return f"[{r.path}:{r.start_line}-{r.end_line}]\n{r.content}"


def _select_chunks(
    results: list[SearchResult], chat_context: int, output_reserve: int
) -> list[SearchResult]:
    budget = chat_context - _SYSTEM_TOKENS - _QUERY_TOKENS - output_reserve
    selected: list[SearchResult] = []
    used = 0
    for r in results:
        tokens = _count_tokens(_format_chunk(r))
        if used + tokens > budget:
            break
        selected.append(r)
        used += tokens
    return selected


def synthesize(
    query: str,
    results: list[SearchResult],
    cfg: Config,
    *,
    system_prompt: str | None = None,
    output_reserve: int | None = None,
) -> str:
    """Call the chat model and return its response.

    Args:
        query: The user query.
        results: Retrieved chunks to include as context.
        cfg: Runtime configuration.
        system_prompt: Override the default system prompt.
        output_reserve: Override the default token reservation for output.
    """
    reserve = output_reserve if output_reserve is not None else _OUTPUT_RESERVE
    prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT

    selected = _select_chunks(results, cfg.lm_studio.chat_context, reserve)
    chunks_text = "\n\n".join(_format_chunk(r) for r in selected)
    user_message = f"<chunks>\n{chunks_text}\n</chunks>\n\nQuestion: {query}"

    client = OpenAI(base_url=cfg.lm_studio.base_url, api_key="lm-studio")
    response = client.chat.completions.create(
        model=cfg.lm_studio.chat_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=reserve,
    )
    return response.choices[0].message.content or ""
