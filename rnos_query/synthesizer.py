from __future__ import annotations

import tiktoken
from openai import OpenAI

from .config import Config
from .searcher import SearchResult

_SYSTEM_PROMPT = (
    "You answer questions about the RNOS-Runtime codebase using only "
    "the provided chunks. The chunks are the materials you have access to "
    "for this answer. If chunks are present, do not say you lack access to "
    "the paper, file, or documentation; answer from the provided excerpts "
    "and state only what is missing from those excerpts if needed. If the "
    "chunks don't contain the answer, say what is missing. Cite every claim "
    "using the format [path:start-end]. Be concise."
)

_EXPLORE_SYSTEM_PROMPT = """\
You are analyzing the RNOS-Runtime codebase. Use the provided context \
as grounding. Structure your response in exactly this format:

GROUNDED:
- ...

INFERRED:
- Observation:
- Implication:
- Risk or edge case:

PROPOSED:
- Mechanism:
- Trigger condition:
- Expected effect:
- Tradeoff:

Rules:
- Do not mix categories. A claim belongs to exactly one section.
- GROUNDED must contain only claims stated directly in the retrieved \
context. Every GROUNDED claim must have a citation in [path:start-end] \
format. Do not hallucinate files, line ranges, or citations.
- INFERRED must describe implications or relationships that are not \
explicitly stated in any single chunk but arise from combining them.
- INFERRED must combine multiple grounded facts where possible and \
focus on interactions, implications, boundary behavior, or edge cases.
- INFERRED must not restate GROUNDED facts, paraphrase a single cited \
claim, or summarize without adding insight.
- Citations are permitted in INFERRED, but they support the premises, \
not the inference itself.
- PROPOSED must be system-specific. Every PROPOSED item must reference \
at least one concrete system variable or mechanism from the grounded \
context.
- PROPOSED must propose concrete mechanisms, not generic improvements. \
Use actual RNOS concepts when available, such as entropy, trust, policy \
thresholds, retry_score, failure_score, cost_score, repeated_tool, \
latency_score, depth_score, DEGRADE, REFUSE, or ADE.
- Each PROPOSED item must include at least one trigger condition, one \
mechanism, one expected effect, and one tradeoff.
- PROPOSED must not include generic suggestions, unrelated ML ideas, or \
phrases like "could be improved" without a specific mechanism.
- If a query has a clean grounded answer and no useful proposals to \
make, write "No proposals for this query; the grounded and inferred \
sections are sufficient." in the PROPOSED section. Do not generate \
speculation to fill space.
- If a section is empty, write "None." Do not omit the section header.
- Be concise. Prefer fewer precise claims over many vague ones.

Before producing the final answer, check internally:
- Did any INFERRED line simply restate a grounded fact? If yes, revise it.
- Is each PROPOSED idea tied to a specific mechanism in the system? If \
not, revise it.\
"""

_OUTPUT_RESERVE = 800
_EXPLORE_OUTPUT_RESERVE = 2000
_TRUNCATION_MARKER = "\n[... truncated for context budget]"
_CHAT_OVERHEAD_TOKENS = 64
_CONTEXT_SLACK_TOKENS = 128
_MIN_OUTPUT_TOKENS = 128
_LONG_ANSWER_RESERVE = 1400
_LONG_ANSWER_QUERY_TOKENS = 70

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _format_chunk(r: SearchResult) -> str:
    return f"[{r.path}:{r.start_line}-{r.end_line}]\n{r.content}"


def _build_user_message(query: str, chunks: list[SearchResult]) -> str:
    chunks_text = "\n\n".join(_format_chunk(r) for r in chunks)
    return f"<chunks>\n{chunks_text}\n</chunks>\n\nQuestion: {query}"


def _count_request_tokens(system_prompt: str, user_message: str) -> int:
    return _count_tokens(system_prompt) + _count_tokens(user_message) + _CHAT_OVERHEAD_TOKENS


def _recommended_output_reserve(query: str, explicit_reserve: int | None) -> int:
    """Return a response budget sized to the apparent complexity of the query."""
    if explicit_reserve is not None:
        return explicit_reserve

    normalized = query.lower()
    if _count_tokens(query) >= _LONG_ANSWER_QUERY_TOKENS:
        return _LONG_ANSWER_RESERVE
    if any(marker in normalized for marker in ("for each", "then,", "finally,", "identify one", "suggest one")):
        return _LONG_ANSWER_RESERVE
    return _OUTPUT_RESERVE


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Return text truncated to at most ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""

    encoded = _enc.encode(text)
    if len(encoded) <= max_tokens:
        return text

    marker_tokens = _enc.encode(_TRUNCATION_MARKER)
    payload_budget = max(1, max_tokens - len(marker_tokens))
    return _enc.decode(encoded[:payload_budget]).rstrip() + _TRUNCATION_MARKER


def _truncate_chunk_to_budget(r: SearchResult, max_tokens: int) -> SearchResult | None:
    """Return a truncated chunk variant that fits the context budget."""
    header = f"[{r.path}:{r.start_line}-{r.end_line}]\n"
    header_tokens = _count_tokens(header)
    if max_tokens <= header_tokens:
        return None

    truncated_content = _truncate_text_to_tokens(r.content, max_tokens - header_tokens)
    if not truncated_content.strip():
        return None

    return SearchResult(
        chunk_id=r.chunk_id,
        path=r.path,
        start_line=r.start_line,
        end_line=r.end_line,
        commit_sha=r.commit_sha,
        content=truncated_content,
        chunk_type=r.chunk_type,
        score=r.score,
    )


def _select_chunks(
    results: list[SearchResult],
    *,
    system_prompt: str,
    query: str,
    chat_context: int,
    output_reserve: int,
) -> list[SearchResult]:
    empty_user_message = _build_user_message(query, [])
    budget = max(
        chat_context
        - _count_request_tokens(system_prompt, empty_user_message)
        - output_reserve
        - _CONTEXT_SLACK_TOKENS,
        0,
    )
    selected: list[SearchResult] = []
    used = 0
    for r in results:
        tokens = _count_tokens(_format_chunk(r))
        remaining = budget - used
        if remaining <= 0:
            break
        if tokens > remaining:
            if not selected:
                truncated = _truncate_chunk_to_budget(r, remaining)
                if truncated is not None:
                    selected.append(truncated)
            continue
        selected.append(r)
        used += tokens
    return selected


def _fit_request_to_context(
    query: str,
    results: list[SearchResult],
    system_prompt: str,
    chat_context: int,
    desired_output_tokens: int,
) -> tuple[str, int]:
    selected = list(results)
    target_output = max(_MIN_OUTPUT_TOKENS, desired_output_tokens)
    while True:
        user_message = _build_user_message(query, selected)
        prompt_tokens = _count_request_tokens(system_prompt, user_message)
        available_output = chat_context - prompt_tokens - _CONTEXT_SLACK_TOKENS
        if available_output >= target_output:
            return user_message, target_output

        if selected:
            overflow = max(target_output - available_output, 1)
            last = selected[-1]
            last_tokens = _count_tokens(_format_chunk(last))
            target_tokens = last_tokens - overflow - 8
            truncated = _truncate_chunk_to_budget(last, target_tokens)
            if truncated is not None and _count_tokens(_format_chunk(truncated)) < last_tokens:
                selected[-1] = truncated
                continue
            selected.pop()
            continue

        empty_message = _build_user_message(query, [])
        prompt_without_chunks = _count_request_tokens(system_prompt, empty_message)
        available_without_chunks = chat_context - prompt_without_chunks - _CONTEXT_SLACK_TOKENS
        if available_without_chunks >= 1:
            return empty_message, max(1, min(target_output, available_without_chunks))

        truncated_query = _truncate_text_to_tokens(
            query,
            max(
                chat_context
                - _count_tokens(system_prompt)
                - _CHAT_OVERHEAD_TOKENS
                - _CONTEXT_SLACK_TOKENS
                - 32,
                1,
            ),
        )
        user_message = _build_user_message(truncated_query, [])
        prompt_tokens = _count_request_tokens(system_prompt, user_message)
        return user_message, max(
            1,
            min(target_output, chat_context - prompt_tokens - _CONTEXT_SLACK_TOKENS),
        )


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
    reserve = _recommended_output_reserve(query, output_reserve)
    prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT

    selected = _select_chunks(
        results,
        system_prompt=prompt,
        query=query,
        chat_context=cfg.lm_studio.chat_context,
        output_reserve=reserve,
    )
    user_message, max_tokens = _fit_request_to_context(
        query,
        selected,
        prompt,
        cfg.lm_studio.chat_context,
        reserve,
    )

    client = OpenAI(base_url=cfg.lm_studio.base_url, api_key="lm-studio")
    response = client.chat.completions.create(
        model=cfg.lm_studio.chat_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
