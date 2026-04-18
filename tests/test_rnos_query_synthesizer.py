from __future__ import annotations

import unittest

from rnos_query.searcher import SearchResult
from rnos_query.synthesizer import (
    _LONG_ANSWER_RESERVE,
    _MIN_OUTPUT_TOKENS,
    _OUTPUT_RESERVE,
    _SYSTEM_PROMPT,
    _build_user_message,
    _count_request_tokens,
    _fit_request_to_context,
    _recommended_output_reserve,
    _select_chunks,
)


class SynthesizerSelectionTests(unittest.TestCase):
    def test_ask_prompt_treats_chunks_as_accessible_material(self) -> None:
        self.assertIn("do not say you lack access", _SYSTEM_PROMPT)
        self.assertIn("provided excerpts", _SYSTEM_PROMPT)

    def test_long_multipart_query_gets_larger_output_budget(self) -> None:
        query = (
            "Using the D3 paper, clearly explain the difference between early refusal "
            "and late refusal. For each type, give one concrete example. Then identify "
            "one limitation and finally suggest one architectural improvement."
        )
        self.assertEqual(_recommended_output_reserve(query, None), _LONG_ANSWER_RESERVE)
        self.assertEqual(_recommended_output_reserve("What is trust?", None), _OUTPUT_RESERVE)

    def test_select_chunks_does_not_drop_context_when_first_chunk_is_oversized(self) -> None:
        oversized = SearchResult(
            chunk_id=1,
            path="paper/d3_paper_v0.2.md",
            start_line=1,
            end_line=400,
            commit_sha="abc123",
            content="entropy " * 4000,
            chunk_type="markdown",
            score=0.01,
        )
        small = SearchResult(
            chunk_id=2,
            path="paper/d3_paper_v0.2.md",
            start_line=401,
            end_line=430,
            commit_sha="abc123",
            content="Early refusal is trust-triggered. Late refusal is entropy-triggered.",
            chunk_type="markdown",
            score=0.02,
        )
        chat_context = (
            _count_request_tokens(_SYSTEM_PROMPT, _build_user_message("early vs late refusal", []))
            + _OUTPUT_RESERVE
            + 280
        )

        selected = _select_chunks(
            [oversized, small],
            system_prompt=_SYSTEM_PROMPT,
            query="early vs late refusal",
            chat_context=chat_context,
            output_reserve=_OUTPUT_RESERVE,
        )

        self.assertTrue(selected)
        self.assertGreaterEqual(len(selected), 1)

    def test_select_chunks_can_truncate_first_chunk_to_fit_budget(self) -> None:
        oversized = SearchResult(
            chunk_id=1,
            path="paper/d3_paper_v0.2.md",
            start_line=398,
            end_line=519,
            commit_sha="abc123",
            content="trust entropy refusal " * 3000,
            chunk_type="markdown",
            score=0.01,
        )
        chat_context = (
            _count_request_tokens(_SYSTEM_PROMPT, _build_user_message("early vs late refusal", []))
            + _OUTPUT_RESERVE
            + 280
        )

        selected = _select_chunks(
            [oversized],
            system_prompt=_SYSTEM_PROMPT,
            query="early vs late refusal",
            chat_context=chat_context,
            output_reserve=_OUTPUT_RESERVE,
        )

        self.assertEqual(len(selected), 1)
        self.assertIn("[... truncated for context budget]", selected[0].content)

    def test_fit_request_to_context_keeps_prompt_under_context_limit(self) -> None:
        oversized = SearchResult(
            chunk_id=1,
            path="paper/d3_paper_v0.2.md",
            start_line=398,
            end_line=519,
            commit_sha="abc123",
            content="early refusal late refusal trust entropy " * 5000,
            chunk_type="markdown",
            score=0.01,
        )
        long_query = (
            "Using the D3 paper, clearly explain the difference between early refusal "
            "(trust-triggered) and late refusal (entropy-triggered). For each type, "
            "give one specific concrete example from the Core Battery or other experimental results. "
        ) * 4

        user_message, max_tokens = _fit_request_to_context(
            long_query,
            [oversized],
            _SYSTEM_PROMPT,
            4096,
            _OUTPUT_RESERVE,
        )

        self.assertLess(_count_request_tokens(_SYSTEM_PROMPT, user_message), 4096)
        self.assertGreaterEqual(max_tokens, 1)
        self.assertGreaterEqual(max_tokens, _MIN_OUTPUT_TOKENS)

    def test_fit_request_to_context_preserves_target_output_when_possible(self) -> None:
        oversized = SearchResult(
            chunk_id=1,
            path="paper/d3_paper_v0.2.md",
            start_line=398,
            end_line=519,
            commit_sha="abc123",
            content="early refusal late refusal trust entropy " * 5000,
            chunk_type="markdown",
            score=0.01,
        )
        long_query = (
            "Using the D3 paper, clearly explain the difference between early refusal "
            "(trust-triggered) and late refusal (entropy-triggered). For each type, "
            "give one specific concrete example from the Core Battery or other experimental results. "
            "Then identify one structural limitation and suggest one architectural improvement."
        )

        user_message, max_tokens = _fit_request_to_context(
            long_query,
            [oversized],
            _SYSTEM_PROMPT,
            4096,
            _LONG_ANSWER_RESERVE,
        )

        self.assertLess(_count_request_tokens(_SYSTEM_PROMPT, user_message) + max_tokens, 4096)
        self.assertEqual(max_tokens, _LONG_ANSWER_RESERVE)


if __name__ == "__main__":
    unittest.main()
