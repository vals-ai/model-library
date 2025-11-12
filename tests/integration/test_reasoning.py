"""
Integration tests for Google Gemini reasoning (real API when available).
"""

from typing import Any

import pytest

from tests.conftest import requires_google_api
from tests.test_helpers import assert_basic_result
from model_library.base import LLMConfig
from model_library.providers.google import GoogleModel
from model_library.providers.google.google import GoogleConfig


@pytest.mark.integration
@requires_google_api
class TestReasoningGenAI:
    """Test Google GenAI model reasoning."""

    @pytest.mark.asyncio
    async def test_thinking_mode(self, create_test_input: Any):
        """Test that GenAI reasoning/thinking mode actually produces reasoning output."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=True))
        result = await model.query(
            create_test_input("What is 25 * 4? Think step by step."),
            thinking_budget=-1,
        )
        assert_basic_result(result)
        assert "100" in (result.output_text or "")

        assert result.reasoning is not None, "Expected reasoning content but got None"
        assert len(result.reasoning) > 0, "Expected non-empty reasoning content"

        if result.metadata.reasoning_tokens is not None:
            assert result.metadata.reasoning_tokens > 0, (
                f"Expected positive reasoning tokens, got {result.metadata.reasoning_tokens}"
            )


@pytest.mark.integration
@requires_google_api
class TestReasoningVertex:
    """Test Google Vertex AI model reasoning."""

    @pytest.mark.asyncio
    async def test_thinking_mode(self, create_test_input: Any, setup_fixture: Any):
        """Test that Vertex AI reasoning/thinking mode actually produces reasoning output."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=True, provider_config=GoogleConfig(use_vertex=True)
            ),
        )
        result = await model.query(
            create_test_input("What is 25 * 4? Think step by step."),
            thinking_budget=-1,
        )
        assert_basic_result(result)
        assert "100" in (result.output_text or "")

        assert result.reasoning is not None, "Expected reasoning content but got None"
        assert len(result.reasoning) > 0, "Expected non-empty reasoning content"

        if result.metadata.reasoning_tokens is not None:
            assert result.metadata.reasoning_tokens > 0, (
                f"Expected positive reasoning tokens, got {result.metadata.reasoning_tokens}"
            )
