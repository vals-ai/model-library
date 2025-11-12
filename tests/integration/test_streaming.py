"""
Integration tests for Google Gemini streaming with real API.
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
class TestStreamingGenAI:
    """Test streaming functionality with GenAI real API calls."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, create_test_input: Any):
        """Test basic GenAI streaming without reasoning."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=False))

        result = await model.query(
            create_test_input("What is 10 + 5? Reply with just the number."),
            stream=True,
        )
        print("result", result)

        assert_basic_result(result)

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        assert result.reasoning is None or result.reasoning == ""

    @pytest.mark.asyncio
    async def test_streaming_with_reasoning(self, create_test_input: Any):
        """Test GenAI streaming with reasoning enabled."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=True))

        result = await model.query(
            create_test_input("What is 25 * 4? Think step by step."),
            stream=True,
            thinking_budget=8192,
        )

        assert_basic_result(result)
        assert "100" in (result.output_text or "")

        assert result.reasoning is not None, "Expected reasoning content from streaming"
        assert len(result.reasoning) > 0, "Expected non-empty reasoning from streaming"

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        if result.metadata.reasoning_tokens is not None:
            assert result.metadata.reasoning_tokens > 0, (
                f"Expected positive reasoning tokens from streaming, got {result.metadata.reasoning_tokens}"
            )

    @pytest.mark.asyncio
    async def test_streaming_fallback(self, create_test_input: Any):
        """Test that GenAI streaming gracefully falls back to non-streaming on error."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=False))

        result = await model.query(
            create_test_input("What is 2 + 2?"),
            stream=True,
            debug_stream=True,
        )

        assert_basic_result(result)
        assert "4" in (result.output_text or "")

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0


@pytest.mark.integration
@requires_google_api
class TestStreamingVertex:
    """Test streaming functionality with Vertex AI real API calls."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, create_test_input: Any, setup_fixture: Any):
        """Test basic Vertex AI streaming without reasoning."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=False, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            create_test_input("What is 10 + 5? Reply with just the number."),
            stream=True,
        )
        print("result", result)

        assert_basic_result(result)

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        assert result.reasoning is None or result.reasoning == ""

    @pytest.mark.asyncio
    async def test_streaming_with_reasoning(
        self, create_test_input: Any, setup_fixture: Any
    ):
        """Test Vertex AI streaming with reasoning enabled."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=True, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            create_test_input("What is 25 * 4? Think step by step."),
            stream=True,
            thinking_budget=8192,
        )

        assert_basic_result(result)
        assert "100" in (result.output_text or "")

        assert result.reasoning is not None, "Expected reasoning content from streaming"
        assert len(result.reasoning) > 0, "Expected non-empty reasoning from streaming"

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        if result.metadata.reasoning_tokens is not None:
            assert result.metadata.reasoning_tokens > 0, (
                f"Expected positive reasoning tokens from streaming, got {result.metadata.reasoning_tokens}"
            )

    @pytest.mark.asyncio
    async def test_streaming_fallback(self, create_test_input: Any, setup_fixture: Any):
        """Test that Vertex AI streaming gracefully falls back to non-streaming on error."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=False, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            create_test_input("What is 2 + 2?"),
            stream=True,
            debug_stream=True,
        )

        assert_basic_result(result)
        assert "4" in (result.output_text or "")

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0
