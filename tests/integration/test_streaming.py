"""
Integration tests for Google Gemini streaming with real API.
"""

from model_library.base import LLMConfig
from model_library.providers.google import GoogleModel
from model_library.providers.google.google import GoogleConfig
from tests.test_helpers import assert_basic_result


class TestStreamingGenAI:
    """Test streaming functionality with GenAI real API calls."""

    async def test_streaming_basic(self):
        """Test basic GenAI streaming without reasoning."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=False))

        result = await model.query(
            "What is 10 + 5? Reply with just the number.",
            stream=True,
        )
        print("result", result)

        assert_basic_result(result)

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        assert result.reasoning is None or result.reasoning == ""

    async def test_streaming_with_reasoning(self):
        """Test GenAI streaming with reasoning enabled."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=True))

        result = await model.query(
            "What is 25 * 4? Think step by step.",
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

    async def test_streaming_fallback(self):
        """Test that GenAI streaming gracefully falls back to non-streaming on error."""
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(reasoning=False))

        result = await model.query(
            "What is 2 + 2?",
            stream=True,
            debug_stream=True,
        )

        assert_basic_result(result)
        assert "4" in (result.output_text or "")

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0


class TestStreamingVertex:
    """Test streaming functionality with Vertex AI real API calls."""

    async def test_streaming_basic(self):
        """Test basic Vertex AI streaming without reasoning."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=False, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            "What is 10 + 5? Reply with just the number.",
            stream=True,
        )
        print("result", result)

        assert_basic_result(result)

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0

        assert result.reasoning is None or result.reasoning == ""

    async def test_streaming_with_reasoning(self):
        """Test Vertex AI streaming with reasoning enabled."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=True, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            "What is 25 * 4? Think step by step.",
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

    async def test_streaming_fallback(self):
        """Test that Vertex AI streaming gracefully falls back to non-streaming on error."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(
                reasoning=False, provider_config=GoogleConfig(use_vertex=True)
            ),
        )

        result = await model.query(
            "What is 2 + 2?",
            stream=True,
            debug_stream=True,
        )

        assert_basic_result(result)
        assert "4" in (result.output_text or "")

        assert result.metadata.in_tokens > 0
        assert result.metadata.out_tokens > 0
