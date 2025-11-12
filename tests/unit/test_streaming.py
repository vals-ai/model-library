"""
Streaming tests for Gemini (Google) provider.

We validate that:
- accumulated output_text and reasoning are separated correctly
- usage metadata from the last chunk is captured
"""

from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

from model_library.base import LLMConfig, TextInput
from model_library.providers.google import GoogleModel


def _chunk(
    thought: str | None = None, text: str | None = None, with_usage: bool = False
):
    chunk = MagicMock()
    candidate = MagicMock()
    parts: list[Any] = []
    if thought is not None:
        p = MagicMock()
        p.text = thought
        p.thought = True
        p.function_call = None
        parts.append(p)
    if text is not None:
        p = MagicMock()
        p.text = text
        p.thought = False
        p.function_call = None
        parts.append(p)

    candidate.content.parts = parts
    chunk.candidates = [candidate]
    if with_usage:

        class Usage:
            def __init__(self):
                self.prompt_token_count = 12
                self.candidates_token_count = 34
                self.thoughts_token_count = 56
                self.cached_content_token_count = 0

        chunk.usage_metadata = Usage()
    else:
        chunk.usage_metadata = None
    return chunk


async def _aiter(items: List[Any]) -> AsyncIterator[Any]:
    for i in items:
        yield i


async def test_provider_streaming_aggregates(
    mock_google_client: Any, mock_model_settings: Any
):
    # Mock the client's stream to serve interleaved thought/output
    seq = [
        _chunk(thought="Plan: "),
        _chunk(thought="compute"),
        _chunk(text="Answer: 4", with_usage=True),
    ]

    async def _side_effect(**kwargs: Any):
        return _aiter(seq)

    mock_google_client.return_value.aio.models.generate_content_stream = AsyncMock(
        side_effect=_side_effect
    )

    model = GoogleModel(
        "gemini-2.5-flash", config=LLMConfig(reasoning=True, max_tokens=64)
    )
    result = await model.query([TextInput(text="2+2?")], stream=True)

    assert result.reasoning == "Plan: compute"
    assert result.output_text == "Answer: 4"
    assert result.metadata.in_tokens == 12
    assert result.metadata.out_tokens == 34
    assert result.metadata.reasoning_tokens == 56
