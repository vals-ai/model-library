import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from model_library.base import TextInput
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel

_INPUT = [TextInput(text="hello")]
_LOGGER = logging.getLogger("test")


def _amazon_response(
    *events: dict[str, Any],
    stop_reason: str = "end_turn",
    output_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "ResponseMetadata": {"RequestId": "amazon-request-1"},
        "stream": [
            {"messageStart": {"role": "assistant"}},
            *events,
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": output_tokens}}},
            {"messageStop": {"stopReason": stop_reason}},
        ],
    }


async def _query_amazon(response: dict[str, Any]):
    model = AmazonModel("anthropic.claude")
    client = MagicMock()
    client.converse_stream.return_value = response

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)


_AMAZON_BLOCK_STOP = {"contentBlockStop": {}}
_AMAZON_TOOL_START = {
    "contentBlockStart": {"start": {"toolUse": {"toolUseId": "tool-1", "name": "ping"}}}
}
_AMAZON_TOOL_DELTA = {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}


def _amazon_text(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"delta": {"text": text}}}


def _amazon_reasoning(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": text}}}}


class _AnthropicMockStream:
    def __init__(
        self,
        message: SimpleNamespace,
        events: list[SimpleNamespace] | None = None,
    ) -> None:
        self._message = message
        self._events = events or []

    async def __aenter__(self) -> "_AnthropicMockStream":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[SimpleNamespace]:
        for event in self._events:
            yield event

    async def get_final_message(self) -> SimpleNamespace:
        return self._message


async def _query_anthropic(
    message: SimpleNamespace,
    events: list[SimpleNamespace] | None = None,
):
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=lambda **_: _AnthropicMockStream(message, events)
            )
        )
    )
    model = AnthropicModel("claude-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)
