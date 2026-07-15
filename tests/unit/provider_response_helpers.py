import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from model_library.base import TextInput
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.xai import XAIModel

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


def _xai_response(
    *,
    content: str | None,
    response_id: str | None = None,
    reasoning_content: str = "",
    finish_reason: str = "REASON_STOP",
    tool_calls: list[SimpleNamespace] | None = None,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> SimpleNamespace:
    response = SimpleNamespace(
        tool_calls=[] if tool_calls is None else tool_calls,
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=1,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
    )
    if response_id is not None:
        response.id = response_id
    return response


async def _query_xai(
    latest_response: SimpleNamespace,
    *,
    chunk_content: str | None,
):
    async def stream() -> AsyncIterator[tuple[SimpleNamespace, SimpleNamespace]]:
        yield (
            latest_response,
            SimpleNamespace(
                reasoning_content=latest_response.reasoning_content,
                content=chunk_content,
                tool_calls=latest_response.tool_calls,
            ),
        )

    chat = SimpleNamespace(stream=stream)
    client = SimpleNamespace(chat=SimpleNamespace(create=lambda **_: chat))
    model = XAIModel("grok-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)
