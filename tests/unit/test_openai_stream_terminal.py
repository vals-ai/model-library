import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

from model_library.exceptions import ImmediateRetryException
from model_library.providers.openai import OpenAIModel


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "completed", "incomplete"])
async def test_stream_requires_terminal_state(status: str):
    response = Response.model_construct(
        id="resp_partial",
        status="in_progress",
        output=[],
        incomplete_details=None,
        usage=None,
        search_results=None,
    )

    async def events():
        yield SimpleNamespace(type="response.created", response=response)
        yield SimpleNamespace(type="response.output_text.delta", delta="Partial answer")
        if status != "in_progress":
            terminal = response.model_copy(
                update={
                    "status": status,
                    "incomplete_details": SimpleNamespace(reason="max_output_tokens")
                    if status == "incomplete"
                    else None,
                }
            )
            yield SimpleNamespace(type=f"response.{status}", response=terminal)

    raw = SimpleNamespace(
        request_id="req_partial", parse=AsyncMock(return_value=events())
    )
    client = MagicMock()
    client.responses.with_streaming_response.create.return_value.__aenter__ = AsyncMock(
        return_value=raw
    )
    model = MagicMock(use_completions=False)
    model.build_body = AsyncMock(return_value={})
    model.get_client.return_value = client

    if status == "in_progress":
        with pytest.raises(
            ImmediateRetryException,
            match="resp_partial",
        ):
            await OpenAIModel._query_impl(
                model, [], tools=[], query_logger=logging.getLogger(__name__)
            )
        return
    result = await OpenAIModel._query_impl(
        model, [], tools=[], query_logger=logging.getLogger(__name__)
    )
    assert result.finish_reason.reason.value == (
        "stop" if status == "completed" else "max_tokens"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nonstream_background_response_does_not_trigger_stream_retry():
    response = Response.model_construct(
        id="resp_background",
        status="in_progress",
        incomplete_details=None,
        usage=None,
        search_results=None,
        output=[
            ResponseOutputMessage.model_construct(
                type="message",
                content=[
                    ResponseOutputText.model_construct(
                        type="output_text",
                        text="Working",
                        annotations=[],
                    )
                ],
            )
        ],
    )
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    model = MagicMock(use_completions=False)
    model.build_body = AsyncMock(return_value={"background": True})
    model.get_client.return_value = client
    result = await OpenAIModel._query_impl(
        model,
        [],
        tools=[],
        query_logger=logging.getLogger(__name__),
        stream=False,
    )
    assert result.output_text == "Working"
    assert result.finish_reason.raw == "in_progress"
