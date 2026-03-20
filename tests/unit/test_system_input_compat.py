"""Backward compatibility tests for system_prompt kwarg.

The system_prompt kwarg was deprecated in favor of passing SystemInput as
the first element of the input sequence. These tests verify that the old
kwarg still works via the query() and count_tokens() entry points.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_library.base.input import SystemInput, TextInput
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.openai import OpenAIModel


def _dummy_result() -> QueryResult:
    return QueryResult(
        output_text="ok",
        metadata=QueryResultMetadata(cost=QueryResultCost(input=0.0, output=0.0)),
        history=[],
    )


@pytest.mark.unit
class TestSystemPromptKwargCompat:
    """system_prompt kwarg is converted to SystemInput before _query_impl is called."""

    async def test_anthropic_prepends_system_input(self):
        model = AnthropicModel("claude-sonnet-4-20250514")

        mock_impl = AsyncMock(return_value=_dummy_result())
        model._query_impl = mock_impl  # type: ignore[method-assign]

        await model.query(
            [TextInput(text="hello")],
            system_prompt="You are helpful.",
        )

        received_input = mock_impl.call_args.args[0]
        assert isinstance(received_input[0], SystemInput)
        assert received_input[0].text == "You are helpful."
        assert isinstance(received_input[1], TextInput)

    async def test_openai_prepends_system_input(self):
        model = OpenAIModel("gpt-4o")

        mock_impl = AsyncMock(return_value=_dummy_result())
        model._query_impl = mock_impl  # type: ignore[method-assign]

        await model.query(
            [TextInput(text="hello")],
            system_prompt="Be concise.",
        )

        received_input = mock_impl.call_args.args[0]
        assert isinstance(received_input[0], SystemInput)
        assert received_input[0].text == "Be concise."

    async def test_system_prompt_kwarg_prepends_before_user_input(self):
        model = OpenAIModel("gpt-4o")

        mock_impl = AsyncMock(return_value=_dummy_result())
        model._query_impl = mock_impl  # type: ignore[method-assign]

        user_msg = TextInput(text="What is the weather?")
        await model.query([user_msg], system_prompt="Be concise.")

        received_input = mock_impl.call_args.args[0]
        assert isinstance(received_input[0], SystemInput)
        assert received_input[1] is user_msg

    async def test_no_system_prompt_kwarg_leaves_input_unchanged(self):
        model = OpenAIModel("gpt-4o")

        mock_impl = AsyncMock(return_value=_dummy_result())
        model._query_impl = mock_impl  # type: ignore[method-assign]

        user_msg = TextInput(text="hello")
        await model.query([user_msg])

        received_input = mock_impl.call_args.args[0]
        assert not any(isinstance(item, SystemInput) for item in received_input)
        assert received_input[0] is user_msg

    async def test_stringify_input_accepts_system_prompt_kwarg(self):
        """system_prompt kwarg also works for stringify_input (used by count_tokens)."""
        model = OpenAIModel("gpt-4o")

        result = await model.stringify_input(
            [TextInput(text="hello")],
            system_prompt="You are helpful.",
        )

        assert "You are helpful." in result
