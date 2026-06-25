"""Backward compatibility tests for system_prompt kwarg.

The system_prompt kwarg was deprecated in favor of passing SystemInput as
the first element of the input sequence. These tests verify that the old
kwarg still works via the query() and count_tokens() entry points.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from model_library.base.base import LLMConfig
from model_library.base.input import FileWithId, FileWithUrl, SystemInput, TextInput
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google.google import GoogleConfig, GoogleModel
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
        model = AnthropicModel("claude-sonnet-4-5-20250929")

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

    async def test_stringify_input_accepts_system_prompt_only(self):
        model = OpenAIModel("gpt-4o")

        result = await model.stringify_input([], system_prompt="You are helpful.")

        assert "You are helpful." in result

    async def test_base_count_tokens_counts_system_prompt_only(self):
        model = OpenAIModel(
            "gpt-4o",
            config=LLMConfig(custom_api_key=SecretStr("dummy")),
        )

        tokens = await model.count_tokens([], system_prompt="You are helpful.")

        assert tokens > 0

    async def test_anthropic_native_count_tokens_uses_normalized_input(self):
        model = AnthropicModel("claude-sonnet-4-5-20250929")
        model.build_body = AsyncMock(return_value={"messages": []})  # type: ignore[method-assign]
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(return_value=MagicMock(input_tokens=7))
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [TextInput(text="hello")],
            system_prompt="You are helpful.",
        )

        assert tokens == 7
        received_input = model.build_body.call_args.args[0]
        assert isinstance(received_input[0], SystemInput)
        assert received_input[0].text == "You are helpful."
        assert isinstance(received_input[1], TextInput)

    async def test_anthropic_count_tokens_with_uploaded_file_id_drops_file_for_native_count(
        self,
    ):
        model = AnthropicModel(
            "claude-sonnet-4-5-20250929",
            config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("dummy")),
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(
            return_value=MagicMock(input_tokens=11)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [
                FileWithId(
                    type="file",
                    name="doc.txt",
                    mime="text/plain",
                    file_id="file-test",
                ),
                TextInput(text="Summarize the uploaded file."),
            ]
        )

        assert tokens == 11
        client.messages.count_tokens.assert_awaited_once()
        sent_messages = client.messages.count_tokens.call_args.kwargs["messages"]
        assert sent_messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize the uploaded file.",
                        "cache_control": model.cache_control,
                    }
                ],
            }
        ]

    async def test_anthropic_count_tokens_with_only_uploaded_file_id_falls_back(
        self,
    ):
        model = AnthropicModel(
            "claude-sonnet-4-5-20250929",
            config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("dummy")),
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(
            return_value=MagicMock(input_tokens=11)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [
                FileWithId(
                    type="file",
                    name="doc.txt",
                    mime="text/plain",
                    file_id="file-test",
                )
            ]
        )

        assert tokens > 0
        client.messages.count_tokens.assert_not_awaited()

    async def test_anthropic_count_tokens_with_pdf_file_url_drops_file_for_native_count(
        self,
    ):
        model = AnthropicModel(
            "claude-sonnet-4-5-20250929",
            config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("dummy")),
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(
            return_value=MagicMock(input_tokens=11)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [
                FileWithUrl(
                    type="file",
                    name="doc.pdf",
                    mime="application/pdf",
                    url="https://example.com/doc.pdf",
                ),
                TextInput(text="Summarize the linked file."),
            ]
        )

        assert tokens == 11
        client.messages.count_tokens.assert_awaited_once()
        sent_messages = client.messages.count_tokens.call_args.kwargs["messages"]
        assert sent_messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize the linked file.",
                        "cache_control": model.cache_control,
                    }
                ],
            }
        ]

    async def test_anthropic_count_tokens_with_only_pdf_file_url_falls_back(
        self,
    ):
        model = AnthropicModel(
            "claude-sonnet-4-5-20250929",
            config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("dummy")),
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(
            return_value=MagicMock(input_tokens=11)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [
                FileWithUrl(
                    type="file",
                    name="doc.pdf",
                    mime="application/pdf",
                    url="https://example.com/doc.pdf",
                )
            ]
        )

        assert tokens > 0
        client.messages.count_tokens.assert_not_awaited()

    async def test_anthropic_count_tokens_with_non_pdf_file_url_uses_native_count(
        self,
    ):
        model = AnthropicModel(
            "claude-sonnet-4-5-20250929",
            config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("dummy")),
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(
            return_value=MagicMock(input_tokens=11)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [
                FileWithUrl(
                    type="file",
                    name="doc.txt",
                    mime="text/plain",
                    url="https://example.com/doc.txt",
                ),
                TextInput(text="Summarize the linked file."),
            ]
        )

        assert tokens == 11
        client.messages.count_tokens.assert_awaited_once()

    async def test_google_vertex_count_tokens_consumes_system_prompt_before_parse_input(
        self,
    ):
        with patch.object(GoogleModel, "get_client", return_value=MagicMock()):
            model = GoogleModel(
                "gemini-2.5-flash",
                config=LLMConfig(
                    custom_api_key=SecretStr("{}"),
                    provider_config=GoogleConfig(use_vertex=True),
                ),
            )

        model.parse_input = AsyncMock(return_value=["parsed"])  # type: ignore[method-assign]
        client = MagicMock()
        client.aio.models.count_tokens = AsyncMock(
            return_value=MagicMock(total_tokens=9)
        )
        model.get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        tokens = await model.count_tokens(
            [TextInput(text="hello")],
            system_prompt="You are helpful.",
        )

        assert tokens == 9
        parsed_input = model.parse_input.call_args.args[0]
        assert len(parsed_input) == 1
        assert isinstance(parsed_input[0], TextInput)
        config = client.aio.models.count_tokens.call_args.kwargs["config"]
        assert config.system_instruction == "You are helpful."
