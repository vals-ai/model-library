"""Unit tests for NVIDIA provider."""

from unittest.mock import AsyncMock, patch

from model_library.base import QueryResult, RawInput, RawResponse, TextInput
from model_library.providers.delegates.nvidia import NvidiaModel
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function


class TestNvidiaBuildBodyToolCallsFix:
    """NVIDIA's vLLM endpoint rejects assistant messages carrying tool_calls=None
    ('Input should be iterable'). Such messages must be sent without the tool_calls
    field. It also rejects assistant content=None, which must be sent as ""."""

    async def test_assistant_message_with_null_tool_calls_drops_tool_calls(self):
        """A degenerate assistant turn is serialized without null fields Nvidia rejects."""
        model = NvidiaModel("public-nvidia-test-model")
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content=None,
            tool_calls=None,
            reasoning_content="Let me check the git history of the test files.",
        )

        body = await model.build_body([RawResponse(response=msg)], tools=[])

        messages = body["messages"]
        assert len(messages) == 1
        m = messages[0]
        assert isinstance(m, dict)
        assert m["content"] == ""
        assert "tool_calls" not in m, (
            "tool_calls=None must not be sent to the NVIDIA endpoint"
        )
        assert (
            m["reasoning_content"] == "Let me check the git history of the test files."
        )

    async def test_assistant_message_with_real_tool_calls_preserves_tool_calls(self):
        """Assistant messages that carry real tool calls preserve them."""
        model = NvidiaModel("public-nvidia-test-model")
        msg = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="execute_bash", arguments='{"command": "ls"}'
                    ),
                )
            ],
        )

        body = await model.build_body([RawResponse(response=msg)], tools=[])

        messages = body["messages"]
        assert len(messages) == 1
        m = messages[0]
        if isinstance(m, dict):
            assert m["content"] == ""
            tool_calls = m["tool_calls"]
        else:
            assert m.content == ""
            tool_calls = m.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1

    async def test_assistant_dict_with_null_content_and_tool_calls_normalized(self):
        model = NvidiaModel("public-nvidia-test-model")

        body = await model.build_body(
            [
                RawInput(
                    input={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": None,
                        "reasoning_content": "thinking",
                    }
                )
            ],
            tools=[],
        )

        messages = body["messages"]
        assert len(messages) == 1
        assert messages[0] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
        }

    async def test_query_normalizes_input_before_delegating(self):
        model = NvidiaModel("public-nvidia-test-model")
        assert model.delegate is not None

        with patch.object(
            model.delegate,
            "_query_impl",
            new=AsyncMock(return_value=QueryResult(output_text="ok")),
        ) as delegate_query:
            await model.query(
                [TextInput(text="respond")],
                history=[
                    RawInput(
                        input={
                            "role": "assistant",
                            "content": None,
                            "tool_calls": None,
                            "reasoning_content": "thinking",
                        }
                    )
                ],
            )

        delegate_query.assert_awaited_once()
        assert delegate_query.await_args is not None
        delegated_input = delegate_query.await_args.args[0]
        assert isinstance(delegated_input[0], RawInput)
        assert delegated_input[0].input == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
        }
