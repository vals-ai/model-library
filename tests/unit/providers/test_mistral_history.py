from typing import Any

from mistralai.client.models import (
    AssistantMessage,
    FunctionCall,
    TextChunk,
    ToolCall as MistralToolCall,
)

from model_library.base import TextInput
from model_library.base.input import RawResponse, ToolCall, ToolResult
from model_library.providers.mistral import MistralModel


def _assistant(
    *tool_call_ids: str, text: str | None = "Let me look."
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextChunk(text=text, type="text")] if text else [],
        tool_calls=[
            MistralToolCall(
                id=tool_call_id,
                function=FunctionCall(name="bash", arguments='{"command": "ls"}'),
            )
            for tool_call_id in tool_call_ids
        ]
        or None,
    )


def _tool_result(tool_call_id: str) -> ToolResult:
    return ToolResult(
        tool_call=ToolCall(id=tool_call_id, name="bash", args={"command": "ls"}),
        result="file.txt",
    )


async def _parse(*items: Any) -> list[dict[str, Any] | Any]:
    return await MistralModel("mistral-test").parse_input(list(items))


async def test_unanswered_tool_call_followed_by_user_text_is_dropped():
    messages = await _parse(
        TextInput(text="Reply with JSON."),
        RawResponse(response=_assistant("call-1")),
        TextInput(text="Previous response had parsing errors."),
    )

    assert [m["role"] if isinstance(m, dict) else m.role for m in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assistant = messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls is None
    assert assistant.content == [TextChunk(text="Let me look.", type="text")]


async def test_unanswered_tool_call_without_content_removes_message():
    messages = await _parse(
        TextInput(text="Reply with JSON."),
        RawResponse(response=_assistant("call-1", text=None)),
        TextInput(text="Previous response had parsing errors."),
    )

    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Reply with JSON."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Previous response had parsing errors."}
            ],
        },
    ]


async def test_answered_tool_calls_are_kept():
    assistant = _assistant("call-1", "call-2")
    messages = await _parse(
        TextInput(text="List files."),
        RawResponse(response=assistant),
        _tool_result("call-1"),
        _tool_result("call-2"),
        TextInput(text="Thanks, now summarize."),
    )

    assert messages[1] is assistant
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == ["call-1", "call-2"]
    assert [m["tool_call_id"] for m in messages[2:4]] == ["call-1", "call-2"]



