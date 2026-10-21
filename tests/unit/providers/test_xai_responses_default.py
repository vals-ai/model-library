"""xAI Responses request bodies and cache keys."""

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from model_library.agent.tool import NativeWebSearch
from model_library.base import LLMConfig
from model_library.base.input import (
    FileWithBase64,
    FileWithId,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.query_ids import ResolvedQueryIds, scoped_query_ids
from model_library.exceptions import BadInputError
from model_library.providers.openai import OpenAIModel
from model_library.providers.xai import XAIConfig, XAIModel
from model_library.registry_utils import get_registry_model


def _delegate(model: XAIModel) -> OpenAIModel:
    assert isinstance(model.delegate, OpenAIModel)
    return model.delegate


async def test_registry_model_uses_responses_request_body():
    model = get_registry_model("grok/grok-4.7")
    body = await model.build_body(
        [TextInput(text="Search for something")], tools=[NativeWebSearch().definition]
    )

    assert body["model"] == "grok-4.7"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Search for something"}]}
    ]
    assert body["tools"] == [{"type": "web_search"}]


async def test_responses_body_requests_encrypted_reasoning():
    model = XAIModel("grok-4.7", config=LLMConfig(reasoning=True))
    body = await _delegate(model).build_body([TextInput(text="hello")], tools=[])
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["store"] is False
    assert body["reasoning"]["summary"] == "auto"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


async def test_prompt_cache_key_is_stable_across_reasoning_and_tool_turns():
    model = XAIModel("grok-4.7", config=LLMConfig(reasoning=True))
    tool = ToolDefinition(
        name="lookup",
        body=ToolBody(name="lookup", description="", properties={}, required=[]),
    )
    reasoning = ResponseReasoningItem(
        id="rs_1", type="reasoning", summary=[], encrypted_content="ciphertext"
    )
    call = ResponseFunctionToolCall(
        id="fc_1", call_id="call_1", type="function_call", name="lookup", arguments="{}"
    )
    first_turn = [SystemInput(text="You are Grok."), TextInput(text="Look it up.")]
    second_turn = [
        *first_turn,
        RawResponse(response=[reasoning, call]),
        ToolResult(tool_call=ToolCall(id="fc_1", call_id="call_1", name="lookup", args="{}"), result="42"),
    ]
    third_turn = [
        *second_turn,
        RawResponse(response=[reasoning, ResponseOutputMessage(id="msg_1", type="message", role="assistant", status="completed", content=[])]),
        TextInput(text="Thanks."),
    ]

    bodies = [
        await _delegate(model).build_body(turn, tools=[tool])
        for turn in (first_turn, second_turn, third_turn)
    ]

    assert len({body["prompt_cache_key"] for body in bodies}) == 1
    for previous, current in zip(bodies, bodies[1:]):
        assert current["input"][: len(previous["input"])] == previous["input"]
    assert bodies[2]["input"][2].encrypted_content == "ciphertext"
    assert bodies[2]["input"][4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "42",
    }


async def test_prompt_cache_key_differs_between_conversations():
    model = XAIModel("grok-4.7")
    first = await _delegate(model).build_body([TextInput(text="a")], tools=[])
    second = await _delegate(model).build_body([TextInput(text="b")], tools=[])
    assert first["prompt_cache_key"] != second["prompt_cache_key"]


async def test_prompt_cache_key_id_mode_uses_query_ids():
    model = XAIModel(
        "grok-4.7", config=LLMConfig(provider_config=XAIConfig(prompt_cache_key="id"))
    )
    with scoped_query_ids(
        ResolvedQueryIds(
            run_id="run", question_id="q1", query_id="x", attempt_id="attempt"
        )
    ):
        first = await _delegate(model).build_body([TextInput(text="a")], tools=[])
        second = await _delegate(model).build_body([TextInput(text="b")], tools=[])
    assert first["prompt_cache_key"] == second["prompt_cache_key"]


async def test_responses_rejects_image_file_id_as_unsupported():
    model = XAIModel("grok-4.7", config=LLMConfig(supports_images=True))
    with pytest.raises(BadInputError, match="does not support image file_id"):
        await _delegate(model).parse_image(
            FileWithId(type="image", name="a.png", mime="image/png", file_id="f1")
        )


async def test_responses_body_encodes_image_file_and_tools():
    model = XAIModel("grok-4.7", config=LLMConfig(supports_images=True))
    tool = ToolDefinition(
        name="echo",
        body=ToolBody(name="echo", description="echo", properties={}, required=[]),
    )
    body = await _delegate(model).build_body(
        [
            TextInput(text="describe"),
            FileWithBase64(type="image", name="a.png", mime="image/png", base64="AA"),
            FileWithId(type="file", name="a.pdf", mime="application/pdf", file_id="f1"),
        ],
        tools=[tool, NativeWebSearch().definition],
    )
    content = body["input"][0]["content"]
    assert content[1] == {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,AA",
    }
    assert content[2] == {"type": "input_file", "file_id": "f1"}
    assert [t["type"] for t in body["tools"]] == ["function", "web_search"]


