import json
import logging
from typing import Literal
from types import TracebackType
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import pytest

from anthropic.types.beta import BetaTextBlock
from openai._models import construct_type
from openai.types.chat import ChatCompletionChunk
from examples.extras.prompt_caching import _run_report

from model_library.base import LLMConfig, TextInput
from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.base.input import (
    RawResponse,
    SystemInput,
    ToolCall,
    ToolResult,
)
from model_library.providers.amazon.chat import AmazonModel
from model_library.registry_utils import compute_model_cost, get_registry_model
from tests.conftest import parametrize_models_for_provider
from tests.test_helpers import get_example_tool_input


@parametrize_models_for_provider("anthropic")
async def test_anthropic_build_body_adds_cache_control_on_system_only(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(
        [SystemInput(text=system_prompt), *input], tools=tools
    )

    # system should be a list with cache_control block
    assert isinstance(body["system"], list)
    assert body["system"][0].get("cache_control", {}).get("type") == "ephemeral"
    # tools should NOT carry cache_control when system prefix is cached
    assert not any("cache_control" in t for t in body["tools"])


@parametrize_models_for_provider("anthropic")
async def test_anthropic_build_body_caches_system_by_default(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(
        [SystemInput(text=system_prompt), *input], tools=tools
    )

    # Default: cache system with ephemeral type, no ttl provided
    assert isinstance(body["system"], list)
    cc = body["system"][0].get("cache_control", {})
    assert cc.get("type") == "ephemeral"
    assert "ttl" not in cc
    assert not any("cache_control" in t for t in body["tools"])


@parametrize_models_for_provider("anthropic")
async def test_anthropic_cache_control_on_tools_when_no_system(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(input, tools=tools)

    # No system -> last tool should carry cache_control
    assert body.get("system") is None
    assert body["tools"][-1].get("cache_control", {}).get("type") == "ephemeral"


@parametrize_models_for_provider("anthropic")
async def test_anthropic_prompt_caching_1h(model_key: str):
    model = get_registry_model(model_key, LLMConfig(prompt_caching="1h"))

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(
        [SystemInput(text=system_prompt), *input], tools=tools
    )

    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_1h_cache_write_cost():
    metadata = QueryResultMetadata(in_tokens=0, cache_write_tokens=1_000_000)

    cost_5m = compute_model_cost("anthropic/claude-opus-5-5", metadata)
    cost_1h = compute_model_cost(
        "anthropic/claude-opus-5-5", metadata, cache_ttl="1h"
    )

    assert cost_5m and cost_5m.cache_write == 5.0
    assert cost_1h and cost_1h.cache_write == 8.0


@pytest.mark.parametrize(
    ("model_key", "batch_cache_read"),
    [("google/gemini-3.8-flash", 0.0375), ("google/gemini-3-flash-preview", 0.05)],
)
def test_batch_cache_read_cost(model_key: str, batch_cache_read: float):
    metadata = QueryResultMetadata(in_tokens=0, cache_read_tokens=1_000_000)

    cost = compute_model_cost(model_key, metadata, batch=True)

    assert cost and cost.cache_read == pytest.approx(batch_cache_read)


def test_anthropic_cache_write_cost_uses_reported_ttl_split():
    metadata = QueryResultMetadata(
        in_tokens=0, cache_write_tokens=1_000_000, cache_write_1h_tokens=250_000
    )

    cost = compute_model_cost("anthropic/claude-opus-5-5", metadata, cache_ttl="5m")

    assert cost and cost.cache_write == 5.75


def test_kimi_prompt_caching_1h():
    model = get_registry_model("kimi/kimi-k3", LLMConfig(prompt_caching="1h"))
    metadata = QueryResultMetadata(in_tokens=0, cache_write_tokens=1_000_000)

    assert model._get_extra_body() == {  # pyright: ignore[reportPrivateUsage]
        "prompt_cache_options": {"mode": "implicit", "ttl": "1h"}
    }
    cost_5m = compute_model_cost("kimi/kimi-k3", metadata)
    cost_1h = compute_model_cost("kimi/kimi-k3", metadata, cache_ttl="1h")
    assert cost_5m and cost_5m.cache_write == 3.0
    assert cost_1h and cost_1h.cache_write == 6.0


async def test_kimi_cache_write_cost_uses_reported_ttl_split():
    model = get_registry_model("kimi/kimi-k3", LLMConfig(prompt_caching="1h"))
    assert model.delegate
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1_000_000},
    }
    chunk = construct_type(
        type_=ChatCompletionChunk,
        value={
            "id": "chatcmpl-1",
            "created": 0,
            "model": "kimi-k3",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "ok"}}],
            "usage": usage,
        },
    )
    stream = MagicMock()
    stream.__aiter__.return_value = [chunk]
    stream.response = httpx.Response(
        200,
        headers={
            "msh-usage-cache-write-tokens-1h": "0",
            "msh-usage-cache-write-tokens-5m": "1000000",
        },
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)

    with (
        patch.object(model.delegate, "get_client", return_value=client),
        patch.object(
            model.delegate, "build_body", new_callable=AsyncMock, return_value={}
        ),
    ):
        result = await model.delegate._query_impl(  # pyright: ignore[reportPrivateUsage]
            [], tools=[], query_logger=logging.getLogger(__name__)
        )

    assert result.metadata.cache_write_1h_tokens == 0
    cost = compute_model_cost("kimi/kimi-k3", result.metadata, cache_ttl="1h")
    assert cost and cost.cache_write == 3.0


@parametrize_models_for_provider("anthropic")
async def test_anthropic_prompt_caching_disabled_sends_no_cache_control(
    model_key: str,
):
    model = get_registry_model(model_key, LLMConfig(prompt_caching="off"))

    input, system_prompt, tools = get_example_tool_input()

    with_system = await model.build_body(
        [SystemInput(text=system_prompt), *input], tools=tools
    )
    without_system = await model.build_body(input, tools=tools)

    assert "cache_control" not in json.dumps([with_system, without_system])


@pytest.mark.parametrize(
    "model_key", ["bedrock/claude-opus-4-20250514-v1", "amazon/amazon.nova-lite-v1:0"]
)
async def test_bedrock_cache_point_on_trailing_tool_result(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()
    tool_call = ToolCall(id="t1", name="get_weather", args={"location": "SF"})
    body = await model.build_body(
        [
            SystemInput(text=system_prompt),
            *input,
            RawResponse(
                response={
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "get_weather",
                                "input": {"location": "SF"},
                            }
                        }
                    ],
                }
            ),
            ToolResult(tool_call=tool_call, result="15C"),
        ],
        tools=tools,
    )

    # Nova rejects a cachePoint next to a toolResult
    has_cache_point = body["messages"][-1]["content"][-1] == {
        "cachePoint": {"type": "default"}
    }
    assert has_cache_point is ("claude" in model_key)
    assert json.dumps(body).count("cachePoint") <= 4


@pytest.mark.parametrize("prompt_caching", ["5m", "off"])
async def test_openai_prompt_caching_disabled_sends_explicit_mode(
    prompt_caching: Literal["off", "5m", "1h"],
):
    model = get_registry_model(
        "openai/gpt-6-luna", LLMConfig(prompt_caching=prompt_caching)
    )

    body = await model.build_body([TextInput(text="hi")], tools=[])

    assert ("prompt_cache_options" in body) is (prompt_caching == "off")
    if prompt_caching == "off":
        assert body["prompt_cache_options"] == {"mode": "explicit"}


def test_prompt_caching_report_handles_missing_performance():
    report = _run_report(
        "no-performance",
        QueryResult(output_text="ok", metadata=QueryResultMetadata()),
    )

    assert report["performance"] is None


async def test_anthropic_query_maps_cache_usage():
    """End-to-end stub of _query_impl to verify cache fields are propagated."""

    class _DummyUsage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 2
            self.cache_read_input_tokens = 456
            self.cache_creation_input_tokens = 123
            self.cache_creation = None
            self.iterations = None

    class _DummyMessage:
        def __init__(self):
            self.id = "msg_1"
            self.model = "claude-haiku-4-5-20251001"
            self.content = [BetaTextBlock(type="text", text="hello")]
            self.usage = _DummyUsage()
            self.stop_reason = None

    class _DummyStream:
        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool:
            return False

        async def __aiter__(self):
            if False:
                yield None

        async def get_final_message(self):
            return _DummyMessage()

    class _DummyMessages:
        def stream(self, *args, **kwargs):  # noqa: D401
            return _DummyStream()

    class _DummyBeta:
        def __init__(self):
            self.messages = _DummyMessages()

    class _DummyClient:
        def __init__(self):
            self.beta = _DummyBeta()

    def get_dummy_client(
        api_key: str | None = None, base_url: str | None = None
    ) -> _DummyClient:
        return _DummyClient()

    model = get_registry_model("anthropic/claude-haiku-4-5-20251001")
    model.get_client = get_dummy_client

    res = await model._query_impl(
        [TextInput(text="hi")], tools=[], query_logger=MagicMock()
    )
    assert res.metadata.in_tokens == 10
    assert res.metadata.out_tokens == 2
    assert res.metadata.cache_read_tokens == 456
    assert res.metadata.cache_write_tokens == 123


@pytest.mark.parametrize(
    ("model_name", "cache_point"),
    [
        (
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"type": "default", "ttl": "1h"},
        ),
        ("us.anthropic.claude-opus-4-20250514-v1:0", {"type": "default"}),
    ],
)
async def test_bedrock_prompt_caching_1h(
    model_name: str, cache_point: dict[str, str]
):
    model = AmazonModel(model_name, config=LLMConfig(prompt_caching="1h"))

    body = await model.build_body(
        [SystemInput(text="system"), TextInput(text="hi")], tools=[]
    )

    assert body["system"][-1] == {"cachePoint": cache_point}
    assert body["messages"][-1]["content"][-1] == {"cachePoint": cache_point}
