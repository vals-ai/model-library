from types import TracebackType

import pytest

from model_library.base import LLMConfig, TextInput, ToolBody, ToolDefinition
from model_library.providers.anthropic import AnthropicModel


@pytest.mark.asyncio
async def test_anthropic_create_body_adds_cache_control_on_system_only():
    m = AnthropicModel("claude-haiku-4-5-20251001", config=LLMConfig())

    tools = [
        ToolDefinition(
            name="get_time",
            body=ToolBody(
                name="get_time",
                description="Get time",
                properties={"tz": {"type": "string"}},
                required=["tz"],
            ),
        )
    ]

    body = await m.create_body(
        [TextInput(text="hi")],
        tools=tools,
        system_prompt="sys",
    )

    # system should be a list with cache_control block
    assert isinstance(body["system"], list)
    assert body["system"][0].get("cache_control", {}).get("type") == "ephemeral"
    # tools should NOT carry cache_control when system prefix is cached
    assert not any("cache_control" in t for t in body["tools"])


@pytest.mark.asyncio
async def test_anthropic_create_body_caches_system_by_default():
    m = AnthropicModel("claude-haiku-4-5-20251001", config=LLMConfig())

    tools = [
        ToolDefinition(
            name="get_time",
            body=ToolBody(
                name="get_time",
                description="Get time",
                properties={"tz": {"type": "string"}},
                required=["tz"],
            ),
        )
    ]

    body = await m.create_body(
        [TextInput(text="hi")],
        tools=tools,
        system_prompt="sys",
    )
    # Default: cache system with ephemeral type, no ttl provided
    assert isinstance(body["system"], list)
    cc = body["system"][0].get("cache_control", {})
    assert cc.get("type") == "ephemeral"
    assert "ttl" not in cc
    assert not any("cache_control" in t for t in body["tools"])


@pytest.mark.asyncio
async def test_anthropic_cache_control_on_tools_when_no_system():
    m = AnthropicModel("claude-haiku-4-5-20251001", config=LLMConfig())

    tools = [
        ToolDefinition(
            name="search_documents",
            body=ToolBody(
                name="search_documents",
                description="Search",
                properties={"q": {"type": "string"}},
                required=["q"],
            ),
        ),
        ToolDefinition(
            name="get_document",
            body=ToolBody(
                name="get_document",
                description="Fetch doc",
                properties={"id": {"type": "string"}},
                required=["id"],
            ),
        ),
    ]

    body = await m.create_body(
        [TextInput(text="hi")],
        tools=tools,
    )

    # No system -> last tool should carry cache_control
    assert body.get("system") is None
    assert body["tools"][-1].get("cache_control", {}).get("type") == "ephemeral"


@pytest.mark.asyncio
async def test_anthropic_query_maps_cache_usage():
    """End-to-end stub of _query_impl to verify cache fields are propagated."""

    class _DummyUsage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 2
            self.cache_read_input_tokens = 456
            self.cache_creation_input_tokens = 123

    class _DummyText:
        def __init__(self, text: str):
            self.type = "text"
            self.text = text

    class _DummyMessage:
        def __init__(self):
            self.id = "msg_1"
            self.content = [_DummyText("hello")]  # minimal block
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

    m = AnthropicModel("claude-haiku-4-5-20251001", config=LLMConfig())
    m.get_client = lambda: _DummyClient()

    res = await m._query_impl([TextInput(text="hi")], tools=[])
    assert res.metadata.in_tokens == 10
    assert res.metadata.out_tokens == 2
    assert res.metadata.cache_read_tokens == 456
    assert res.metadata.cache_write_tokens == 123
