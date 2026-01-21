from types import TracebackType
from unittest.mock import MagicMock

from model_library.base import TextInput
from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_models_for_provider
from tests.test_helpers import get_example_tool_input


@parametrize_models_for_provider("anthropic")
async def test_anthropic_build_body_adds_cache_control_on_system_only(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(input, tools=tools, system_prompt=system_prompt)

    # system should be a list with cache_control block
    assert isinstance(body["system"], list)
    assert body["system"][0].get("cache_control", {}).get("type") == "ephemeral"
    # tools should NOT carry cache_control when system prefix is cached
    assert not any("cache_control" in t for t in body["tools"])


@parametrize_models_for_provider("anthropic")
async def test_anthropic_build_body_caches_system_by_default(model_key: str):
    model = get_registry_model(model_key)

    input, system_prompt, tools = get_example_tool_input()

    body = await model.build_body(input, tools=tools, system_prompt=system_prompt)

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

    model = get_registry_model("anthropic/claude-haiku-4-5-20251001")
    model.get_client = lambda: _DummyClient()

    res = await model._query_impl(
        [TextInput(text="hi")], tools=[], query_logger=MagicMock()
    )
    assert res.metadata.in_tokens == 10
    assert res.metadata.out_tokens == 2
    assert res.metadata.cache_read_tokens == 456
    assert res.metadata.cache_write_tokens == 123
