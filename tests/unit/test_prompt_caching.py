from types import TracebackType
from unittest.mock import MagicMock

from anthropic.types.beta import BetaTextBlock
from examples.extras.prompt_caching import _run_report

from model_library.base import TextInput
from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.base.input import SystemInput
from model_library.registry_utils import get_registry_model
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
