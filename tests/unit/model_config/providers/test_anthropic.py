"""Tests for Anthropic provider configuration."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from anthropic.types.beta import (
    BetaFallbackBlock,
    BetaFallbackMessageIterationUsage,
    BetaMessageIterationUsage,
)
from pydantic import ValidationError
import pytest

from model_library.base import LLMConfig, QueryResult, QueryResultMetadata
from model_library.base.output import FallbackHop, FallbackInfo
from model_library.base.input import TextInput
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.registry_utils import get_model_cost, get_registry_model

_INPUT = [TextInput(text="")]

async def _query_anthropic_with_provider_config(
    provider_config: AnthropicConfig,
    *,
    model_name: str = "claude-primary-test",
    thinking_tokens: int | None = None,
    fallback_blocks: list[BetaFallbackBlock] | None = None,
    custom_endpoint: str | None = None,
) -> tuple[dict[str, object], QueryResult]:
    captured: dict[str, object] = {}

    # one declined `message` hop per fallback block, then the served `fallback_message` hop
    declined_models = (
        [block.from_.model for block in fallback_blocks]
        if fallback_blocks
        else [model_name]
    )
    served_model = fallback_blocks[-1].to.model if fallback_blocks else "claude-fallback-test"
    fallback_iterations = [
        *(
            BetaMessageIterationUsage(
                type="message",
                model=declined,
                input_tokens=1,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
            for declined in declined_models
        ),
        BetaFallbackMessageIterationUsage(
            type="fallback_message",
            model=served_model,
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    ]

    class _DummyUsage:
        input_tokens = 1
        output_tokens = 7 if thinking_tokens is not None else 1
        output_tokens_details = None
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        iterations = fallback_iterations if provider_config.fallback_models else None

    class _DummyMessage:
        id = "msg_test"
        model = "claude-primary-test"
        content = [
            SimpleNamespace(type="text", text="ok"),
            *(fallback_blocks or []),
        ]
        usage = _DummyUsage()
        stop_reason = "end_turn"

    class _DummyStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

        async def __aiter__(self):
            if thinking_tokens is not None:
                yield SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(
                        output_tokens_details=SimpleNamespace(
                            thinking_tokens=thinking_tokens
                        )
                    ),
                )

        async def get_final_message(self):
            return _DummyMessage()

    class _DummyMessages:
        def stream(self, **kwargs: object) -> _DummyStream:
            captured.update(kwargs)
            return _DummyStream()

    class _DummyBeta:
        messages = _DummyMessages()

    class _DummyClient:
        beta = _DummyBeta()

    model = AnthropicModel(
        model_name,
        config=LLMConfig(
            max_tokens=4096,
            reasoning=False,
            provider_config=provider_config,
            custom_endpoint=custom_endpoint,
            custom_api_key="test-key" if custom_endpoint else None,
        ),
    )
    object.__setattr__(model, "get_client", MagicMock(return_value=_DummyClient()))

    result = await model._query_impl(_INPUT, tools=[], query_logger=MagicMock())
    return captured, result


class TestAnthropicConfig:
    async def test_supports_auto_thinking_uses_adaptive(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=True,
                provider_config=AnthropicConfig(supports_auto_thinking=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"] == {"type": "adaptive"}

    async def test_no_auto_thinking_uses_enabled_with_budget(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=True,
                provider_config=AnthropicConfig(supports_auto_thinking=False),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"]["type"] == "enabled"
        assert "budget_tokens" in body["thinking"]

    async def test_no_thinking_when_reasoning_disabled(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=False,
                provider_config=AnthropicConfig(supports_auto_thinking=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"] == {"type": "disabled"}

    async def test_supports_compute_effort_adds_output_config(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(supports_compute_effort=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["output_config"] == {"effort": "max"}

    async def test_compute_effort_not_added_when_unsupported(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(supports_compute_effort=False),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "output_config" not in body

    async def test_compute_effort_not_added_when_no_effort_value(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                provider_config=AnthropicConfig(supports_compute_effort=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "output_config" not in body

    async def test_task_budget_tokens_are_sent_with_the_beta(self):
        captured, _ = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                supports_auto_thinking=True,
                supports_compute_effort=True,
                task_budget_tokens=64_000,
            )
        )

        assert captured["output_config"] == {
            "task_budget": {"type": "tokens", "total": 64_000}
        }
        assert captured["betas"] == [
            "files-api-2025-04-14",
            "task-budgets-2026-03-13",
        ]

    async def test_no_task_budget_leaves_the_request_untouched(self):
        captured, _ = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                supports_auto_thinking=True,
                supports_compute_effort=True,
            )
        )

        assert "output_config" not in captured
        assert captured["betas"] == ["files-api-2025-04-14"]

    async def test_task_budget_sends_no_beta_off_the_anthropic_endpoint(self):
        captured, _ = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                supports_auto_thinking=True,
                supports_compute_effort=True,
                task_budget_tokens=64_000,
            ),
            custom_endpoint="https://bedrock.example.com/v1/",
        )

        assert "betas" not in captured

    async def test_count_tokens_drops_the_task_budget(self):
        captured: dict[str, object] = {}

        class _DummyMessages:
            async def count_tokens(self, **kwargs: object):
                captured.update(kwargs)
                return SimpleNamespace(input_tokens=11)

        class _DummyClient:
            messages = _DummyMessages()

        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(
                    supports_compute_effort=True,
                    task_budget_tokens=64_000,
                ),
            ),
        )
        object.__setattr__(model, "get_client", MagicMock(return_value=_DummyClient()))

        assert await model.count_tokens(_INPUT) == 11
        assert captured["output_config"] == {"effort": "max"}

    async def test_task_budget_keeps_compute_effort(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(
                    supports_compute_effort=True,
                    task_budget_tokens=20_000,
                ),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["output_config"] == {
            "effort": "max",
            "task_budget": {"type": "tokens", "total": 20_000},
        }

    async def test_server_side_fallback_models_uses_current_request_shape(self):
        captured, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                fallback_models=["claude-fallback-test"],
                supports_auto_thinking=True,
            )
        )

        assert captured["betas"] == [
            "files-api-2025-04-14",
            "server-side-fallback-2026-06-01",
        ]
        extra_body = cast(dict[str, object], captured["extra_body"])
        assert extra_body == {"fallbacks": [{"model": "claude-fallback-test"}]}
        assert result.metadata.fallback == FallbackInfo(
            requested_model="anthropic/claude-primary-test",
            served_model="anthropic/claude-fallback-test",
            hops=[
                FallbackHop(
                    model="anthropic/claude-primary-test",
                    served=False,
                    usage=QueryResultMetadata(
                        in_tokens=1,
                        out_tokens=0,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                    ),
                ),
                FallbackHop(
                    model="anthropic/claude-fallback-test",
                    served=True,
                    usage=QueryResultMetadata(
                        in_tokens=1,
                        out_tokens=1,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                    ),
                ),
            ],
        )
        assert result.metadata.extra == {
            "anthropic_response_model": "claude-primary-test"
        }

    async def test_no_fallback_leaves_metadata_fallback_unset(self):
        _, result = await _query_anthropic_with_provider_config(AnthropicConfig())

        assert result.metadata.fallback is None

    async def test_server_side_fallback_records_trigger_per_declined_hop(self):
        blocks = [
            BetaFallbackBlock.model_validate(
                {
                    "type": "fallback",
                    "from": {"model": "claude-primary-test"},
                    "to": {"model": "claude-fallback-test"},
                    "trigger": {"type": "refusal", "category": "cyber"},
                }
            ),
            BetaFallbackBlock.model_validate(
                {
                    "type": "fallback",
                    "from": {"model": "claude-fallback-test"},
                    "to": {"model": "claude-backup-test"},
                    "trigger": {"type": "refusal", "category": "bio"},
                }
            ),
        ]

        _, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                fallback_models=["claude-fallback-test", "claude-backup-test"]
            ),
            fallback_blocks=blocks,
        )

        assert result.metadata.fallback is not None
        assert result.metadata.fallback.served_model == "anthropic/claude-backup-test"
        assert [
            (hop.model, hop.served, hop.trigger, hop.category)
            for hop in result.metadata.fallback.hops
        ] == [
            ("anthropic/claude-primary-test", False, "refusal", "cyber"),
            ("anthropic/claude-fallback-test", False, "refusal", "bio"),
            ("anthropic/claude-backup-test", True, None, None),
        ]

    async def test_stream_thinking_tokens_are_split_and_billed_once(self):
        _, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(supports_auto_thinking=True),
            thinking_tokens=3,
        )

        assert result.metadata.out_tokens == 4
        assert result.metadata.reasoning_tokens == 3
        assert result.metadata.total_output_tokens == 7

        model = get_registry_model("anthropic/claude-opus-5-max")
        assert model.metadata is not None
        assert model.metadata.costs_per_million_token is not None
        output_price = model.metadata.costs_per_million_token.output
        cost = await model._calculate_cost(result.metadata)

        assert cost is not None
        assert cost.output == pytest.approx(4 * output_price / 1_000_000)
        assert cost.reasoning == pytest.approx(3 * output_price / 1_000_000)
        assert cost.total_output == pytest.approx(7 * output_price / 1_000_000)

    async def test_zero_stream_thinking_tokens_preserve_none(self):
        _, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(supports_auto_thinking=True),
            thinking_tokens=0,
        )

        assert result.metadata.out_tokens == 7
        assert result.metadata.reasoning_tokens is None

    async def test_server_side_fallback_models_preserve_order(self):
        captured, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                fallback_models=["claude-fallback-test", "claude-backup-test"],
                supports_auto_thinking=True,
            )
        )

        assert captured["betas"] == [
            "files-api-2025-04-14",
            "server-side-fallback-2026-06-01",
        ]
        extra_body = cast(dict[str, object], captured["extra_body"])
        assert extra_body == {
            "fallbacks": [
                {"model": "claude-fallback-test"},
                {"model": "claude-backup-test"},
            ]
        }
        assert result.metadata.fallback is not None

    async def test_fallback_cost_is_billed_at_serving_model_price(self):
        requested = "anthropic/claude-fable-5-1"
        served = "anthropic/claude-sonnet-4-6"
        model = get_registry_model(requested)
        requested_costs = get_model_cost(requested)
        served_costs = get_model_cost(served)
        assert requested_costs is not None and served_costs is not None
        assert served_costs.cache is not None
        assert requested_costs.output != served_costs.output

        metadata = QueryResultMetadata(
            in_tokens=1_100,
            out_tokens=180,
            reasoning_tokens=20,
            cache_read_tokens=500,
            cache_write_tokens=0,
            fallback=FallbackInfo(
                requested_model=requested,
                served_model=served,
                hops=[],
            ),
        )

        cost = await model._calculate_cost(metadata)

        million = 1_000_000
        assert cost is not None
        assert cost.input == pytest.approx(1_100 * served_costs.input / million)
        assert cost.cache_read == pytest.approx(500 * served_costs.cache.read / million)
        assert cost.output == pytest.approx(180 * served_costs.output / million)
        assert cost.reasoning == pytest.approx(20 * served_costs.output / million)

    async def test_unregistered_fallback_model_yields_cost_none(self):
        model = get_registry_model("anthropic/claude-opus-5")
        result = QueryResult(
            output_text="ok",
            metadata=QueryResultMetadata(
                in_tokens=1,
                out_tokens=1,
                fallback=FallbackInfo(
                    requested_model="anthropic/claude-opus-5",
                    served_model="anthropic/not-in-registry",
                    hops=[],
                ),
            ),
            history=[],
        )
        model._query_impl = AsyncMock(return_value=result)  # type: ignore[method-assign]

        output = await model.query(_INPUT)

        assert output.metadata.cost is None
        assert output.metadata.fallback is not None

    def test_fallback_model_is_not_a_supported_config_field(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AnthropicConfig.model_validate({"fallback_model": "claude-fallback-test"})

    def test_fallback_models_rejects_more_than_three_entries(self):
        with pytest.raises(ValueError, match="at most 3"):
            AnthropicConfig(
                fallback_models=[
                    "claude-fallback-test",
                    "claude-backup-test",
                    "claude-third-test",
                    "claude-fourth-test",
                ]
            )

    def test_fallback_models_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            AnthropicConfig(
                fallback_models=["claude-fallback-test", "claude-fallback-test"]
            )

    async def test_fallback_models_rejects_requested_model(self):
        with pytest.raises(ValueError, match="must not include requested model"):
            await _query_anthropic_with_provider_config(
                AnthropicConfig(
                    fallback_models=["claude-primary-test"],
                    supports_auto_thinking=True,
                ),
                model_name="claude-primary-test",
            )
