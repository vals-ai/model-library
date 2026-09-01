"""Tests for Anthropic provider configuration."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from anthropic.types.beta import BetaFallbackBlock
from pydantic import ValidationError
import pytest

from model_library.base import LLMConfig, QueryResult
from model_library.base.input import TextInput
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.registry_utils import get_registry_model

_INPUT = [TextInput(text="")]

async def _query_anthropic_with_provider_config(
    provider_config: AnthropicConfig,
    *,
    model_name: str = "claude-primary-test",
    thinking_tokens: int | None = None,
    fallback_block: BetaFallbackBlock | None = None,
    custom_endpoint: str | None = None,
) -> tuple[dict[str, object], QueryResult]:
    captured: dict[str, object] = {}

    class _DummyIteration:
        type = "fallback_message"
        model = "claude-fallback-test"

    class _DummyUsage:
        input_tokens = 1
        output_tokens = 7 if thinking_tokens is not None else 1
        output_tokens_details = None
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        iterations = [_DummyIteration()]

    class _DummyMessage:
        id = "msg_test"
        model = "claude-primary-test"
        content = [
            SimpleNamespace(type="text", text="ok"),
            *([fallback_block] if fallback_block is not None else []),
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
        assert result.metadata.extra["fallback"] is True
        assert "anthropic_fallback_blocks" not in result.metadata.extra

    async def test_server_side_fallback_retains_native_boundary_block(self):
        fallback_block = BetaFallbackBlock.model_validate(
            {
                "type": "fallback",
                "from": {"model": "claude-primary-test"},
                "to": {"model": "claude-fallback-test"},
                "trigger": {"type": "refusal", "category": "general_harms"},
            }
        )

        _, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(fallback_models=["claude-fallback-test"]),
            fallback_block=fallback_block,
        )

        assert result.metadata.extra["fallback"] is True
        assert result.metadata.extra["anthropic_response_model"] == (
            "claude-primary-test"
        )
        assert "anthropic_usage_iterations" in result.metadata.extra
        assert result.metadata.extra["anthropic_fallback_blocks"] == [
            {
                "type": "fallback",
                "from": {"model": "claude-primary-test"},
                "to": {"model": "claude-fallback-test"},
                "trigger": {"type": "refusal", "category": "general_harms"},
            }
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
        assert result.metadata.extra["fallback"] is True

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
