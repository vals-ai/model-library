import logging

import pytest
from google.genai.types import GenerateContentConfig, ServiceTier, ThinkingLevel

from model_library.base import LLMConfig, QueryResultMetadata
from model_library.base.input import TextInput
from model_library.providers.google.google import (
    GoogleConfig,
    GoogleModel,
)

_INPUT = [TextInput(text="")]


async def test_service_tier_added_to_generation_config():
    model = GoogleModel(
        "gemini-2.5-flash",
        config=LLMConfig(provider_config=GoogleConfig(service_tier="flex")),
    )

    body = await model.build_body(_INPUT, tools=[])

    config = body["config"]
    assert isinstance(config, GenerateContentConfig)
    assert config.service_tier == ServiceTier.FLEX


async def test_service_tier_omitted_from_generation_config_by_default():
    model = GoogleModel("gemini-2.5-flash", config=LLMConfig())

    body = await model.build_body(_INPUT, tools=[])

    config = body["config"]
    assert isinstance(config, GenerateContentConfig)
    assert config.service_tier is None


def test_service_tier_rejected_for_openai_compatible_delegate():
    with pytest.raises(ValueError, match="service_tier"):
        GoogleModel(
            "gemini-2.5-flash",
            config=LLMConfig(
                native=False, provider_config=GoogleConfig(service_tier="flex")
            ),
        )


async def test_flex_service_tier_billed_at_batch_rate():
    metadata = QueryResultMetadata(in_tokens=1_000_000, out_tokens=1_000_000)
    key = "google/gemini-3.1-pro-preview"
    flex = GoogleModel(
        "gemini-3.1-pro-preview",
        config=LLMConfig(
            registry_key=key, provider_config=GoogleConfig(service_tier="flex")
        ),
    )
    standard = GoogleModel(
        "gemini-3.1-pro-preview", config=LLMConfig(registry_key=key)
    )

    flex_cost = await flex._calculate_cost(metadata)
    standard_cost = await standard._calculate_cost(metadata)

    assert flex_cost is not None and standard_cost is not None
    assert flex_cost.input == pytest.approx(standard_cost.input / 2)
    assert flex_cost.output == pytest.approx(standard_cost.output / 2)


async def test_thinking_level_models_accept_sampling_when_capable():
    model = GoogleModel(
        "gemini-3.5-flash",
        config=LLMConfig(temperature=0.5, top_p=0.9),
    )

    body = await model.build_body(_INPUT, tools=[], top_k=40)

    config = body["config"]
    assert config.temperature == 0.5
    assert config.top_p == 0.9
    assert config.top_k == 40


@pytest.mark.parametrize("budget_mode", [False, True])
@pytest.mark.parametrize("capable", [False, True])
async def test_google_sampling_capability_independent_of_thinking_mode(
    budget_mode, capable, caplog
):
    model = GoogleModel(
        "gemini-test",
        config=LLMConfig(
            temperature=0.4, top_p=0.8, top_k=11,
            supports_temperature=capable,
            provider_config=GoogleConfig(uses_thinking_budget=budget_mode),
        ),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        body = await model.build_body(
            _INPUT, tools=[], temperature=0, top_p=None, top_k=0
        )
    config = body["config"]
    assert config.temperature == 0
    assert config.top_p is None
    assert config.top_k == 0
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == (
        0 if capable else 3
    )


async def test_thinking_budget_models_accept_temperature():
    model = GoogleModel(
        "gemini-2.5-flash",
        config=LLMConfig(
            temperature=0.5,
            provider_config=GoogleConfig(uses_thinking_budget=True),
        ),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["config"].temperature == 0.5


async def test_thinking_level_models_leave_raw_thinking_budget_on_config(caplog):
    model = GoogleModel(
        "gemini-3.5-flash",
        config=LLMConfig(
            reasoning=True,
            reasoning_effort="high",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="llm"):
        body = await model.build_body(_INPUT, tools=[], thinking_budget=0)

    thinking_config = body["config"].thinking_config
    assert thinking_config.thinking_level == ThinkingLevel.HIGH
    assert thinking_config.thinking_budget is None
    assert body["config"].thinking_budget == 0
    assert "thinking_budget" not in caplog.text


async def test_thinking_level_models_accept_thinking_level_kwarg():
    model = GoogleModel(
        "gemini-3.5-flash",
        config=LLMConfig(
            reasoning=True,
        ),
    )

    body = await model.build_body(_INPUT, tools=[], thinking_level="low")

    assert body["config"].thinking_config.thinking_level == ThinkingLevel.LOW


async def test_thinking_budget_models_keep_thinking_budget():
    model = GoogleModel(
        "gemini-2.5-flash",
        config=LLMConfig(
            reasoning=True,
            provider_config=GoogleConfig(uses_thinking_budget=True),
        ),
    )

    body = await model.build_body(_INPUT, tools=[], thinking_budget=8192)

    thinking_config = body["config"].thinking_config
    assert thinking_config.thinking_budget == 8192
    assert thinking_config.thinking_level is None
