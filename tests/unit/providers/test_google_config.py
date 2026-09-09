import pytest
from google.genai.types import GenerateContentConfig, ServiceTier

from model_library.base import LLMConfig, QueryResultMetadata
from model_library.base.input import TextInput
from model_library.providers.google.google import GoogleConfig, GoogleModel

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
