import pytest
from pydantic import ValidationError

from model_gateway.types import QueryRequest
from model_library.base import LLMConfig
from model_library.providers.openai.openai import OpenAIConfig


def _request(provider_config: dict[str, object] | None) -> dict[str, object]:
    config = {} if provider_config is None else {"provider_config": provider_config}
    return {"model": "openai/gpt-6-sol", "inputs": [], "config": config}


def test_request_provider_config_cannot_set_access_programs():
    with pytest.raises(ValidationError, match="access_programs cannot be set"):
        QueryRequest.model_validate(
            _request({"access_programs": {"cyber": "daybreak_blue"}})
        )


def test_typed_provider_config_cannot_set_access_programs():
    config = LLMConfig(
        provider_config=OpenAIConfig(access_programs={"cyber": "daybreak_blue"})
    )

    with pytest.raises(ValidationError, match="access_programs cannot be set"):
        QueryRequest(model="openai/gpt-6-sol", inputs=[], config=config)


@pytest.mark.parametrize("provider_config", [{"verbosity": "low"}, {"access_programs": None}])
def test_request_provider_config_without_access_programs_is_accepted(
    provider_config: dict[str, object],
):
    request = QueryRequest.model_validate(_request(provider_config))

    assert request.config.provider_config == provider_config


def test_request_with_its_own_api_key_may_set_access_programs():
    request = QueryRequest.model_validate(
        {
            "model": "openai/gpt-6-sol",
            "inputs": [],
            "config": {
                "custom_api_key": "sk-caller",
                "provider_config": {"access_programs": {"cyber": "daybreak_blue"}},
            },
        }
    )

    assert request.config.custom_api_key is not None
