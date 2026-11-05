"""`workload_identity` models authenticate by federation, never by a key."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from anthropic.lib.credentials import WorkloadIdentityCredentials
from pydantic import SecretStr

from model_library.base import LLMConfig
import model_library.providers.anthropic as anthropic_provider
from model_library.providers.anthropic import (
    AnthropicConfig,
    AnthropicModel,
    _cyber_client_key,
)

ENDPOINT = "https://cyber-credentials-test.anthropic.example"


@pytest.fixture(autouse=True)
def mock_all_get_client():
    """Build real SDK clients; no request leaves the test."""
    yield


def _credentials() -> WorkloadIdentityCredentials:
    return WorkloadIdentityCredentials(
        identity_token_provider=lambda: "jwt",
        federation_rule_id="fdrl_test",
        organization_id="00000000-0000-0000-0000-000000000000",
    )


def _config(registry_key: str, *, native: bool = True) -> LLMConfig:
    # Supply a static key explicitly to prove the federated client ignores it.
    return LLMConfig(
        registry_key=registry_key,
        custom_api_key=SecretStr("sk-static"),
        custom_endpoint=ENDPOINT,
        native=native,
        provider_config=AnthropicConfig(
            workload_identity=registry_key.endswith("-cyber")
        ),
    )


@pytest.mark.unit
def test_cyber_model_uses_federated_credentials_and_ignores_the_static_key() -> None:
    with patch(
        "model_library.providers.anthropic.cyber_credentials", _credentials
    ):
        cyber = AnthropicModel(
            "claude-opus-5-5", config=_config("anthropic/claude-opus-5-5-cyber")
        )
        plain = AnthropicModel(
            "claude-opus-5-5", config=_config("anthropic/claude-opus-5-5")
        )

    client = cyber.get_client()
    assert client.api_key is None
    assert client.credentials is not None
    assert plain.get_client().api_key == "sk-static"
    assert cyber.get_client() is not plain.get_client()


@pytest.mark.unit
def test_cyber_model_refuses_the_key_only_openai_path() -> None:
    with pytest.raises(ValueError, match="requires the native client"):
        AnthropicModel(
            "claude-opus-5-5",
            config=_config("anthropic/claude-opus-5-5-cyber", native=False),
        )


@pytest.mark.unit
def test_new_federation_ids_get_a_new_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def cyber_client(rule_id: str) -> object:
        settings = SimpleNamespace(
            ANTHROPIC_CYBER_FEDERATION_RULE_ID=rule_id,
            ANTHROPIC_CYBER_ORGANIZATION_ID="00000000-0000-0000-0000-000000000000",
            ANTHROPIC_CYBER_SERVICE_ACCOUNT_ID="svac_test",
            ANTHROPIC_CYBER_WORKSPACE_ID="wrkspc_test",
        )
        monkeypatch.setattr(anthropic_provider, "model_library_settings", settings)
        model = AnthropicModel(
            "claude-opus-5-5", config=_config("anthropic/claude-opus-5-5-cyber")
        )
        return model.get_client()

    with patch.object(anthropic_provider, "cyber_credentials", _credentials):
        assert cyber_client("fdrl_old") is not cyber_client("fdrl_new")


@pytest.mark.unit
@pytest.mark.parametrize("cyber_first", [True, False])
def test_a_static_key_equal_to_the_federated_slot_gets_its_own_client(
    cyber_first: bool,
) -> None:
    """A request may supply any `custom_api_key`, including the federated slot's stand-in."""

    def cyber() -> AnthropicModel:
        return AnthropicModel(
            "claude-opus-5-5", config=_config("anthropic/claude-opus-5-5-cyber")
        )

    def plain() -> AnthropicModel:
        return AnthropicModel(
            "claude-opus-5-5",
            config=LLMConfig(
                registry_key="anthropic/claude-opus-5-5",
                custom_api_key=SecretStr(_cyber_client_key()),
                custom_endpoint=ENDPOINT,
            ),
        )

    with patch.object(anthropic_provider, "cyber_credentials", _credentials):
        first, second = (cyber(), plain()) if cyber_first else (plain(), cyber())

    assert first.get_client() is not second.get_client()
    federated, static = (first, second) if cyber_first else (second, first)
    assert federated.get_client().credentials is not None
    assert static.get_client().api_key == _cyber_client_key()
