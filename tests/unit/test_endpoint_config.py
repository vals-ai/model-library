"""Unit tests for custom_endpoint"""

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from model_library.base import LLMConfig, client_registry
from model_library.providers.openai import OpenAIModel

# Save the original get_client before conftest's autouse fixture replaces it
_original_get_client = OpenAIModel.get_client


def clear_registry():
    client_registry.clear()


class TestCustomEndpoint:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_custom_endpoint_on_llm_config(self):
        """custom_endpoint can be set on LLMConfig."""
        config = LLMConfig(custom_endpoint="https://custom.api.com/v1")

        assert config.custom_endpoint == "https://custom.api.com/v1"

    def test_custom_endpoint_defaults_to_none(self):
        """custom_endpoint is None by default."""
        config = LLMConfig()
        assert config.custom_endpoint is None

    def test_custom_endpoint_passed_to_openai(self):
        """When custom_endpoint is set, it's passed as base_url to client."""
        config = LLMConfig(
            custom_endpoint="https://my-proxy.com/v1",
            custom_api_key=SecretStr("proxy-key"),
        )

        OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]
        try:
            with patch(
                "model_library.providers.openai.create_openai_client_with_defaults"
            ) as mock_create:
                mock_create.return_value = MagicMock()

                OpenAIModel("gpt-4", config=config)

                mock_create.assert_called_once_with(
                    base_url="https://my-proxy.com/v1",
                    api_key="proxy-key",
                    dns_resolve=None,
                )
        finally:
            OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]

    def test_custom_endpoint_disables_batch(self):
        """When custom_endpoint is set, batch support is disabled."""
        config = LLMConfig(
            custom_endpoint="https://custom.api.com/v1",
            custom_api_key=SecretStr("test-key"),
            supports_batch=True,
        )

        model = OpenAIModel("gpt-4", config=config)
        assert model.supports_batch is False

    def test_custom_endpoint_without_api_key_uses_default(self):
        """custom_endpoint can be set without custom_api_key — uses provider default."""
        config = LLMConfig(
            custom_endpoint="https://my-proxy.com/v1",
        )

        OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]
        try:
            with patch(
                "model_library.providers.openai.create_openai_client_with_defaults"
            ) as mock_create:
                mock_create.return_value = MagicMock()

                OpenAIModel("gpt-4", config=config)

                call_kwargs = mock_create.call_args
                assert call_kwargs.kwargs.get("base_url") == "https://my-proxy.com/v1"
        finally:
            OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]

    def test_custom_api_key_without_endpoint(self):
        """custom_api_key can be set without custom_endpoint — uses default URL."""
        config = LLMConfig(
            custom_api_key=SecretStr("my-key"),
        )

        OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]
        try:
            with patch(
                "model_library.providers.openai.create_openai_client_with_defaults"
            ) as mock_create:
                mock_create.return_value = MagicMock()

                OpenAIModel("gpt-4", config=config)

                mock_create.assert_called_once_with(
                    base_url=None,
                    api_key="my-key",
                    dns_resolve=None,
                )
        finally:
            OpenAIModel.get_client = _original_get_client  # type: ignore[method-assign]
