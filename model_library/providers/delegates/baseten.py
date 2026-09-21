from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import DelegateOnly, LLMConfig, ProviderConfig
from model_library.register_models import register_provider


class BasetenConfig(ProviderConfig):
    api_base: str | None = None


@register_provider("baseten")
class BasetenModel(DelegateOnly):
    """OpenAI-compatible Baseten Model APIs and dedicated deployments."""

    provider_config = BasetenConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["baseten"] = "baseten",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        config = config or LLMConfig()
        if not config.custom_endpoint:
            config.custom_endpoint = (
                self.provider_config.api_base
                or model_library_settings.BASETEN_API_BASE_URL
            )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.BASETEN_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
            normalize_null_assistant_history_fields=True,
        )
