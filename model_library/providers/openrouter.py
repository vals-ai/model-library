from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("openrouter")
class OpenRouterModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["openrouter"] = "openrouter",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://openrouter.ai/docs/guides/community/openai-sdk
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key=SecretStr(model_library_settings.OPENROUTER_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )
