from typing import Literal

from model_library import model_library_settings
from model_library.base import DelegateOnly, LLMConfig
from model_library.providers.anthropic import AnthropicModel
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client

from anthropic import AsyncAnthropic


@register_provider("minimax")
class MinimaxModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["minimax"] = "minimax",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.delegate = AnthropicModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=AsyncAnthropic(
                api_key=model_library_settings.MINIMAX_API_KEY,
                base_url="https://api.minimax.io/anthropic",
                http_client=default_httpx_client(),
                max_retries=1,
            ),
        )
