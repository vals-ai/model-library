from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("poolside")
class PoolsideModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["poolside"] = "poolside",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://inference.poolside.ai/v1/"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.POOLSIDE_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )
