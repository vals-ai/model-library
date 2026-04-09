from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("xiaomi")
class XiaomiModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["xiaomi"] = "xiaomi",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.xiaomimimo.com/#/docs/quick-start/first-api-call
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://api.xiaomimimo.com/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.XIAOMI_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )
