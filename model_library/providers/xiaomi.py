from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
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
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://api.xiaomimimo.com/v1",
                api_key=SecretStr(model_library_settings.XIAOMI_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )
