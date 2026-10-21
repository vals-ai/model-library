from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("inception")
class MercuryModel(DelegateOnly):
    api_key_setting = "MERCURY_API_KEY"

    def __init__(
        self,
        model_name: str,
        provider: Literal["mercury"] = "mercury",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.inceptionlabs.ai/get-started/get-started#external-libraries-compatibility
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://api.inceptionlabs.ai/v1/"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            self._api_key_from(model_library_settings)
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )
