from typing import Literal

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("inception")
class MercuryModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["mercury"] = "mercury",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.inceptionlabs.ai/get-started/get-started#external-libraries-compatibility
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://api.inceptionlabs.ai/v1/",
                api_key=SecretStr(model_library_settings.MERCURY_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )
