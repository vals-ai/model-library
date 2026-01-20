from typing import Literal

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("cohere")
class CohereModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["cohere"] = "cohere",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.cohere.com/docs/compatibility-api
        self.init_delegate(
            config=config,
            base_url="https://api.cohere.ai/compatibility/v1",
            api_key=model_library_settings.COHERE_API_KEY,
            use_completions=True,
            delegate_provider="openai",
        )
