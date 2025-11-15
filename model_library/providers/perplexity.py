from typing import Literal

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


@register_provider("perplexity")
class PerplexityModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["perplexity"] = "perplexity",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.perplexity.ai/guides/chat-completions-guide
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.PERPLEXITY_API_KEY,
                base_url="https://api.perplexity.ai",
            ),
            use_completions=True,
        )
