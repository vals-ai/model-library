from typing import Literal

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


@register_provider("kimi")
class KimiModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["kimi"] = "kimi",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.KIMI_API_KEY,
                base_url="https://api.moonshot.ai/v1/",
            ),
            use_completions=True,
        )
