"""
See deepseek data retention policy
https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html
"""

from typing import Literal

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("deepseek")
class DeepSeekModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["deepseek"] = "deepseek",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://api-docs.deepseek.com/
        self.init_delegate(
            config=config,
            base_url="https://api.deepseek.com/v1",
            api_key=model_library_settings.DEEPSEEK_API_KEY,
            use_completions=True,
            delegate_provider="openai",
        )
