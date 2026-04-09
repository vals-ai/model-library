from typing import Any, Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("openrouter")
class OpenRouterModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["openrouter"] = "openrouter",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://openrouter.ai/docs/guides/community/openai-sdk
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://openrouter.ai/api/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.OPENROUTER_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        if self.reasoning:
            return {"reasoning": {"enabled": True}}
        return {}
