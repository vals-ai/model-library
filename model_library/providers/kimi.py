from typing import Any, Literal

from typing_extensions import override

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


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
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://api.moonshot.ai/v1/",
                api_key=SecretStr(model_library_settings.KIMI_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """
        Build extra body parameters for Kimi-specific features.
        see https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart#parameters-differences-in-request-body
        """
        return {"thinking": {"type": "enabled" if self.reasoning else "disabled"}}
