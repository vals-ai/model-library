from typing import Any, Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
    ProviderConfig,
)
from model_library.register_models import register_provider

# https://docs.z.ai/guides/develop/openai/python
_INTERNATIONAL_ENDPOINT = "https://api.z.ai/api/paas/v4/"
_MAINLAND_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/"


class ZAIConfig(ProviderConfig):
    """Configuration for ZAI (GLM) models.

    Attributes:
        clear_thinking: When disabled, reasoning content from previous turns is
            preserved in context. This is useful for multi-turn conversations where
            you want the model to maintain coherent reasoning across turns.
            Enabled by default on the standard API endpoint.
            See: https://docs.z.ai/guides/capabilities/thinking-mode
        supports_disabling_thinking: Whether the model accepts
            ``thinking.type: "disabled"``. Newer GLM models reject the request
            when thinking is disabled.
        international: Route to the api.z.ai endpoint instead of the mainland
            ZhipuAI endpoint at open.bigmodel.cn.
    """

    clear_thinking: bool = True
    supports_disabling_thinking: bool = True
    international: bool = False


@register_provider("zai")
class ZAIModel(DelegateOnly):
    provider_config = ZAIConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["zai"] = "zai",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.clear_thinking = self.provider_config.clear_thinking
        self.supports_disabling_thinking = (
            self.provider_config.supports_disabling_thinking
        )

        config = config or LLMConfig()
        if config.custom_api_key is None:
            config.custom_endpoint = config.custom_endpoint or self._default_endpoint()
            config.custom_api_key = SecretStr(self._default_api_key())
        else:
            config.custom_endpoint = config.custom_endpoint or self._public_endpoint()

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    def _default_endpoint(self) -> str:
        return self._public_endpoint()

    def _public_endpoint(self) -> str:
        if self.provider_config.international:
            return _INTERNATIONAL_ENDPOINT
        return _MAINLAND_ENDPOINT

    def _default_api_key(self) -> str:
        return model_library_settings.ZAI_API_KEY

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """Build extra body parameters for GLM-specific features."""
        thinking_enabled = self.reasoning or not self.supports_disabling_thinking
        return {
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled",
                "clear_thinking": self.clear_thinking,
            }
        }
