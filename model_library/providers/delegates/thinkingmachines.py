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


class ThinkingMachinesConfig(ProviderConfig):
    """Configuration for Thinking Machines (Tinker) models.

    Attributes:
        separate_reasoning: When true, the server parses out the reasoning
            portion and returns it on a dedicated reasoning_content field
            rather than inlining it into content.
            See: https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
    """

    separate_reasoning: bool = True


@register_provider("thinkingmachines")
class ThinkingMachinesModel(DelegateOnly):
    provider_config = ThinkingMachinesConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["thinkingmachines"] = "thinkingmachines",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint
            or "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.TINKER_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """Build extra body parameters for Tinker-specific features."""
        extra_body: dict[str, Any] = {
            "separate_reasoning": self.provider_config.separate_reasoning
        }
        # Tinker requests that Inkling's numeric effort be sent as a string.
        if isinstance(self.reasoning_effort, str):
            extra_body["reasoning_effort"] = self.reasoning_effort
        return extra_body
