import logging
from typing import Any, Literal, Sequence

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    InputItem,
    LLMConfig,
    ProviderConfig,
    QueryResult,
    ToolDefinition,
)
from model_library.register_models import register_provider


class ZAIConfig(ProviderConfig):
    """Configuration for ZAI (GLM) models.

    Attributes:
        clear_thinking: When disabled, reasoning content from previous turns is
            preserved in context. This is useful for multi-turn conversations where
            you want the model to maintain coherent reasoning across turns.
            Enabled by default on the standard API endpoint.
            See: https://docs.z.ai/guides/capabilities/thinking-mode
    """

    clear_thinking: bool = True


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

        # https://docs.z.ai/guides/develop/openai/python
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                api_key=SecretStr(model_library_settings.ZAI_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )

    def get_extra_body(self) -> dict[str, Any]:
        """Build extra body parameters for GLM-specific features."""
        return {
            "thinking": {
                "type": "enabled" if self.reasoning else "disabled",
                "clear_thinking": self.clear_thinking,
            }
        }

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        **kwargs: object,
    ) -> QueryResult:
        return await self.delegate_query(
            input,
            tools=tools,
            query_logger=query_logger,
            extra_body=self.get_extra_body(),
            **kwargs,
        )
