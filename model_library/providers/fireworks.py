from typing import Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.register_models import register_provider


class FireworksConfig(ProviderConfig):
    serverless: bool = True


@register_provider("fireworks")
class FireworksModel(DelegateOnly):
    provider_config = FireworksConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["fireworks"] = "fireworks",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        if self.provider_config.serverless:
            self.model_name = "accounts/fireworks/models/" + self.model_name
        else:
            self.model_name = "accounts/rayan-936e28/deployedModels/" + self.model_name

        # https://docs.fireworks.ai/tools-sdks/openai-compatibility
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=SecretStr(model_library_settings.FIREWORKS_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
        )

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        # https://docs.fireworks.ai/guides/prompt-caching
        # Prompt caching is enabled by default for all Fireworks models and deployments.

        # Discounts for prompt caching are available for enterprise deployments. Contact us to learn more.

        # https://docs.fireworks.ai/faq-new/billing-pricing/is-prompt-caching-billed-differently
        # prompt caching does not affect billing for serverless models

        return await super()._calculate_cost(
            metadata, batch, bill_reasoning=bill_reasoning
        )
