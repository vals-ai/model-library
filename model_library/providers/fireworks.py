from typing import Literal

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.base.delegate_only import DelegateOnly
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


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
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
            ),
            use_completions=True,
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
