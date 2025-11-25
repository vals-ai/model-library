from typing import Literal

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


class TogetherConfig(ProviderConfig):
    serverless: bool = True


@register_provider("together")
class TogetherModel(DelegateOnly):
    provider_config = TogetherConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["together"] = "together",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        # https://docs.together.ai/docs/openai-api-compatibility
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.TOGETHER_API_KEY,
                base_url="https://api.together.xyz/v1",
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
        # https://docs.together.ai/docs/dedicated-inference#prompt-caching
        # By default, caching is not enabled. To turn on prompt caching, remove --no-prompt-cache from the create command

        # https://docs.together.ai/docs/inference-faqs#can-i-cache-prompts-or-use-speculative-decoding%3F
        # TODO: Together supports optimizations like prompt caching and speculative decoding for models that allow it, reducing latency and improving throughput.
        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)
