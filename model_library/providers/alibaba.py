from typing import Literal

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


@register_provider("alibaba")
class AlibabaModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["alibaba"] = "alibaba",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.DASHSCOPE_API_KEY,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
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
        """
        Calculate cost for Qwen models using tiered pricing.
        Pricing is based on total input tokens:
        - 0-32k: $1.2/M input, $6/M output
        - 32k-128k: $2.4/M input, $12/M output
        - 128k+: $3/M input, $15/M output
        Cached tokens are billed at 20% of regular cost.
        """
        MILLION = 1_000_000
        CACHE_DISCOUNT = 0.20
        # Calculate total input tokens (including cached tokens)
        total_input_tokens = metadata.in_tokens + (metadata.cache_read_tokens or 0)

        # Determine pricing tier based on total input tokens
        if total_input_tokens <= 32_000:
            input_price = 1.2
            output_price = 6.0
        elif total_input_tokens <= 128_000:
            input_price = 2.4
            output_price = 12.0
        else:
            input_price = 3.0
            output_price = 15.0

        # Calculate cache costs (20% of regular price)
        cache_read_cost = input_price * CACHE_DISCOUNT

        # Calculate actual costs
        return QueryResultCost(
            input=input_price * metadata.in_tokens / MILLION,
            output=output_price * metadata.out_tokens / MILLION,
            reasoning=output_price * metadata.reasoning_tokens / MILLION
            if metadata.reasoning_tokens is not None and bill_reasoning
            else None,
            cache_read=cache_read_cost * metadata.cache_read_tokens / MILLION
            if metadata.cache_read_tokens is not None
            else None,
            cache_write=None,
        )
