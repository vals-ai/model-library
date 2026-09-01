from typing import Any, Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
    rate_limit_timestamp_from_headers,
)
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client


# https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
_INTERNATIONAL_ENDPOINT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_MAINLAND_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_INTERNATIONAL_QUOTA_ENDPOINT = "https://dashscope-intl.aliyuncs.com/api/v1/quotas"
_MAINLAND_QUOTA_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/quotas"


class AlibabaConfig(ProviderConfig):
    """Configuration for Alibaba (Qwen) models.

    Attributes:
        preserve_thinking: When enabled, previous reasoning content is preserved
            in context across turns instead of being stripped and re-serialized.
            Supported by Qwen 3.6+ reasoning models. This improves KV cache utilization
            and decision consistency in agentic workflows.
            See: https://qwen.ai/blog?id=qwen3.6-27b
        mainland: Route to the mainland China DashScope endpoint, authenticated
            with `DASHSCOPE_CN_API_KEY`. Model Studio accounts and keys are
            region-scoped: an international key is rejected by the mainland
            endpoint and vice versa.
    """

    preserve_thinking: bool = False
    mainland: bool = False


@register_provider("alibaba")
class AlibabaModel(DelegateOnly):
    provider_config = AlibabaConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["alibaba"] = "alibaba",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.preserve_thinking = self.provider_config.preserve_thinking

        config = config or LLMConfig()
        config.custom_endpoint = config.custom_endpoint or (
            _MAINLAND_ENDPOINT
            if self.provider_config.mainland
            else _INTERNATIONAL_ENDPOINT
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            self._default_api_key()
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
            normalize_null_assistant_history_fields=True,
        )

    def _default_api_key(self) -> str:
        if self.provider_config.mainland:
            return model_library_settings.DASHSCOPE_CN_API_KEY
        return model_library_settings.DASHSCOPE_API_KEY

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        if self._has_custom_connection:
            return None

        def per_minute(limit: int, period_seconds: int) -> int:
            # This exposes per-minute throughput; shorter provider bursts remain.
            return limit * 60 // period_seconds

        quota_endpoint = (
            _MAINLAND_QUOTA_ENDPOINT
            if self.provider_config.mainland
            else _INTERNATIONAL_QUOTA_ENDPOINT
        )
        async with default_httpx_client() as client:
            response = await client.get(
                quota_endpoint,
                headers={"Authorization": f"Bearer {self._default_api_key()}"},
                params={"model": self.model_name},
            )
            response.raise_for_status()
            payload = response.json()

            request_limits: list[int] = []
            token_limits: list[int] = []
            for quota in payload["output"]["quotas"]:
                if quota["model"] != self.model_name:
                    continue
                model_limit = quota["model_limit"]
                request_limit = model_limit.get("request_limit")
                request_limit_period = model_limit.get("request_limit_period")
                if request_limit is not None and request_limit_period is not None:
                    request_limits.append(
                        per_minute(request_limit, request_limit_period)
                    )
                usage_limit = model_limit.get("usage_limit")
                usage_limit_period = model_limit.get("usage_limit_period")
                if (
                    model_limit.get("usage_limit_field") == "total_tokens"
                    and usage_limit is not None
                    and usage_limit_period is not None
                ):
                    token_limits.append(per_minute(usage_limit, usage_limit_period))

            if not request_limits and not token_limits:
                return None
            return RateLimit(
                requests=(
                    (RequestRateLimit(limit=min(request_limits)),)
                    if request_limits
                    else ()
                ),
                tokens=(
                    TokenRateLimit(total=RateLimitCapacity(limit=min(token_limits)))
                    if token_limits
                    else None
                ),
                scope="api_key",
                unix_timestamp=rate_limit_timestamp_from_headers(response.headers),
            )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """Build extra body parameters for Qwen-specific features."""
        extra: dict[str, Any] = {}
        # Enable thinking mode for Qwen3 reasoning models
        # https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api
        if self.reasoning:
            extra["enable_thinking"] = True
            if self.preserve_thinking:
                extra["preserve_thinking"] = True
        return extra

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        # qwen3-max and qwen3-vl-plus use hardcoded tiered pricing
        if "qwen3-max" not in self.model_name and "qwen3-vl" not in self.model_name:
            return await super()._calculate_cost(metadata, batch, bill_reasoning)

        # Hardcoded tiered pricing for qwen3-max models
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
