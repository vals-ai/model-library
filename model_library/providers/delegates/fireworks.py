import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    ProviderConfig,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    ToolDefinition,
)
from model_library.rate_limits import RateLimit, RateLimitCapacity, TokenRateLimit
from model_library.register_models import register_provider

FIREROUTER_ENDPOINT_PREFIX = "accounts/fireworks/routers/"


class FireworksConfig(ProviderConfig):
    serverless: bool = True
    firerouter_routing_preference: int | None = None


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

        is_firerouter = self.model_name.startswith(FIREROUTER_ENDPOINT_PREFIX)
        if not is_firerouter:
            if self.provider_config.serverless:
                self.model_name = "accounts/fireworks/models/" + self.model_name
            else:
                self.model_name = (
                    "accounts/rayan-936e28/deployedModels/" + self.model_name
                )

        # https://docs.fireworks.ai/tools-sdks/openai-compatibility
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://api.fireworks.ai/inference/v1"
        )
        # FireRouter's closed (Claude/GPT) legs use Provider Keys stored on the
        # Nexus account; the regular key would silently serve open models only.
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.FIREWORKS_NEXUS_API_KEY
            if is_firerouter
            else model_library_settings.FIREWORKS_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
            normalize_null_assistant_history_fields=True,
        )

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if self.is_router:
            self.enable_router_mode()
            preference = self.provider_config.firerouter_routing_preference
            if preference is not None:
                kwargs["extra_headers"] = {"x-routing-preference": str(preference)}
        return await super()._query_impl(
            input,
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        assert self.delegate
        rate_limit = await self.delegate.get_rate_limit()
        if rate_limit is None:
            return None

        tokens = rate_limit.tokens
        if tokens is None or tokens.input is None or tokens.output is None:
            return rate_limit

        def per_minute(
            capacity: RateLimitCapacity | None,
        ) -> RateLimitCapacity | None:
            if capacity is None:
                return None
            return RateLimitCapacity(
                limit=capacity.limit * 60,
                remaining=(
                    capacity.remaining * 60 if capacity.remaining is not None else None
                ),
            )

        return RateLimit(
            requests=rate_limit.requests,
            tokens=TokenRateLimit(
                input=per_minute(tokens.input),
                uncached_input=per_minute(tokens.uncached_input),
                output=per_minute(tokens.output),
            ),
            scope=rate_limit.scope,
            unix_timestamp=rate_limit.unix_timestamp,
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
            metadata,
            batch,
            bill_reasoning=bill_reasoning,
        )
