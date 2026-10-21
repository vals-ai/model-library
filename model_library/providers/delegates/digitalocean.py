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
from model_library.base.query_ids import prompt_cache_key_from_query_ids
from model_library.register_models import register_provider
from model_library.registry_utils import compute_model_cost, get_registry_config


class DigitalOceanConfig(ProviderConfig):
    routing_preference: Literal["balanced", "budget", "quality"] | None = None


@register_provider("digitalocean")
class DigitalOceanModel(DelegateOnly):
    api_key_setting = "DIGITAL_OCEAN_API_KEY"
    provider_config = DigitalOceanConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["digitalocean"] = "digitalocean",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint
                or "https://inference.do-ai.run/v1",
                "custom_api_key": config.custom_api_key
                or SecretStr(self._api_key_from(model_library_settings)),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        if preference := self.provider_config.routing_preference:
            return {"routing_preference": preference}
        return {}

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
            kwargs["extra_headers"] = {
                "X-Model-Affinity": prompt_cache_key_from_query_ids(
                    model_name=self._registry_key or self.model_name
                )
            }

        return await super()._query_impl(
            input,
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        if not self.is_router:
            return await super()._calculate_cost(metadata, batch, bill_reasoning)
        if not metadata.served_model:
            return None
        served_key = f"digitalocean/{metadata.served_model}"
        if get_registry_config(served_key) is None:
            return None
        return compute_model_cost(served_key, metadata)
