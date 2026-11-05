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
    ToolDefinition,
)
from model_library.base.router import RouterLLM
from model_library.register_models import register_provider
from model_library.registry_utils import get_registry_config


class DigitalOceanConfig(ProviderConfig):
    routing_preference: Literal["balanced", "budget", "quality"] | None = None


@register_provider("digitalocean")
class DigitalOceanModel(DelegateOnly, RouterLLM):
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
                "X-Model-Affinity": self.router_session_id(input, tools)
            }

        return await super()._query_impl(
            input,
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    @override
    def _served_model_cost_key(self, served_model: str) -> str | None:
        key = f"digitalocean/{served_model}"
        return key if get_registry_config(key) else None
