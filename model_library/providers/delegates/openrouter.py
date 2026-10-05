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
from model_library.base.query_ids import prompt_cache_key_from_query_ids
from model_library.register_models import register_provider


class OpenRouterConfig(ProviderConfig):
    openrouter_allowed_models: list[str] | None = None


@register_provider("openrouter")
class OpenRouterModel(DelegateOnly):
    provider_config = OpenRouterConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["openrouter"] = "openrouter",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://openrouter.ai/docs/guides/community/openai-sdk
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://openrouter.ai/api/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.OPENROUTER_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        if self.reasoning:
            return {"reasoning": {"enabled": True}}
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
        assert self.delegate
        extra_body = self._get_extra_body()
        if self.is_router:
            self.enable_router_mode()
            allowed_models = self.provider_config.openrouter_allowed_models
            if allowed_models is not None:
                extra_body["plugins"] = [
                    {"id": "auto-router", "allowed_models": allowed_models}
                ]
            # One session per task attempt, so the served model sticks within it.
            extra_body["session_id"] = prompt_cache_key_from_query_ids(
                model_name=self._registry_key or f"openrouter/{self.model_name}"
            )
            # Auto may pick a host that silently ignores response_format or tools;
            # require_parameters makes it pick one that supports them (or fail).
            if output_schema is not None or tools:
                extra_body["provider"] = {"require_parameters": True}

        return await self.delegate_query(
            input,
            tools=tools,
            query_logger=query_logger,
            extra_body=extra_body,
            output_schema=output_schema,
            **kwargs,
        )
