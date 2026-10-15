import logging
from typing import Any, Literal, Sequence, cast

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
from model_library.exceptions import (
    NoRetryException,
    ProviderErrorPayload,
    exception_http_status_code,
    exception_to_provider_error,
)
from model_library.providers.openai import OpenAIConfig
from model_library.register_models import register_provider


class RampNoRetryError(NoRetryException):
    """A Ramp error that never succeeds on retry; reports the original provider error."""

    def __init__(self, original: Exception):
        self.original = original
        super().__init__(str(original))

    def to_provider_error(self) -> ProviderErrorPayload:
        return exception_to_provider_error(self.original)


class RampConfig(ProviderConfig):
    # Switchyard only routes requests that name a known coding-agent client and
    # carry its session id; plain API requests always get the requested model.
    switchyard_client: Literal["claude-code"] | None = None


SWITCHYARD_SESSION_HEADERS = {"claude-code": "X-Claude-Code-Session-Id"}


@register_provider("ramp")
class RampModel(DelegateOnly):
    provider_config = RampConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["ramp"] = "ramp",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.router.com/api/endpoint
        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint
                or "https://api.router.com/v1",
                "custom_api_key": config.custom_api_key
                or SecretStr(model_library_settings.RAMP_API_KEY),
                # Ramp routes on prompt_cache_key: one key per task attempt, so the
                # served model sticks within it.
                "provider_config": OpenAIConfig(prompt_cache_key="id"),
            }
        )

        # Ramp has no Chat Completions surface.
        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=False,
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
        self.enable_router_mode()
        if client := self.provider_config.switchyard_client:
            # Per registry key, so a variant never shares a session with its pinned entry.
            session_id = prompt_cache_key_from_query_ids(
                model_name=self._registry_key or f"ramp/{self.model_name}"
            )
            kwargs["prompt_cache_key"] = session_id
            kwargs["extra_headers"] = {
                **cast(dict[str, str], kwargs.get("extra_headers") or {}),
                "X-Gateway-Client": client,
                SWITCHYARD_SESSION_HEADERS[client]: session_id,
            }
        try:
            return await super()._query_impl(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )
        except Exception as e:
            # Out of credits (402, whose message says "retry") and unsupported
            # features (501) never succeed on retry.
            if exception_http_status_code(e) in (402, 501):
                raise RampNoRetryError(e) from e
            raise
