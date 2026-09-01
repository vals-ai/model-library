import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    QueryResult,
    ToolDefinition,
)
from model_library.exceptions import (
    MaxContextWindowExceededError,
    exception_http_status_code,
)
from model_library.register_models import register_provider


@register_provider("thomsonreuters")
class ThomsonReutersModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["thomsonreuters"] = "thomsonreuters",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # OpenAI-compatible Chat Completions endpoint on Thomson Reuters' Mariner
        # proxy, overridable via THOMSONREUTERS_API_BASE_URL for other Mariner
        # deployments. Both settings are injected from secret fields that can
        # exist but be blank, so blank has to be treated as absent: an empty URL
        # would route nowhere, and an empty key is falsy downstream, which would
        # fall through to the OpenAI delegate's OPENAI_API_KEY default and send
        # an OpenAI key to this host.
        config = config or LLMConfig()
        # No public Mariner endpoint: without an override there is nothing to call.
        default_base_url = ""
        base_url_override = model_library_settings.get(
            "THOMSONREUTERS_API_BASE_URL", ""
        ).strip()
        config.custom_endpoint = (
            config.custom_endpoint or base_url_override or default_base_url
        )
        if not config.custom_api_key:
            api_key = model_library_settings.get("THOMSONREUTERS_API_KEY", "").strip()
            if not api_key:
                # Same failure as an unset key, which ModelLibrarySettings raises.
                raise AttributeError("Missing config key: THOMSONREUTERS_API_KEY")
            config.custom_api_key = SecretStr(api_key)

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
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
        try:
            return await super()._query_impl(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )
        except Exception as e:
            # Mariner rejects a prompt plus max output tokens over the context window
            # with a reasonless 400, the only 400 seen from this provider, so all of
            # them are surfaced as context overflow to let callers shorten history.
            if exception_http_status_code(e) == 400:
                raise MaxContextWindowExceededError(str(e)) from e
            raise
