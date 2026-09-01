import asyncio
import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    FileWithId,
    InputItem,
    LLMConfig,
    ProviderConfig,
    RawInput,
    QueryResult,
    ToolDefinition,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
    rate_limit_timestamp_from_headers,
)
from model_library.providers.openai import OpenAIConfig
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client

_KIMI_K3_PUBLIC_MODEL = "kimi-k3"
_DEFAULT_ENDPOINT = "https://api.moonshot.ai/v1/"


class KimiConfig(ProviderConfig):
    parallel_tool_calls: bool | None = None
    prompt_cache_key: PromptCacheKeyMode | None = None
    thinking_keep: Literal["all"] | None = None
    thinking_effort: Literal["max"] | None = None


@register_provider("kimi")
class KimiModel(DelegateOnly):
    provider_config = KimiConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["kimi"] = "kimi",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility
        config = config or LLMConfig()
        default_api_key = SecretStr(self._default_api_key())
        resolved_endpoint = config.custom_endpoint or _DEFAULT_ENDPOINT
        resolved_api_key = config.custom_api_key or default_api_key
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": resolved_endpoint,
                "custom_api_key": resolved_api_key,
                "provider_config": OpenAIConfig(
                    parallel_tool_calls=self.provider_config.parallel_tool_calls,
                    prompt_cache_key=self.provider_config.prompt_cache_key,
                ),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=True,
        )

    def _default_api_key(self) -> str:
        api_key = model_library_settings.KIMI_API_KEY
        return api_key

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        if self._has_custom_connection:
            return None

        async with default_httpx_client() as client:
            response = await client.get(
                f"{_DEFAULT_ENDPOINT.rstrip('/')}/users/me",
                headers={"Authorization": f"Bearer {self._default_api_key()}"},
            )
            response.raise_for_status()
            payload = response.json()

            organization = payload["data"]["organization"]
            requests: list[RequestRateLimit] = []
            if (
                request_limit := organization.get("max_request_per_minute")
            ) is not None:
                requests.append(RequestRateLimit(limit=request_limit))
            if (concurrency := organization.get("max_concurrency")) is not None:
                requests.append(RequestRateLimit(limit=concurrency, mode="concurrency"))
            token_limit = organization.get("max_token_per_minute")
            if not requests and token_limit is None:
                return None
            return RateLimit(
                requests=tuple(requests),
                tokens=(
                    TokenRateLimit(total=RateLimitCapacity(limit=token_limit))
                    if token_limit is not None
                    else None
                ),
                scope="shared",
                unix_timestamp=rate_limit_timestamp_from_headers(response.headers),
            )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """
        Build extra body parameters for Kimi-specific features.
        see https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart#parameters-differences-in-request-body
        """
        if self.model_name == _KIMI_K3_PUBLIC_MODEL:
            return {}

        thinking: dict[str, str] = {"type": "enabled" if self.reasoning else "disabled"}
        if self.reasoning and self.provider_config.thinking_keep is not None:
            thinking["keep"] = self.provider_config.thinking_keep
        if self.reasoning and self.provider_config.thinking_effort is not None:
            thinking["effort"] = self.provider_config.thinking_effort
        return {"thinking": thinking}

    async def _preprocess_files(self, input: Sequence[InputItem]) -> list[InputItem]:
        """Replace file items with TextInput containing extracted text.

        Moonshot doesn't support OpenAI-style file content blocks. Files are
        uploaded via the files API, text is extracted, and injected as plain
        text so the delegate never sees file items.
        """

        async def preprocess_item(item: InputItem) -> InputItem:
            if not isinstance(item, FileWithId):
                return item

            assert self.delegate
            response = await self.delegate.get_client().files.content(
                file_id=item.file_id
            )
            return RawInput(
                input={
                    "role": "system",
                    "content": response.text,
                }
            )

        tasks = [asyncio.create_task(preprocess_item(item)) for item in input]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        preprocessed = await self._preprocess_files(input)
        return await super().delegate_query(
            preprocessed,
            tools=tools,
            query_logger=query_logger,
            extra_body=self._get_extra_body(),
            output_schema=output_schema,
            **kwargs,
        )
