import io
from typing import Literal, Sequence

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    FileBase,
    FileWithId,
    InputItem,
    LLMConfig,
    ProviderConfig,
    ToolDefinition,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.providers.openai import OpenAIConfig
from model_library.rate_limits import RateLimit
from model_library.rate_limits.probe import probe_chat_completions_rate_limit
from model_library.register_models import register_provider

_BASE_URL = "https://api.x.ai/v1"


class XAIConfig(ProviderConfig):
    # A stable conversation key improves xAI prompt-cache hits.
    # https://docs.x.ai/developers/advanced-api-usage/prompt-caching/best-practices
    prompt_cache_key: PromptCacheKeyMode = "hash"


@register_provider("grok")
class XAIModel(DelegateOnly):
    api_key_setting = "XAI_API_KEY"

    provider_config = XAIConfig()

    @override
    def _get_default_api_key(self) -> str:
        return self._api_key_from(model_library_settings)

    def __init__(
        self,
        model_name: str,
        provider: Literal["xai"] = "xai",
        *,
        config: LLMConfig | None = None,
    ):
        config = config or LLMConfig()
        super().__init__(model_name, provider, config=config)
        self._custom_api_key = config.custom_api_key

        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint or _BASE_URL,
                "custom_api_key": config.custom_api_key
                or SecretStr(self._get_default_api_key()),
                "provider_config": OpenAIConfig(
                    prompt_cache_key=self.provider_config.prompt_cache_key,
                ),
            }
        )
        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=False,
        )

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        assert self.delegate
        return await self.delegate.upload_file(name, mime, bytes, type=type)

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        if self.custom_endpoint is not None:
            return None

        return await probe_chat_completions_rate_limit(
            base_url=_BASE_URL,
            api_key=(
                self._custom_api_key.get_secret_value()
                if self._custom_api_key
                else self._get_default_api_key()
            ),
            model_name=self.model_name,
        )

    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        # Exclude files: their serialized form does not reflect content token usage.
        return await super().count_tokens(
            [item for item in input if not isinstance(item, FileBase)],
            history=[item for item in history if not isinstance(item, FileBase)],
            tools=tools,
            **kwargs,
        )
