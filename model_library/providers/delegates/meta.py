import io
from typing import Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    FileWithId,
    LLMConfig,
    ProviderConfig,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.providers.openai import OpenAIConfig
from model_library.register_models import register_provider


class MetaConfig(ProviderConfig):
    use_responses: bool = False
    prompt_cache_key: PromptCacheKeyMode | None = None


@register_provider("meta")
class MetaModel(DelegateOnly):
    provider_config = MetaConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["meta"] = "meta",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        base_url = "https://api.meta.ai/v1"

        # https://docs.llama.com
        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint or base_url,
                "custom_api_key": config.custom_api_key
                or SecretStr(model_library_settings.META_API_KEY),
                "provider_config": OpenAIConfig(
                    prompt_cache_key=self.provider_config.prompt_cache_key,
                ),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=not self.provider_config.use_responses,
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
        # Meta only accepts the "user_data" purpose, so the delegate's own
        # upload_file ("assistants") cannot be reused.
        response = await self.delegate.get_client().files.create(
            file=(name, bytes, mime),
            purpose="user_data",
        )

        return FileWithId(
            type=type,
            name=response.filename,
            mime=mime,
            file_id=response.id,
        )
