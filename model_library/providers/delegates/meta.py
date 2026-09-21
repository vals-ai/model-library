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
    TranscriptionResult,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.base.transcription import TranscriptionRequest
from model_library.providers.openai import OpenAIConfig
from model_library.providers.voice.meta import transcribe_realtime
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
        self._api_key = config.custom_api_key or SecretStr(
            model_library_settings.META_API_KEY
        )
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint or base_url,
                "custom_api_key": self._api_key,
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
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        if self.custom_endpoint is not None:
            raise ValueError("custom_endpoint is not supported for Meta transcription")
        return await transcribe_realtime(
            api_key=self._api_key.get_secret_value(),
            model_name=self.model_name,
            request=request,
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
