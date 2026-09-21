from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import LLMConfig, TranscriptionOnly, TranscriptionResult
from model_library.providers.openai import OpenAIModel
from model_library.base.transcription import (
    TranscriptionRequest,
)
from model_library.register_models import register_provider

_GROQ_OPENAI_COMPATIBLE_ENDPOINT = "https://api.groq.com/openai/v1"


@register_provider("groq")
class GroqModel(TranscriptionOnly):
    """Complete-file transcription through Groq's OpenAI-compatible endpoint."""

    provider_name = "groq"

    def __init__(
        self,
        model_name: str,
        provider: str | None = None,
        *,
        config: LLMConfig | None = None,
    ) -> None:
        super().__init__(model_name, provider, config=config)

        self._openai_delegate: OpenAIModel = OpenAIModel(
            model_name,
            self.provider,
            config=LLMConfig(
                custom_api_key=SecretStr(self._api_key()),
                custom_endpoint=self.custom_endpoint
                or _GROQ_OPENAI_COMPATIBLE_ENDPOINT,
            ),
        )
        self.delegate = self._openai_delegate

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.GROQ_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return await self._openai_delegate.transcribe_file_audio(
            name=request.name,
            mime=request.mime,
            audio=request.audio,
            language=request.language,
        )
