import pytest
from pydantic import SecretStr
from typing_extensions import override

from model_library.base import (
    TranscriptionConfig,
    TranscriptionMetadata,
    TranscriptionOnly,
    TranscriptionOnlyException,
    TranscriptionResult,
)


class DummyTranscriptionModel(TranscriptionOnly):
    def __init__(self, *, config: TranscriptionConfig | None = None) -> None:
        self.client_credentials: tuple[str | None, str | None] | None = None
        super().__init__("dummy-model", "dummy", config=config)

    @override
    def _get_default_api_key(self) -> str:
        return "default-key"

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> object:
        if api_key is None:
            return super().get_client()
        self.client_credentials = (api_key, base_url)
        client = object()
        self.assign_client(client)
        return client

    @override
    async def transcribe_audio(
        self,
        *,
        name: str,
        mime: str,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=name,
            metadata=TranscriptionMetadata(
                audio_bytes=len(audio),
                request_duration_seconds=0,
            ),
        )


async def test_transcription_only_model_transcribes_and_rejects_text_generation() -> (
    None
):
    model = DummyTranscriptionModel()

    result = await model.transcribe_audio(
        name="sample.wav", mime="audio/wav", audio=b"audio", language="en"
    )
    assert result.text == "sample.wav"

    with pytest.raises(TranscriptionOnlyException):
        await model.query("hello")
    with pytest.raises(TranscriptionOnlyException):
        await model.parse_tools([])


def test_transcription_config_drives_capabilities_and_client_initialization() -> None:
    model = DummyTranscriptionModel(
        config=TranscriptionConfig(
            custom_api_key=SecretStr("caller-key"),
            custom_endpoint="https://proxy.test/v1",
            registry_key="dummy/dummy-model",
        )
    )

    assert model.client_credentials == ("caller-key", "https://proxy.test/v1")
    assert model.supports_transcription
    assert not model.supports_temperature
    assert not model.supports_tools


def test_transcription_only_model_falls_back_to_the_provider_api_key() -> None:
    model = DummyTranscriptionModel()

    assert model.client_credentials == ("default-key", None)
