import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from typing_extensions import override

import model_library.base.base as base_module
from model_library.base import (
    LLMConfig,
    TranscriptionMetadata,
    TranscriptionOnly,
    TranscriptionOnlyException,
    TranscriptionResult,
)
from model_library.base.transcription import TranscriptionRequest


class DummyTranscriptionModel(TranscriptionOnly):
    provider_name = "dummy"

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__("dummy-model", config=config)

    @override
    def _get_default_api_key(self) -> str:
        return "test-key"

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=request.name,
            metadata=TranscriptionMetadata(
                audio_bytes=len(request.audio),
                request_duration_seconds=0,
            ),
        )


class SuperCallingTranscriptionModel(TranscriptionOnly):
    provider_name = "super-calling"

    @override
    def _get_default_api_key(self) -> str:
        return "test-key"

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    @override
    async def transcribe_audio(
        self,
        *,
        name: str,
        mime: str,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        return await super().transcribe_audio(
            name=name,
            mime=mime,
            audio=audio,
            language=language,
        )


class MissingProviderTranscriptionModel(TranscriptionOnly):
    @override
    def _get_default_api_key(self) -> str:
        return "test-key"

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=request.name,
            metadata=TranscriptionMetadata(
                audio_bytes=len(request.audio),
                request_duration_seconds=0,
            ),
        )


async def test_public_override_calling_super_does_not_recurse() -> None:
    model = SuperCallingTranscriptionModel("super-calling-model")

    with pytest.raises(NotImplementedError, match="not supported"):
        await asyncio.wait_for(
            model.transcribe_audio(
                name="clip.wav",
                mime="audio/wav",
                audio=b"audio",
                language="en",
            ),
            timeout=1,
        )


async def test_transcription_only_model_rejects_text_generation() -> None:
    model = DummyTranscriptionModel()

    with pytest.raises(TranscriptionOnlyException):
        await model.query("hello")
    with pytest.raises(TranscriptionOnlyException):
        await model.parse_tools([])


def test_transcription_only_defaults_capabilities_and_provider() -> None:
    model = DummyTranscriptionModel()

    assert model.provider == "dummy"
    assert model.supports_transcription is True
    assert model.supports_temperature is False


def test_transcription_only_preserves_explicit_capabilities() -> None:
    model = DummyTranscriptionModel(
        LLMConfig(
            supports_transcription=False,
            supports_temperature=True,
        )
    )

    assert model.supports_transcription is False
    assert model.supports_temperature is True


def test_transcription_only_requires_provider_identity() -> None:
    with pytest.raises(AttributeError, match="provider_name"):
        MissingProviderTranscriptionModel("missing-provider")


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected"),
    [(12, None, 12), (None, 3, 3), (None, None, None)],
)
def test_transcription_total_tokens_handles_partial_usage(
    input_tokens: int | None,
    output_tokens: int | None,
    expected: int | None,
) -> None:
    metadata = TranscriptionMetadata(
        audio_bytes=1,
        request_duration_seconds=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    assert metadata.total_tokens == expected


async def test_transcribe_audio_uses_public_request_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moments = iter((10.0, 12.5))
    monkeypatch.setattr(base_module, "perf_counter", lambda: next(moments))
    model = DummyTranscriptionModel(
        LLMConfig(registry_key="azure_speech/universal-language-model")
    )

    result = await model.transcribe_audio(
        name="clip.wav",
        mime="audio/mpeg",
        audio=b"audio",
        language="en-US",
    )

    assert result.metadata.request_duration_seconds == 2.5


async def test_transcribe_audio_unwraps_single_nested_exception() -> None:
    model = DummyTranscriptionModel(
        LLMConfig(registry_key="azure_speech/universal-language-model")
    )
    error = ValueError("provider rejected request")
    group = ExceptionGroup("outer", [ExceptionGroup("inner", [error])])

    with (
        patch.object(model, "_transcribe_audio", new=AsyncMock(side_effect=group)),
        pytest.raises(ValueError, match="provider rejected request") as raised,
    ):
        await model.transcribe_audio(
            name="clip.wav",
            mime="audio/mpeg",
            audio=b"audio",
        )

    assert raised.value is error
    assert raised.value.__cause__ is group


async def test_transcribe_audio_preserves_multiple_exceptions() -> None:
    model = DummyTranscriptionModel(
        LLMConfig(registry_key="azure_speech/universal-language-model")
    )
    group = ExceptionGroup(
        "provider and transport failed",
        [ValueError("provider rejected request"), ConnectionError("socket closed")],
    )

    with (
        patch.object(model, "_transcribe_audio", new=AsyncMock(side_effect=group)),
        pytest.raises(ExceptionGroup) as raised,
    ):
        await model.transcribe_audio(
            name="clip.wav",
            mime="audio/mpeg",
            audio=b"audio",
        )

    assert raised.value is group
