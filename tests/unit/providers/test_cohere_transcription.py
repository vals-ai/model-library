"""Tests for Cohere complete-file transcription.

Run: uv run pytest tests/unit/providers/test_cohere_transcription.py -m unit

Covers the OpenAI delegate boundary and endpoint configuration.
"""

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.audio.transcription import Transcription
from pydantic import SecretStr

from examples.data.audio import tone_wav
from model_library.base import LLMConfig, TranscriptionMetadata, TranscriptionResult
from model_library.providers.delegates.cohere import CohereModel
from model_library.providers.voice.groq import GroqModel

_AUDIO = tone_wav()
_MODEL_KEY = "cohere/cohere-transcribe-03-2026"
_MODEL_NAME = "cohere-transcribe-03-2026"


def _config(*, endpoint: str | None = None) -> LLMConfig:
    return LLMConfig(
        supports_transcription=True,
        custom_api_key=SecretStr("cohere-key"),
        custom_endpoint=endpoint,
        registry_key=_MODEL_KEY,
    )


async def _transcribe(
    model: CohereModel, *, language: str | None = "en"
) -> TranscriptionResult:
    return await model.transcribe_audio(
        name="clip.wav",
        mime="audio/wav",
        audio=_AUDIO,
        language=language,
    )


async def test_cohere_transcribes_through_openai_delegate() -> None:
    model = CohereModel(_MODEL_NAME, config=_config())
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=Transcription(text="hello world")
    )

    assert model.delegate is not None
    with patch.object(model.delegate, "get_client", return_value=client):
        result = await _transcribe(model)

    assert result.text == "hello world"
    assert result.metadata.audio_bytes == len(_AUDIO)
    assert result.metadata.audio_duration_seconds == pytest.approx(0.75)
    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", _AUDIO, "audio/wav"),
        model=_MODEL_NAME,
        response_format="json",
        language="en",
    )


async def test_delegate_only_transcription_does_not_use_capability_metadata() -> None:
    config = _config()
    config.supports_transcription = False
    model = CohereModel(_MODEL_NAME, config=config)
    delegated_result = TranscriptionResult(
        text="hello world",
        metadata=TranscriptionMetadata(
            audio_bytes=len(_AUDIO),
            request_duration_seconds=0,
        ),
    )

    assert model.delegate is not None
    with patch.object(
        model.delegate,
        "_transcribe_audio",
        new_callable=AsyncMock,
        return_value=delegated_result,
    ) as transcribe:
        result = await _transcribe(model)

    assert result.text == "hello world"
    transcribe.assert_awaited_once()


def _cohere_model(config: LLMConfig) -> CohereModel:
    return CohereModel(_MODEL_NAME, config=config)


def _groq_model(config: LLMConfig) -> GroqModel:
    return GroqModel("whisper-large-v3-turbo", config=config)


@pytest.mark.parametrize(
    ("model_factory", "endpoint", "expected"),
    [
        (
            _cohere_model,
            None,
            "https://api.cohere.ai/compatibility/v1",
        ),
        (
            _cohere_model,
            "https://cohere-proxy.example/compatibility/v1",
            "https://cohere-proxy.example/compatibility/v1",
        ),
        (
            _groq_model,
            None,
            "https://api.groq.com/openai/v1",
        ),
        (
            _groq_model,
            "https://proxy.example/openai/v1/audio/transcriptions",
            "https://proxy.example/openai/v1/audio/transcriptions",
        ),
    ],
)
def test_openai_delegate_uses_expected_endpoint(
    model_factory: Callable[[LLMConfig], CohereModel | GroqModel],
    endpoint: str | None,
    expected: str,
) -> None:
    model = model_factory(
        LLMConfig(
            supports_transcription=True,
            custom_api_key=SecretStr("provider-key"),
            custom_endpoint=endpoint,
            registry_key=_MODEL_KEY,
        )
    )

    assert model.delegate is not None
    assert model.delegate.custom_endpoint == expected
