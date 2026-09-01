from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.audio.transcription import Transcription, UsageTokens

from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_config


async def test_openai_transcription_normalizes_text_and_usage() -> None:
    response = Transcription(
        text="hello world",
        usage=UsageTokens(
            type="tokens",
            input_tokens=12,
            output_tokens=3,
            total_tokens=15,
            input_token_details={"audio_tokens": 10, "text_tokens": 2},
        ),
    )
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=response)
    model = OpenAIModel("gpt-4o-transcribe")
    model._metadata = get_registry_config(  # pyright: ignore[reportPrivateUsage]
        "openai/gpt-4o-transcribe"
    )

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav",
            mime="audio/wav",
            audio=b"RIFF-test",
            language="en",
        )

    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", b"RIFF-test", "audio/wav"),
        model="gpt-4o-transcribe",
        language="en",
        response_format="json",
    )
    assert result.text == "hello world"
    assert result.metadata.audio_bytes == 9
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 3
    assert result.metadata.total_tokens == 15
    assert result.metadata.audio_tokens == 10
    assert result.metadata.text_tokens == 2
    assert result.metadata.cost_usd == 0.00006
    assert result.metadata.request_duration_seconds >= 0


async def test_openai_transcription_omits_unspecified_language() -> None:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=Transcription(text="hello")
    )
    model = OpenAIModel("gpt-4o-transcribe")

    with patch.object(model, "get_client", return_value=client):
        await model.transcribe_audio(
            name="clip.wav",
            mime="audio/wav",
            audio=b"RIFF-test",
            language=None,
        )

    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", b"RIFF-test", "audio/wav"),
        model="gpt-4o-transcribe",
        response_format="json",
    )
