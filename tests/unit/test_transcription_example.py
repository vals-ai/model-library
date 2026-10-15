from unittest.mock import AsyncMock, patch

import pytest

from model_library.base.output.transcription import (
    TranscriptionMetadata,
    TranscriptionResult,
)
from model_library.providers.openai import OpenAIModel
from examples.extras.transcription import main


@pytest.mark.asyncio
async def test_default_transcription_example_reaches_audio_provider() -> None:
    result = TranscriptionResult(
        text="The quick brown fox.",
        metadata=TranscriptionMetadata(audio_bytes=42, request_duration_seconds=0.1),
    )
    with (
        patch("sys.argv", ["transcription.py"]),
        patch.object(OpenAIModel, "_transcribe_audio", new_callable=AsyncMock) as provider,
        patch("examples.extras.transcription.console_log") as console_log,
    ):
        provider.return_value = result

        await main()

    provider.assert_awaited_once()
    request = provider.call_args.args[0]
    assert request.audio.startswith(b"RIFF")
    assert request.language == "en"
    console_log.assert_any_call("Text: The quick brown fox.")
