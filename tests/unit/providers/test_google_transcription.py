import asyncio
import io
from collections.abc import AsyncIterator
import wave
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from google.genai import Client
from google.genai.types import LiveServerMessage
from model_library.providers.google.transcribe import (
    transcribe_interactions,
    transcribe_live,
)
from model_library.base.transcription import TranscriptionRequest
from tests.unit.providers.transcription_test_support import Connection


def _audio() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 3200)
    return output.getvalue()


def _request() -> TranscriptionRequest:
    return TranscriptionRequest(
        name="clip.wav",
        mime="audio/wav",
        audio=_audio(),
        language="en-US",
    )


@pytest.fixture
def interactions_client() -> SimpleNamespace:
    return SimpleNamespace(
        aio=SimpleNamespace(
            files=SimpleNamespace(
                upload=AsyncMock(
                    return_value=SimpleNamespace(
                        name="files/audio",
                        uri="https://example.invalid/audio",
                        mime_type="audio/wav",
                    )
                ),
                delete=AsyncMock(return_value=None),
            ),
            interactions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        output_text="transcript",
                        usage=SimpleNamespace(
                            total_input_tokens=12,
                            total_output_tokens=3,
                            total_tokens=999,
                        ),
                    )
                )
            ),
        )
    )


@pytest.mark.unit
async def test_interactions_transcription_sends_audio_only(
    interactions_client: SimpleNamespace,
) -> None:
    client = interactions_client

    result = await transcribe_interactions(
        cast(Client, client), "gemini-3.5-transcribe", _request()
    )

    assert result.text == "transcript"
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 3
    assert result.metadata.total_tokens == 15
    client.aio.files.upload.assert_awaited_once()
    client.aio.files.delete.assert_awaited_once_with(name="files/audio")
    call = client.aio.interactions.create.await_args.kwargs
    assert call["model"] == "gemini-3.5-transcribe"
    assert call["input"] == [
        {
            "type": "audio",
            "uri": "https://example.invalid/audio",
            "mime_type": "audio/wav",
        }
    ]


@pytest.mark.unit
async def test_interactions_cleanup_failure_warns_without_failing(
    caplog: pytest.LogCaptureFixture,
    interactions_client: SimpleNamespace,
) -> None:
    client = interactions_client
    client.aio.files.delete.side_effect = RuntimeError("secret file id")

    result = await transcribe_interactions(
        cast(Client, client), "gemini-3.5-transcribe", _request()
    )

    assert result.text == "transcript"
    assert "Google transcription upload cleanup failed" in caplog.text
    assert "secret file id" not in caplog.text


@pytest.mark.unit
async def test_interactions_cleanup_failure_preserves_primary_error(
    caplog: pytest.LogCaptureFixture,
    interactions_client: SimpleNamespace,
) -> None:
    client = interactions_client
    client.aio.files.delete.side_effect = RuntimeError("secret file id")
    client.aio.interactions.create.side_effect = RuntimeError("interaction failed")

    with pytest.raises(RuntimeError, match="interaction failed"):
        await transcribe_interactions(
            cast(Client, client), "gemini-3.5-transcribe", _request()
        )

    assert "Google transcription upload cleanup failed" in caplog.text
    assert "secret file id" not in caplog.text


async def _messages() -> Any:
    yield SimpleNamespace(
        voice_activity=None,
        usage_metadata=None,
        server_content=SimpleNamespace(
            interim_input_transcription=SimpleNamespace(text="interim transcript"),
            input_transcription=None,
            turn_complete=False,
        ),
    )
    yield SimpleNamespace(
        voice_activity=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            response_token_count=3,
        ),
        server_content=SimpleNamespace(
            interim_input_transcription=None,
            input_transcription=SimpleNamespace(text="transcript", finished=True),
            turn_complete=True,
        ),
    )


@pytest.mark.unit
async def test_live_transcription_caps_ingest_rate_and_ends_the_stream() -> None:
    session = SimpleNamespace()
    session.send_realtime_input = AsyncMock()
    session.receive = lambda: _messages()

    def connect(*, model: str, config: Any) -> Connection[SimpleNamespace]:
        assert model == "gemini-3.5-transcribe-live"
        assert config.input_audio_transcription is not None
        assert config.realtime_input_config.automatic_activity_detection.disabled
        return Connection(session)

    client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=connect)))

    with patch(
        "model_library.base.transcription.asyncio.sleep", new=AsyncMock()
    ) as sleep:
        result = await transcribe_live(
            cast(Client, client), "gemini-3.5-transcribe-live", _request()
        )

    assert result.text == "transcript"
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 3
    assert result.metadata.time_to_first_partial_seconds is not None
    calls = session.send_realtime_input.await_args_list
    assert len(calls) == 4
    assert "activity_start" in calls[0].kwargs
    assert calls[1].kwargs["audio"].mime_type == "audio/pcm;rate=16000"
    assert b"".join(call.kwargs["audio"].data for call in calls[1:3]) == (
        b"\0\0" * 3200
    )
    assert "activity_end" in calls[3].kwargs
    # Live rejects audio delivered faster than 10x realtime.
    assert sleep.await_count == 1
    assert sleep.await_args is not None
    assert sleep.await_args.args[0] == pytest.approx(0.01, abs=0.005)


@pytest.mark.unit
@pytest.mark.parametrize("with_usage", [False, True])
@pytest.mark.parametrize("final_text", [None, "", "hello world"])
async def test_live_transcription_finishes_after_all_audio_segments(
    with_usage: bool,
    final_text: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One manual turn completes on final text, never on activity or elapsed time."""
    ended = asyncio.Event()
    audio = bytearray()

    async def send(**kwargs: Any) -> None:
        if "audio" in kwargs:
            audio.extend(kwargs["audio"].data)
        if "activity_end" in kwargs:
            assert bytes(audio) == b"\0\0" * 3200
            ended.set()

    async def messages() -> AsyncIterator[LiveServerMessage]:
        await ended.wait()
        yield LiveServerMessage.model_validate(
            {
                "voiceActivity": {
                    "voiceActivityType": "ACTIVITY_END",
                    "audioOffset": "0.200s",
                }
            }
        )
        yield LiveServerMessage.model_validate(
            {
                "serverContent": {
                    "interimInputTranscription": {"text": "hello"},
                    "generationComplete": True,
                }
            }
        )
        if with_usage:
            yield LiveServerMessage.model_validate(
                {"usageMetadata": {"promptTokenCount": 12, "responseTokenCount": 3}}
            )
        if final_text is not None:
            yield LiveServerMessage.model_validate(
                {"serverContent": {"inputTranscription": {"text": final_text}}}
            )
        await asyncio.Event().wait()

    session = SimpleNamespace(send_realtime_input=send, receive=messages)
    client = SimpleNamespace(
        aio=SimpleNamespace(
            live=SimpleNamespace(connect=lambda **_: Connection(session))
        )
    )
    if not final_text:
        monkeypatch.setattr(
            "model_library.providers.google.transcribe._GOOGLE_FINAL_TIMEOUT_SECONDS",
            0.01,
        )
        with pytest.raises(TimeoutError):
            await transcribe_live(
                cast(Client, client), "gemini-3.5-transcribe-live", _request()
            )
        return
    result = await asyncio.wait_for(
        transcribe_live(cast(Client, client), "gemini-3.5-transcribe-live", _request()),
        1,
    )
    assert result.text == "hello world"
    assert result.metadata.input_tokens == (12 if with_usage else None)
    assert result.metadata.output_tokens == (3 if with_usage else None)
