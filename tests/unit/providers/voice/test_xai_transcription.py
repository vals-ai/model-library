"""xAI transcription adapter tests."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from pydantic import SecretStr

from model_library import raw_model
from model_library.base import LLMConfig
from model_library.exceptions import ModelNoOutputError
from model_library.providers.voice.xai import XAITranscriptionModel
from model_library.providers.xai import XAIModel
from model_library.registry_utils import get_transcription_model
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection,
    ScriptedStream,
    config,
)


class _XAIStream(ScriptedStream):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(cast(list[object], responses))
        self._audio_done = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        self._record_send(message)
        if isinstance(message, str) and json.loads(message).get("type") == "audio.done":
            self._audio_done.set()

    async def __aiter__(self) -> AsyncIterator[str]:
        yield json.dumps({"type": "transcript.created"})
        await self._audio_done.wait()
        for response in self._script:
            if isinstance(response, dict):
                yield json.dumps(response)


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (
            [
                {
                    "type": "transcript.partial",
                    "text": "hello",
                    "is_final": True,
                    "speech_final": False,
                },
                {
                    "type": "transcript.partial",
                    "text": "world",
                    "is_final": True,
                    "speech_final": False,
                },
                {"type": "transcript.done", "text": ""},
            ],
            "hello world",
        ),
        (
            [
                {
                    "type": "transcript.partial",
                    "text": "hello",
                    "is_final": True,
                    "speech_final": False,
                },
                {
                    "type": "transcript.partial",
                    "text": "hello world",
                    "is_final": True,
                    "speech_final": True,
                },
                {"type": "transcript.done", "text": ""},
            ],
            "hello world",
        ),
        (
            [
                {
                    "type": "transcript.partial",
                    "text": "stale chunk",
                    "is_final": True,
                    "speech_final": False,
                },
                {
                    "type": "transcript.partial",
                    "text": "stale stitched utterance",
                    "is_final": True,
                    "speech_final": True,
                },
                {"type": "transcript.done", "text": "authoritative done"},
            ],
            "authoritative done",
        ),
        (
            [
                {
                    "type": "transcript.partial",
                    "text": "speech final",
                    "speech_final": True,
                },
                {"type": "transcript.done", "text": ""},
            ],
            "speech final",
        ),
    ],
)
async def test_xai_accepts_documented_final_transcript_events(
    responses: list[dict[str, object]], expected: str
) -> None:
    stream = _XAIStream(responses)
    model = XAITranscriptionModel(
        "grok-voice-transcribe-2.0", config=config("grok/grok-voice-transcribe-2.0")
    )

    with (
        patch(
            "model_library.providers.voice.xai.connect",
            return_value=Connection(stream),
        ) as connect,
        patch(
            "model_library.base.transcription.asyncio.sleep", new=AsyncMock()
        ) as sleep,
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == expected
    assert any(isinstance(message, bytes) for message in stream.sends)
    assert sleep.await_count > 0
    endpoint = cast(str, connect.call_args.args[0])
    assert "model=grok-voice-transcribe-2.0" in endpoint


async def test_xai_rejects_empty_done_transcript() -> None:
    stream = _XAIStream(
        [
            {"type": "transcript.partial", "text": "first partial"},
            {"type": "transcript.partial", "text": "second partial"},
            {"type": "transcript.partial", "text": "stitched partial"},
            {"type": "transcript.done", "text": ""},
        ]
    )
    model = XAITranscriptionModel(
        "grok-voice-transcribe-1.0", config=config("grok/grok-voice-transcribe-1.0")
    )

    with (
        patch(
            "model_library.providers.voice.xai.connect",
            return_value=Connection(stream),
        ),
        pytest.raises(ModelNoOutputError, match="did not include a final transcript"),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)


@pytest.mark.parametrize(
    "key", ["grok/grok-voice-transcribe-2.0", "grok/grok-voice-transcribe-1.0"]
)
def test_transcription_keys_build_the_grok_model_without_a_key(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with patch("model_library.providers.xai.model_library_settings", new=object()):
        model = get_transcription_model(key)

    assert isinstance(model, XAIModel)
    assert model.provider == "grok"
    assert isinstance(model.delegate, XAITranscriptionModel)


async def test_transcription_model_skips_the_chat_rate_limit_probe() -> None:
    model = get_transcription_model("grok/grok-voice-transcribe-2.0")

    with patch(
        "model_library.providers.xai.probe_chat_completions_rate_limit"
    ) as probe:
        assert await model.get_rate_limit() is None

    probe.assert_not_called()


async def test_raw_grok_model_transcribes_over_the_socket() -> None:
    stream = _XAIStream([{"type": "transcript.done", "text": "hello"}])
    model = raw_model(
        "grok/grok-voice-transcribe-2.0",
        config=LLMConfig(custom_api_key=SecretStr("provider-key")),
    )

    with (
        patch(
            "model_library.providers.voice.xai.connect",
            return_value=Connection(stream),
        ) as connect,
        patch("model_library.base.transcription.asyncio.sleep", new=AsyncMock()),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "hello"
    endpoint = cast(str, connect.call_args.args[0])
    assert endpoint.startswith("wss://api.x.ai/v1/stt")
    assert "model=grok-voice-transcribe-2.0" in endpoint
