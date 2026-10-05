"""Reson8 transcription adapter tests."""

import asyncio
import json
from typing import cast
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from model_library.base.transcription import parse_mono_pcm16_wav
from model_library.providers.voice.reson8 import Reson8Model
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection,
    ScriptedStream,
    config,
)


async def test_reson8_prerecorded_posts_raw_audio_and_returns_text() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"text": "hello world"})

    model = Reson8Model("resonant-1", config=config("reson8/resonant-1"))
    with patch(
        "model_library.providers.voice.reson8.default_httpx_client",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="nl"
        )

    assert result.text == "hello world"
    assert result.metadata.billable_duration_seconds == pytest.approx(0.75)
    assert result.metadata.cost_usd == pytest.approx(0.0000859725)
    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == (
        "https://api.reson8.dev/v1/speech-to-text/prerecorded?language=nl"
    )
    assert request.headers["Authorization"] == "ApiKey provider-key"
    assert request.headers["Content-Type"] == "application/octet-stream"
    assert request.content == AUDIO


class _Reson8RealtimeStream(ScriptedStream):
    """Emits the script only after the client flushes, then stays open."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(cast(list[object], responses))
        self._flushed = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        self._record_send(message)
        if isinstance(message, str) and json.loads(message) == {
            "type": "flush_request"
        }:
            self._flushed.set()

    async def __aiter__(self):
        await self._flushed.wait()
        for response in self._script:
            yield json.dumps(response)
        await asyncio.Event().wait()


async def test_reson8_realtime_streams_pcm_and_stops_at_flush_confirmation() -> None:
    stream = _Reson8RealtimeStream(
        [
            {"type": "transcript", "text": "first", "is_final": False},
            {"type": "transcript", "text": "first segment", "is_final": True},
            {"type": "transcript", "text": "second segment", "is_final": True},
            {"type": "flush_confirmation", "id": None},
        ]
    )
    connect = MagicMock(return_value=Connection(stream))
    model = Reson8Model(
        "resonant-1-realtime", config=config("reson8/resonant-1-realtime")
    )

    with patch("model_library.providers.voice.reson8.connect", connect):
        result = await asyncio.wait_for(
            model.transcribe_audio(
                name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
            ),
            timeout=2,
        )

    assert result.text == "first segment second segment"
    assert result.metadata.billable_duration_seconds == pytest.approx(0.75)
    assert result.metadata.cost_usd == pytest.approx(0.000171945)
    assert result.metadata.time_to_first_partial_seconds is not None
    url = urlsplit(connect.call_args.args[0])
    assert (url.scheme, url.netloc, url.path) == (
        "wss",
        "api.reson8.dev",
        "/v1/speech-to-text/realtime",
    )
    assert parse_qs(url.query) == {
        "encoding": ["pcm_s16le"],
        "sample_rate": ["16000"],
        "channels": ["1"],
        "include_interim": ["true"],
        "language": ["en"],
    }
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "ApiKey provider-key"
    }
    audio_sent = b"".join(sent for sent in stream.sends if isinstance(sent, bytes))
    assert audio_sent == parse_mono_pcm16_wav(AUDIO).frames
    assert stream.sends[-1] == json.dumps({"type": "flush_request"})
