"""Cartesia transcription adapter tests."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from cartesia.types.stt import (
    STTManualFinalizeDoneResponse,
    STTManualFinalizeFlushDoneResponse,
    STTManualFinalizeTranscriptResponse,
)
from pydantic import SecretStr

from model_library.base import TranscriptionOnly
from model_library.providers.voice.cartesia import CartesiaModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    ScriptedStream,
    config,
)

_TRANSCRIPTION_ONLY_GET_CLIENT = TranscriptionOnly.get_client
_CARTESIA_GET_CLIENT = CartesiaModel.get_client


def _transcript(text: str, *, is_final: bool) -> STTManualFinalizeTranscriptResponse:
    return STTManualFinalizeTranscriptResponse(
        type="transcript", request_id="request-1", text=text, is_final=is_final
    )


class _Connection(ScriptedStream):
    def __init__(self, responses: list[STTManualFinalizeTranscriptResponse]) -> None:
        super().__init__(cast(list[object], responses))
        self._responses = responses
        self._audio_sent = asyncio.Event()
        self._finalized = asyncio.Event()
        self._closed = asyncio.Event()
        self.audio: list[bytes] = []
        self.commands: list[str] = []
        self.events: list[str] = []

    async def send_raw(self, data: bytes | str) -> None:
        assert isinstance(data, bytes)
        self._record_send(data)
        self.audio.append(data)
        self.events.append("audio")
        self._audio_sent.set()
        await asyncio.sleep(0)

    async def send(self, command: str) -> None:
        self.commands.append(command)
        self.events.append(command)
        if command == "finalize":
            self._finalized.set()
        elif command == "close":
            self._closed.set()

    async def __aiter__(self):
        await self._audio_sent.wait()
        for response in self._responses:
            if not response.is_final:
                self.events.append("partial")
                yield response
        await self._finalized.wait()
        self.events.append("flush_done")
        yield STTManualFinalizeFlushDoneResponse(
            type="flush_done", request_id="request-1"
        )
        await self._closed.wait()
        for response in self._responses:
            if response.is_final:
                self.events.append("final")
                yield response
        yield STTManualFinalizeDoneResponse(type="done", request_id="request-1")


class _WebsocketManager:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _Connection:
        return self._connection

    async def __aexit__(self, *_args: object) -> None:
        self._connection.events.append("ws_closed")


class _Client:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self.websocket_kwargs: dict[str, object] | None = None
        self.closed = False
        self.stt = SimpleNamespace(
            manual_finalize=SimpleNamespace(websocket=self._websocket)
        )

    def _websocket(self, **kwargs: object) -> _WebsocketManager:
        self.websocket_kwargs = kwargs
        return _WebsocketManager(self._connection)


async def test_cartesia_streams_manual_finalize_and_tracks_ttfp() -> None:
    connection = _Connection(
        [
            _transcript("h", is_final=False),
            _transcript(" hel", is_final=True),
            _transcript("lo ", is_final=True),
        ]
    )
    client = _Client(connection)
    model_config = config("cartesia/ink-2")
    model_config.custom_endpoint = "https://proxy.example/cartesia/stt/websocket"
    different_config = config("cartesia/ink-2")
    different_config.custom_endpoint = model_config.custom_endpoint
    different_config.custom_api_key = SecretStr("other-key")

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(CartesiaModel, "get_client", _CARTESIA_GET_CLIENT),
        patch(
            "model_library.providers.voice.cartesia.AsyncCartesia",
            return_value=client,
        ) as constructor,
        patch(
            "model_library.base.base.perf_counter",
            side_effect=[0.0, 0.9, 1.0, 1.9],
        ),
    ):
        first_model = CartesiaModel("ink-2", config=model_config)
        second_model = CartesiaModel("ink-2", config=model_config)
        different_model = CartesiaModel("ink-2", config=different_config)

        assert first_model.get_client() is second_model.get_client()
        assert first_model._client_registry_key != different_model._client_registry_key  # pyright: ignore[reportPrivateUsage]

        result = await first_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )
        second_result = await second_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "hello"
    assert second_result.text == "hello"
    assert result.metadata.request_duration_seconds == 0.9
    assert result.metadata.time_to_first_partial_seconds is not None
    assert constructor.call_count == 2
    assert constructor.call_args_list[0].kwargs == {
        "api_key": "provider-key",
        "websocket_base_url": "https://proxy.example/cartesia/stt/websocket",
    }
    assert client.websocket_kwargs == {
        "encoding": "pcm_s16le",
        "model": "ink-2",
        "sample_rate": 16_000,
        "language": "en",
    }
    assert b"".join(connection.audio)
    assert connection.commands == ["finalize", "close"] * 2
    assert connection.events.index("audio") < connection.events.index("partial")
    assert connection.events.index("partial") < connection.events.index("finalize")
    assert connection.events.index("finalize") < connection.events.index("flush_done")
    assert connection.events.index("flush_done") < connection.events.index("close")
    assert connection.events.index("close") < connection.events.index("final")
    assert connection.events[-1] == "ws_closed"
    assert not client.closed
