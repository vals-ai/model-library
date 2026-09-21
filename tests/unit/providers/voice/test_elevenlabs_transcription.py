"""ElevenLabs transcription adapter tests."""

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

from elevenlabs import RealtimeEvents
import pytest
from pydantic import SecretStr

from model_library.base import TranscriptionOnly
from model_library.exceptions import ModelNoOutputError
from model_library.providers.voice.elevenlabs import ElevenLabsModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    ScriptedStream,
    config,
)

_TRANSCRIPTION_ONLY_GET_CLIENT = TranscriptionOnly.get_client
_ELEVENLABS_GET_CLIENT = ElevenLabsModel.get_client


class _Connection(ScriptedStream):
    def __init__(
        self,
        *,
        first_audio_events: list[tuple[RealtimeEvents, object]] | None = None,
        commit_events: list[tuple[RealtimeEvents, object]] | None = None,
        block_send: bool = False,
        block_commit: bool = False,
    ) -> None:
        super().__init__([])
        self._handlers: dict[RealtimeEvents, list[Callable[..., None]]] = {}
        self._first_audio_events = first_audio_events or []
        self._commit_events = commit_events or []
        self._block_send = block_send
        self._block_commit = block_commit
        self.events: list[str] = []
        self.closed = False

    def on(self, event: RealtimeEvents, callback: Callable[..., None]) -> None:
        self._handlers.setdefault(event, []).append(callback)

    def _emit(self, event: RealtimeEvents, *args: object) -> None:
        self.events.append(event.value)
        for handler in self._handlers.get(event, []):
            handler(*args)

    def _emit_script(self, script: list[tuple[RealtimeEvents, object]]) -> None:
        for event, payload in script:
            self._emit(event, payload)

    async def send(self, data: dict[str, object]) -> None:
        self._record_send(data)
        self.events.append("audio")
        if len(self.sends) == 1:
            self._emit_script(self._first_audio_events)
        if self._block_send:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    async def commit(self) -> None:
        self.events.append("commit")
        if self._block_commit:
            await asyncio.Event().wait()
        self._emit_script(self._commit_events)

    async def close(self) -> None:
        self.closed = True


def _client(connection: _Connection) -> MagicMock:
    client = MagicMock()
    client.speech_to_text.realtime.connect = AsyncMock(return_value=connection)
    return client


def test_elevenlabs_shares_client_between_model_instances() -> None:
    client = MagicMock()

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(ElevenLabsModel, "get_client", _ELEVENLABS_GET_CLIENT),
        patch(
            "model_library.providers.voice.elevenlabs.AsyncElevenLabs",
            return_value=client,
        ) as constructor,
    ):
        first = ElevenLabsModel(
            "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
        )
        second = ElevenLabsModel(
            "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
        )
        different_config = config("elevenlabs/scribe_v2_realtime")
        different_config.custom_api_key = SecretStr("other-key")
        different = ElevenLabsModel("scribe_v2_realtime", config=different_config)

        assert first.get_client() is second.get_client()
        assert first._client_registry_key != different._client_registry_key  # pyright: ignore[reportPrivateUsage]

    assert constructor.call_count == 2
    assert constructor.call_args_list[0].kwargs == {
        "api_key": "provider-key",
        "base_url": None,
    }


async def test_elevenlabs_commit_arms_timeout_for_pending_receive() -> None:
    connection = _Connection(
        first_audio_events=[
            (
                RealtimeEvents.PARTIAL_TRANSCRIPT,
                {"message_type": "partial_transcript", "text": "hel"},
            ),
            (
                RealtimeEvents.PARTIAL_TRANSCRIPT,
                {"message_type": "partial_transcript", "text": "hello"},
            ),
        ]
    )
    model = ElevenLabsModel(
        "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
    )

    with (
        patch.object(model, "get_client", return_value=_client(connection)),
        patch(
            "model_library.providers.voice.elevenlabs._ELEVENLABS_FINAL_IDLE_TIMEOUT_SECONDS",
            0,
        ),
        pytest.raises(ModelNoOutputError, match="did not include a final transcript"),
    ):
        await asyncio.wait_for(
            model.transcribe_audio(
                name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
            ),
            timeout=1,
        )

    assert "commit" in connection.events
    assert connection.closed


@pytest.mark.parametrize("block_send", [True, False])
async def test_elevenlabs_bounds_blocked_transport(block_send: bool) -> None:
    connection = _Connection(block_send=block_send, block_commit=not block_send)
    model = ElevenLabsModel(
        "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
    )

    with (
        patch.object(model, "get_client", return_value=_client(connection)),
        patch(
            "model_library.providers.voice.elevenlabs._ELEVENLABS_TRANSPORT_TIMEOUT_SECONDS",
            0.01,
        ),
        pytest.raises(TimeoutError),
    ):
        await asyncio.wait_for(
            model.transcribe_audio(
                name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
            ),
            timeout=1,
        )

    assert ("commit" in connection.events) is not block_send
    assert connection.closed


async def test_elevenlabs_surfaces_non_suffix_error_event() -> None:
    connection = _Connection(
        first_audio_events=[
            (
                RealtimeEvents.ERROR,
                {"message_type": "quota_exceeded", "error": "quota exhausted"},
            )
        ]
    )
    model = ElevenLabsModel(
        "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
    )

    with (
        patch.object(model, "get_client", return_value=_client(connection)),
        pytest.raises(RuntimeError, match="quota exhausted"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )


async def test_elevenlabs_maps_padded_audio_to_billing() -> None:
    connection = _Connection(
        commit_events=[
            (
                RealtimeEvents.COMMITTED_TRANSCRIPT,
                {"message_type": "committed_transcript", "text": "hello"},
            ),
            (RealtimeEvents.CLOSE, {}),
        ]
    )
    model = ElevenLabsModel(
        "scribe_v2_realtime", config=config("elevenlabs/scribe_v2_realtime")
    )

    with (
        patch.object(model, "get_client", return_value=_client(connection)),
        patch(
            "model_library.providers.voice.elevenlabs._ELEVENLABS_FINAL_IDLE_TIMEOUT_SECONDS",
            0,
        ),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "hello"
    assert result.metadata.billable_duration_seconds == 1
    assert result.metadata.cost_usd == pytest.approx(0.39 / 3_600)
