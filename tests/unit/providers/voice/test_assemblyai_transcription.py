"""AssemblyAI transcription adapter tests."""

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from assemblyai.streaming.v3 import (  # pyright: ignore[reportMissingImports]
    RealTimeError,
    RealTimeEvents,
    TerminationEvent,
    TurnEvent,
)
import pytest  # pyright: ignore[reportMissingImports]

from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    parse_mono_pcm16_wav,
)
from model_library.exceptions import ModelNoOutputError
from model_library.providers.voice.assemblyai import (
    AssemblyAIModel,
    _run_until_provider_error,
)
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    ScriptedStream,
    config,
)


def _turn(transcript: str, *, end_of_turn: bool) -> TurnEvent:
    return TurnEvent(
        type="Turn",
        turn_order=0,
        turn_is_formatted=False,
        end_of_turn=end_of_turn,
        transcript=transcript,
        end_of_turn_confidence=1.0,
        words=[],
    )


class _Client(ScriptedStream):
    def __init__(
        self,
        *,
        first_audio_events: list[tuple[RealTimeEvents, object]] | None = None,
        audio_event_send_index: int = 1,
        audio_event_gate: asyncio.Event | None = None,
        terminate_events: list[tuple[RealTimeEvents, object]] | None = None,
        block_stream: bool = False,
        block_terminate: bool = False,
    ) -> None:
        super().__init__([])
        self._handlers: dict[RealTimeEvents, list[Callable[..., None]]] = {}
        self._first_audio_events = first_audio_events or []
        self._audio_event_send_index = audio_event_send_index
        self._audio_event_gate = audio_event_gate
        self._terminate_events = terminate_events or []
        self._block_stream = block_stream
        self._block_terminate = block_terminate
        self.params: object | None = None
        self.events: list[str] = []
        self.disconnects: list[bool] = []

    def on(self, event: RealTimeEvents, callback: Callable[..., None]) -> None:
        self._handlers.setdefault(event, []).append(callback)

    def _emit(self, event: RealTimeEvents, payload: object) -> None:
        self.events.append(event.value)
        for handler in self._handlers.get(event, []):
            handler(self, payload)

    def _emit_script(self, script: list[tuple[RealTimeEvents, object]]) -> None:
        for event, payload in script:
            self._emit(event, payload)

    async def connect(self, params: object) -> None:
        self.params = params

    async def stream(self, chunk: bytes) -> None:
        self._record_send(chunk)
        self.events.append("audio")
        if len(self.sends) == self._audio_event_send_index:
            self._emit_script(self._first_audio_events)
            if self._audio_event_gate is not None:
                self._audio_event_gate.set()
        if self._block_stream:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    async def disconnect(self, terminate: bool = False) -> None:
        self.disconnects.append(terminate)
        if terminate:
            self.events.append("terminate")
            if self._block_terminate:
                await asyncio.Event().wait()
            self._emit_script(self._terminate_events)


async def test_assemblyai_error_monitor_cancels_operation_with_parent() -> None:
    operation_started = asyncio.Event()
    operation_stopped = asyncio.Event()

    async def operation() -> None:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_stopped.set()

    task = asyncio.create_task(
        _run_until_provider_error(operation, asyncio.Event(), [])
    )
    await operation_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert operation_stopped.is_set()


async def test_assemblyai_surfaces_error_event() -> None:
    client = _Client(
        first_audio_events=[
            (RealTimeEvents.Error, RealTimeError("quota exhausted", code=4003))
        ]
    )
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="quota exhausted"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    assert client.disconnects == [False]


async def test_assemblyai_surfaces_error_on_final_audio_chunk() -> None:
    receiver_gate = asyncio.Event()
    frame_bytes = len(parse_mono_pcm16_wav(AUDIO).frames)
    audio_chunks = (frame_bytes + PCM16_16KHZ_CHUNK_BYTES - 1) // (
        PCM16_16KHZ_CHUNK_BYTES
    )
    client = _Client(
        first_audio_events=[
            (RealTimeEvents.Error, RealTimeError("quota exhausted", code=4003))
        ],
        audio_event_send_index=audio_chunks,
        audio_event_gate=receiver_gate,
    )
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )
    original_collect = AssemblyAIModel._collect_transcript

    async def gated_collect(*args: Any):
        await receiver_gate.wait()
        return await original_collect(*args)

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        patch.object(
            AssemblyAIModel,
            "_collect_transcript",
            new=staticmethod(gated_collect),
        ),
        pytest.raises(RuntimeError, match="quota exhausted"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    assert len(client.sends) == audio_chunks
    assert client.disconnects == [False]


@pytest.mark.parametrize("block_stream", [True, False])
async def test_assemblyai_bounds_blocked_transport(block_stream: bool) -> None:
    client = _Client(block_stream=block_stream, block_terminate=not block_stream)
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        patch(
            "model_library.providers.voice.assemblyai._ASSEMBLYAI_TRANSPORT_TIMEOUT_SECONDS",
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

    assert client.disconnects == ([False] if block_stream else [True, False])


async def test_assemblyai_error_during_completed_finalization_skips_cleanup() -> None:
    client = _Client(
        terminate_events=[
            (RealTimeEvents.Error, RealTimeError("quota exhausted", code=4003))
        ]
    )
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="quota exhausted"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    assert client.disconnects == [True]


async def test_assemblyai_requires_termination_event() -> None:
    client = _Client(
        terminate_events=[(RealTimeEvents.Turn, _turn("hello world", end_of_turn=True))]
    )
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        patch(
            "model_library.providers.voice.assemblyai._ASSEMBLYAI_FINAL_IDLE_TIMEOUT_SECONDS",
            0,
        ),
        pytest.raises(ModelNoOutputError, match="ended before termination"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )


async def test_assemblyai_maps_session_duration_to_billing() -> None:
    client = _Client(
        terminate_events=[
            (RealTimeEvents.Turn, _turn("hello", end_of_turn=True)),
            (
                RealTimeEvents.Termination,
                TerminationEvent(
                    type="Termination",
                    audio_duration_seconds=1,
                    session_duration_seconds=2,
                ),
            ),
        ]
    )
    model = AssemblyAIModel(
        "universal-3-5-pro", config=config("assemblyai/universal-3-5-pro")
    )

    with (
        patch(
            "model_library.providers.voice.assemblyai.AsyncStreamingClient",
            return_value=client,
        ),
        patch(
            "model_library.providers.voice.assemblyai._ASSEMBLYAI_FINAL_IDLE_TIMEOUT_SECONDS",
            0,
        ),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "hello"
    assert result.metadata.billable_duration_seconds == 2
    assert result.metadata.cost_usd == pytest.approx(0.45 * 2 / 3_600)
    assert client.disconnects == [True]
