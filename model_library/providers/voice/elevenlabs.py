import asyncio
import base64
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, cast

from elevenlabs import (
    AsyncElevenLabs,
    AudioFormat,
    CommitStrategy,
    RealtimeAudioOptions,
    RealtimeEvents,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import TranscriptionOnly, TranscriptionResult
from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    get_stream_event,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)
from model_library.register_models import register_provider

if TYPE_CHECKING:
    from elevenlabs.speech_to_text_custom import AsyncSpeechToTextClient

_ELEVENLABS_SAMPLE_RATE = 16_000
_ELEVENLABS_BYTES_PER_SECOND = _ELEVENLABS_SAMPLE_RATE * 2
_ELEVENLABS_MIN_AUDIO_BYTES = _ELEVENLABS_BYTES_PER_SECOND
_ELEVENLABS_FINAL_IDLE_TIMEOUT_SECONDS = 10.0
_ELEVENLABS_TRANSPORT_TIMEOUT_SECONDS = 10.0

_ELEVENLABS_EVENTS = (
    RealtimeEvents.PARTIAL_TRANSCRIPT,
    RealtimeEvents.COMMITTED_TRANSCRIPT,
    RealtimeEvents.ERROR,
    RealtimeEvents.CLOSE,
)

_ElevenLabsEvent = tuple[RealtimeEvents, dict[str, object]]


@register_provider("elevenlabs")
class ElevenLabsModel(TranscriptionOnly):
    """Complete-file transcription using ElevenLabs Scribe V2 Realtime."""

    provider_name = "elevenlabs"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.ELEVENLABS_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncElevenLabs:
        if not self.has_client():
            assert api_key is not None
            self.assign_client(AsyncElevenLabs(api_key=api_key, base_url=base_url))
        return cast(AsyncElevenLabs, super().get_client())

    @staticmethod
    async def _collect_transcript(
        events: asyncio.Queue[_ElevenLabsEvent],
        collector: TranscriptCollector,
        finalized: asyncio.Event,
    ) -> None:
        while True:
            try:
                event, payload = await get_stream_event(
                    events,
                    finalized,
                    _ELEVENLABS_FINAL_IDLE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                break
            if event is RealtimeEvents.CLOSE:
                break
            if event is RealtimeEvents.ERROR:
                raise RuntimeError(f"Transcription failed: {payload.get('error')}")

            raw_text = payload.get("text")
            collector.observe(
                raw_text.strip() if isinstance(raw_text, str) else "",
                is_final=event is RealtimeEvents.COMMITTED_TRANSCRIPT,
            )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(
            request.audio,
            required_sample_rate=_ELEVENLABS_SAMPLE_RATE,
        )
        # Scribe rejects clips shorter than a second, so pad them with silence.
        audio = parsed.frames.ljust(_ELEVENLABS_MIN_AUDIO_BYTES, b"\x00")
        options: RealtimeAudioOptions = {
            "model_id": self.model_name,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": _ELEVENLABS_SAMPLE_RATE,
            "commit_strategy": CommitStrategy.VAD,
        }
        if request.language is not None:
            options["language_code"] = request.language
        speech_to_text = cast(
            "AsyncSpeechToTextClient", self.get_client().speech_to_text
        )
        connection = await speech_to_text.realtime.connect(options)

        events: asyncio.Queue[_ElevenLabsEvent] = asyncio.Queue()

        def enqueue(event: RealtimeEvents, *args: object) -> None:
            first = args[0] if args else None
            payload = (
                cast("dict[str, object]", first) if isinstance(first, dict) else {}
            )
            events.put_nowait((event, payload))

        for event in _ELEVENLABS_EVENTS:
            connection.on(event, partial(enqueue, event))  # pyright: ignore[reportUnknownMemberType]

        try:

            async def send(chunk: bytes) -> None:
                await connection.send(
                    {"audio_base_64": base64.b64encode(chunk).decode("ascii")}
                )

            collector = TranscriptCollector()
            finalized = asyncio.Event()
            async with asyncio.TaskGroup() as task_group:
                _ = task_group.create_task(
                    self._collect_transcript(events, collector, finalized)
                )
                await stream_audio_chunks(
                    audio,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=send,
                    collector=collector,
                    send_timeout_seconds=_ELEVENLABS_TRANSPORT_TIMEOUT_SECONDS,
                )
                await asyncio.wait_for(
                    connection.commit(),
                    _ELEVENLABS_TRANSPORT_TIMEOUT_SECONDS,
                )
                finalized.set()
        finally:
            with suppress(Exception):
                await asyncio.wait_for(
                    connection.close(),
                    _ELEVENLABS_TRANSPORT_TIMEOUT_SECONDS,
                )
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            billable_duration_seconds=len(audio) / _ELEVENLABS_BYTES_PER_SECOND,
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
