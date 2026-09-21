import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeAlias

from assemblyai.streaming.v3 import (  # pyright: ignore[reportMissingImports]
    AsyncStreamingClient,
    Encoding,
    RealTimeError,
    RealTimeEvents,
    SpeechModel,
    StreamingClientOptions,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import LLMConfig, TranscriptionOnly, TranscriptionResult
from model_library.exceptions import ModelNoOutputError
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

_ASSEMBLYAI_SAMPLE_RATE = 16_000
_ASSEMBLYAI_FINAL_IDLE_TIMEOUT_SECONDS = 10.0
_ASSEMBLYAI_TRANSPORT_TIMEOUT_SECONDS = 10.0

_AssemblyAIEvent: TypeAlias = TurnEvent | TerminationEvent


async def _run_until_provider_error(
    operation: Callable[[], Awaitable[None]],
    error_received: asyncio.Event,
    provider_errors: list[RealTimeError],
) -> None:
    if provider_errors:
        raise RuntimeError(f"Transcription failed: {provider_errors[0]}")

    error_waiter = asyncio.create_task(error_received.wait())
    operation_task = asyncio.ensure_future(operation())
    try:
        done, _ = await asyncio.wait(
            (error_waiter, operation_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if error_waiter in done:
            operation_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await operation_task
            raise RuntimeError(f"Transcription failed: {provider_errors[0]}")
        await operation_task
        if provider_errors:
            raise RuntimeError(f"Transcription failed: {provider_errors[0]}")
    finally:
        error_waiter.cancel()
        operation_task.cancel()
        with suppress(asyncio.CancelledError):
            await error_waiter
        with suppress(asyncio.CancelledError, Exception):
            await operation_task


@register_provider("assemblyai")
class AssemblyAIModel(TranscriptionOnly):
    """Complete-file transcription through AssemblyAI's v3 streaming socket."""

    provider_name = "assemblyai"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.ASSEMBLYAI_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    @staticmethod
    async def _collect_transcript(
        events: asyncio.Queue[_AssemblyAIEvent],
        collector: TranscriptCollector,
        finalized: asyncio.Event,
    ) -> TerminationEvent:
        while True:
            try:
                payload = await get_stream_event(
                    events,
                    finalized,
                    _ASSEMBLYAI_FINAL_IDLE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                raise ModelNoOutputError(
                    "Transcription stream ended before termination"
                ) from None
            if isinstance(payload, TerminationEvent):
                return payload
            collector.observe(
                payload.transcript.strip(),
                is_final=payload.end_of_turn,
            )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(
            request.audio,
            required_sample_rate=_ASSEMBLYAI_SAMPLE_RATE,
        )
        client = AsyncStreamingClient(
            StreamingClientOptions(
                api_key=self._api_key(),
                api_host=self.custom_endpoint or "streaming.assemblyai.com",
            )
        )

        events: asyncio.Queue[_AssemblyAIEvent] = asyncio.Queue()
        error_received = asyncio.Event()
        provider_errors: list[RealTimeError] = []

        def enqueue(
            _client: AsyncStreamingClient,
            payload: _AssemblyAIEvent | RealTimeError,
        ) -> None:
            if isinstance(payload, RealTimeError):
                if not provider_errors:
                    provider_errors.append(payload)
                error_received.set()
                return
            events.put_nowait(payload)

        for event in (
            RealTimeEvents.Turn,
            RealTimeEvents.Termination,
            RealTimeEvents.Error,
        ):
            client.on(event, enqueue)  # pyright: ignore[reportUnknownMemberType]

        terminated = False

        async def terminate() -> None:
            nonlocal terminated
            await asyncio.wait_for(
                client.disconnect(terminate=True),
                _ASSEMBLYAI_TRANSPORT_TIMEOUT_SECONDS,
            )
            terminated = True

        try:
            await client.connect(
                StreamingParameters(
                    speech_model=SpeechModel(self.model_name),
                    encoding=Encoding.pcm_s16le,
                    sample_rate=_ASSEMBLYAI_SAMPLE_RATE,
                    language_codes=(
                        [request.language] if request.language is not None else None
                    ),
                )
            )
            collector = TranscriptCollector()
            finalized = asyncio.Event()
            async with asyncio.TaskGroup() as task_group:
                receiver = task_group.create_task(
                    self._collect_transcript(events, collector, finalized)
                )
                await _run_until_provider_error(
                    lambda: stream_audio_chunks(
                        parsed.frames,
                        chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                        send=client.stream,
                        collector=collector,
                        send_timeout_seconds=_ASSEMBLYAI_TRANSPORT_TIMEOUT_SECONDS,
                    ),
                    error_received,
                    provider_errors,
                )
                await _run_until_provider_error(
                    terminate,
                    error_received,
                    provider_errors,
                )
                finalized.set()
            termination = receiver.result()
        finally:
            if not terminated:
                await asyncio.wait_for(
                    client.disconnect(),
                    _ASSEMBLYAI_TRANSPORT_TIMEOUT_SECONDS,
                )

        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            billable_duration_seconds=termination.session_duration_seconds,
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
