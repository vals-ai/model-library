"""Audio and response helpers shared by the standalone transcription providers."""

import asyncio
import io
import time
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeVar, cast

from model_library.base.output.transcription import (
    TranscriptionMetadata,
    TranscriptionResult,
)
from model_library.exceptions import ModelNoOutputError

# 100 ms of mono 16 kHz PCM16 audio.
PCM16_16KHZ_CHUNK_BYTES = 3_200

_TRAILING_PUNCTUATION = ",.!?;:"

_EventT = TypeVar("_EventT")


@dataclass(frozen=True)
class TranscriptionRequest:
    """The complete-file request passed to a transcription provider."""

    name: str
    mime: str
    audio: bytes
    # Already resolved to the provider's expected spelling by transcribe_audio.
    language: str | None


@dataclass(frozen=True, slots=True)
class MonoPcm16Audio:
    """Raw frames parsed from a mono, uncompressed PCM16 WAV payload."""

    frames: bytes
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class NormalizedTranscriptionUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    audio_tokens: int | None = None
    text_tokens: int | None = None
    billable_duration_seconds: float | None = None


def single_exception_from_group(
    group: ExceptionGroup[Exception],
) -> Exception | None:
    """Return one nested exception, or preserve a group with multiple failures."""
    exceptions = cast(tuple[Exception, ...], group.exceptions)
    if len(exceptions) != 1:
        return None
    error = exceptions[0]
    if isinstance(error, ExceptionGroup):
        return single_exception_from_group(cast(ExceptionGroup[Exception], error))
    return error


class TranscriptCollector:
    """Accumulates final segments and time to first transcript text for one stream."""

    def __init__(self) -> None:
        self.time_to_first_partial_seconds: float | None = None
        self._segments: list[str] = []
        self._audio_started_at = time.perf_counter()

    @property
    def has_final(self) -> bool:
        return bool(self._segments)

    def audio_started(self) -> None:
        """Mark the instant audio starts leaving, the baseline for first-text timing."""
        self._audio_started_at = time.perf_counter()

    def observe(self, text: str, *, is_final: bool) -> None:
        """Record one transcript event and stamp time to first nonempty text."""
        if not text:
            return
        if self.time_to_first_partial_seconds is None:
            self.time_to_first_partial_seconds = (
                time.perf_counter() - self._audio_started_at
            )
        if is_final:
            self._segments.append(text)

    def transcript(self) -> str:
        """Join the final segments, spacing all but those opening with punctuation."""
        text = ""
        for segment in self._segments:
            stripped = segment.strip()
            if not stripped:
                continue
            space = " " if text and stripped[0] not in _TRAILING_PUNCTUATION else ""
            text = f"{text}{space}{stripped}"
        if not text:
            raise ModelNoOutputError(
                "Transcription response did not include a final transcript"
            )
        return text


def build_transcription_result(
    *,
    text: str,
    audio_bytes: int,
    usage: NormalizedTranscriptionUsage | None = None,
    billable_duration_seconds: float | None = None,
    time_to_first_partial_seconds: float | None = None,
) -> TranscriptionResult:
    """Build common result metadata for a complete-file transcription."""
    usage = usage or NormalizedTranscriptionUsage()
    return TranscriptionResult(
        text=text.strip(),
        metadata=TranscriptionMetadata(
            audio_bytes=audio_bytes,
            request_duration_seconds=0.0,
            billable_duration_seconds=(
                usage.billable_duration_seconds
                if usage.billable_duration_seconds is not None
                else billable_duration_seconds
            ),
            time_to_first_partial_seconds=time_to_first_partial_seconds,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            audio_tokens=usage.audio_tokens,
            text_tokens=usage.text_tokens,
        ),
    )


def finalize_transcription_result(
    *,
    result: TranscriptionResult,
    registry_key: str | None,
    audio: bytes,
    mime: str,
) -> TranscriptionResult:
    """Attach exact duration and registry-derived cost to a provider result."""
    audio_duration_seconds: float | None = None
    if mime in {"audio/wav", "audio/x-wav"}:
        with suppress(EOFError, wave.Error):
            with wave.open(io.BytesIO(audio), "rb") as source:
                frame_rate = source.getframerate()
                if frame_rate > 0:
                    audio_duration_seconds = source.getnframes() / frame_rate
    result.metadata.audio_duration_seconds = audio_duration_seconds

    if registry_key is None:
        return result

    from model_library.registry_utils import (
        compute_transcription_cost,
        get_transcription_registry_config,
    )

    registry_config = get_transcription_registry_config(registry_key)
    if (
        registry_config is not None
        and registry_config.transcription_cost is not None
        and registry_config.transcription_cost.billing_basis == "audio"
        and result.metadata.billable_duration_seconds is None
    ):
        result.metadata.billable_duration_seconds = audio_duration_seconds
    result.metadata.cost_usd = compute_transcription_cost(registry_key, result.metadata)
    return result


def parse_mono_pcm16_wav(
    audio: bytes,
    *,
    required_sample_rate: int | None = None,
) -> MonoPcm16Audio:
    """Extract raw frames from one mono, uncompressed PCM16 WAV payload."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            if source.getcomptype() != "NONE" or channels != 1 or sample_width != 2:
                raise ValueError("Transcription requires mono PCM16 WAV audio")
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError("Transcription requires a valid WAV audio payload") from exc

    if sample_rate == 0:
        raise ValueError("Transcription requires WAV audio with a positive sample rate")
    if len(frames) != frame_count * channels * sample_width:
        raise ValueError("Transcription requires a complete WAV audio payload")
    if required_sample_rate is not None and sample_rate != required_sample_rate:
        raise ValueError(
            f"Transcription requires {required_sample_rate // 1_000} kHz mono PCM16 WAV audio"
        )
    if not frames:
        raise ValueError("Transcription audio must not be empty")
    return MonoPcm16Audio(frames=frames, sample_rate_hz=sample_rate)


async def get_stream_event(
    events: asyncio.Queue[_EventT],
    finalized: asyncio.Event,
    terminal_timeout_seconds: float,
) -> _EventT:
    """Receive continuously, arming the timeout only after upload finalization."""
    if finalized.is_set():
        try:
            return events.get_nowait()
        except asyncio.QueueEmpty:
            return await asyncio.wait_for(events.get(), terminal_timeout_seconds)

    event_task = asyncio.create_task(events.get())
    finalized_task = asyncio.create_task(finalized.wait())
    try:
        done, _ = await asyncio.wait(
            {event_task, finalized_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if event_task in done:
            return event_task.result()
        return await asyncio.wait_for(event_task, terminal_timeout_seconds)
    finally:
        for task in (event_task, finalized_task):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


async def stream_audio_chunks(
    audio: bytes,
    *,
    chunk_bytes: int,
    send: Callable[[bytes], Awaitable[None]],
    collector: TranscriptCollector,
    max_bytes_per_second: int | None = None,
    send_timeout_seconds: float | None = None,
) -> None:
    """Send audio as fast as the transport accepts.

    `max_bytes_per_second` throttles the send rate for engines that drop
    transcripts, or close the socket, when a whole file arrives at once. The
    throttle follows the schedule the first chunk set, so send latency cannot
    accumulate into a rate below the requested one. `send_timeout_seconds`
    bounds each transport write without limiting the total upload duration.
    """
    collector.audio_started()
    started = time.perf_counter() if max_bytes_per_second is not None else 0.0
    for offset in range(0, len(audio), chunk_bytes):
        chunk = audio[offset : offset + chunk_bytes]
        if send_timeout_seconds is None:
            await send(chunk)
        else:
            await asyncio.wait_for(send(chunk), send_timeout_seconds)
        sent = offset + len(chunk)
        if max_bytes_per_second is not None and sent < len(audio):
            behind = sent / max_bytes_per_second - (time.perf_counter() - started)
            if behind > 0:
                await asyncio.sleep(behind)


def resolve_language(language: str | None, registry_key: str | None) -> str | None:
    """Spell `language` the way the registered model expects it."""
    from model_library.registry_utils import get_transcription_registry_config

    registry_config = (
        get_transcription_registry_config(registry_key) if registry_key else None
    )
    spec = registry_config.transcription_language if registry_config else None
    if spec is None:
        return language
    if not language:
        return spec.default
    if spec.format == "iso-639-1":
        return language.split("-", 1)[0]
    if language == spec.default.split("-", 1)[0]:
        return spec.default
    return language
