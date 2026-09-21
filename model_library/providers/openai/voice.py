from __future__ import annotations

import asyncio
import base64
import struct
import time
from typing import TYPE_CHECKING

from openai import omit
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.audio.transcription import (
    UsageDuration as FileTranscriptionUsageDuration,
    UsageTokens as FileTranscriptionUsageTokens,
)
from openai.types.audio.transcription_diarized import (
    UsageDuration as DiarizedTranscriptionUsageDuration,
    UsageTokens as DiarizedTranscriptionUsageTokens,
)
from openai.types.audio.transcription_text_done_event import (
    Usage as StreamingTranscriptionUsage,
)
from openai.types.audio.transcription_verbose import (
    Usage as VerboseTranscriptionUsageDuration,
)
from openai.types.realtime import RealtimeTranscriptionSessionCreateRequestParam
from openai.types.realtime.audio_transcription_param import AudioTranscriptionParam
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    Usage as RealtimeTranscriptionUsage,
    UsageTranscriptTextUsageDuration as RealtimeTranscriptionUsageDuration,
    UsageTranscriptTextUsageTokens as RealtimeTranscriptionUsageTokens,
)

from model_library.base import TranscriptionResult
from model_library.base.transcription import (
    NormalizedTranscriptionUsage,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)
from model_library.exceptions import ModelNoOutputError

if TYPE_CHECKING:
    from model_library.providers.openai.openai import OpenAIModel


OpenAITranscriptionUsage = (
    FileTranscriptionUsageTokens
    | FileTranscriptionUsageDuration
    | DiarizedTranscriptionUsageTokens
    | DiarizedTranscriptionUsageDuration
    | VerboseTranscriptionUsageDuration
    | StreamingTranscriptionUsage
    | RealtimeTranscriptionUsageTokens
    | RealtimeTranscriptionUsageDuration
)


_OPENAI_REALTIME_CHUNK_BYTES = 4_800


def _normalize_transcription_usage(
    usage: OpenAITranscriptionUsage | None,
) -> NormalizedTranscriptionUsage | None:
    if usage is None:
        return None
    if usage.type == "duration":
        return NormalizedTranscriptionUsage(billable_duration_seconds=usage.seconds)
    details = usage.input_token_details
    return NormalizedTranscriptionUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        audio_tokens=details.audio_tokens if details is not None else None,
        text_tokens=details.text_tokens if details is not None else None,
    )


def _realtime_pcm_24khz(audio: bytes) -> bytes:
    """Decode 16 kHz PCM WAV input and resample it to OpenAI's 24 kHz PCM."""
    pcm = parse_mono_pcm16_wav(audio, required_sample_rate=16_000).frames

    # 16 kHz to 24 kHz is a 2:3 ratio, so every 2 input samples become 3
    # output samples, linearly interpolated. audioop would do this, but it
    # was removed in Python 3.13.
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    output_samples = bytearray()
    for output_index in range((len(samples) * 3 + 1) // 2):
        source_index, remainder = divmod(output_index * 2, 3)
        first = samples[source_index]
        second = samples[min(source_index + 1, len(samples) - 1)]
        sample = first + (second - first) * remainder // 3
        output_samples.extend(struct.pack("<h", sample))
    return bytes(output_samples)


async def _receive_realtime_transcription(
    connection: AsyncRealtimeConnection,
    session_created: asyncio.Event,
    session_updated: asyncio.Event,
    collector: TranscriptCollector,
) -> RealtimeTranscriptionUsage:
    committed_item_id: str | None = None
    async for event in connection:
        match event.type:
            case "session.created":
                session_created.set()
            case "session.updated":
                session_updated.set()
            case "input_audio_buffer.committed":
                committed_item_id = event.item_id
            case "conversation.item.input_audio_transcription.failed" | "error":
                raise RuntimeError(
                    f"Transcription failed: {event.error.message or 'unknown error'}"
                )
            case "conversation.item.input_audio_transcription.delta":
                collector.observe(event.delta or "", is_final=False)
            case "conversation.item.input_audio_transcription.completed" if (
                event.item_id == committed_item_id
            ):
                collector.observe(event.transcript.strip(), is_final=True)
                if collector.has_final:
                    return event.usage
            case _:
                pass
    raise ModelNoOutputError(
        "Transcription response did not include a final transcript"
    )


async def transcribe_file_audio(
    model: OpenAIModel,
    *,
    name: str,
    mime: str,
    audio: bytes,
    language: str | None,
) -> TranscriptionResult:
    """Transcribe one complete audio file through the upload endpoint."""
    response = await model.get_client().audio.transcriptions.create(
        file=(name, audio, mime),
        model=model.model_name,
        response_format="json",
        language=language if language is not None else omit,
    )
    if not response.text.strip():
        raise ModelNoOutputError(
            "Transcription response did not include transcript text"
        )
    return build_transcription_result(
        text=response.text,
        audio_bytes=len(audio),
        usage=_normalize_transcription_usage(response.usage),
    )


async def _transcribe_file_audio_streaming_response(
    model: OpenAIModel,
    *,
    name: str,
    mime: str,
    audio: bytes,
    language: str | None,
) -> TranscriptionResult:
    started = time.perf_counter()
    stream = await model.get_client().audio.transcriptions.create(
        file=(name, audio, mime),
        model=model.model_name,
        response_format="json",
        stream=True,
        language=language if language is not None else omit,
    )
    final_text: str | None = None
    usage: StreamingTranscriptionUsage | None = None
    time_to_first_partial_seconds: float | None = None
    try:
        async for event in stream:
            if (
                event.type == "transcript.text.delta"
                and event.delta
                and time_to_first_partial_seconds is None
            ):
                time_to_first_partial_seconds = time.perf_counter() - started
            elif event.type == "transcript.text.done":
                final_text = event.text
                usage = event.usage
    finally:
        await stream.close()

    if final_text is None or not final_text.strip():
        raise ModelNoOutputError("Transcription stream did not include transcript text")
    return build_transcription_result(
        text=final_text,
        audio_bytes=len(audio),
        usage=_normalize_transcription_usage(usage),
        time_to_first_partial_seconds=time_to_first_partial_seconds,
    )


async def _transcribe_realtime_audio(
    model: OpenAIModel,
    *,
    audio: bytes,
    language: str | None,
) -> TranscriptionResult:
    pcm_audio = await asyncio.to_thread(_realtime_pcm_24khz, audio)
    async with model.get_client().realtime.connect(
        extra_query={"intent": "transcription"}
    ) as connection:
        session_created = asyncio.Event()
        session_updated = asyncio.Event()
        collector = TranscriptCollector()

        async def send(chunk: bytes) -> None:
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(chunk).decode("ascii")
            )

        async with asyncio.TaskGroup() as task_group:
            receiver = task_group.create_task(
                _receive_realtime_transcription(
                    connection,
                    session_created,
                    session_updated,
                    collector,
                )
            )
            await session_created.wait()
            transcription: AudioTranscriptionParam = {"model": model.model_name}
            if language is not None:
                transcription["language"] = language
            session: RealtimeTranscriptionSessionCreateRequestParam = {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24_000,
                        },
                        "transcription": transcription,
                        "turn_detection": None,
                    }
                },
            }
            await connection.session.update(session=session)
            await session_updated.wait()
            await stream_audio_chunks(
                pcm_audio,
                chunk_bytes=_OPENAI_REALTIME_CHUNK_BYTES,
                send=send,
                collector=collector,
            )
            await connection.input_audio_buffer.commit()
    return build_transcription_result(
        text=collector.transcript(),
        audio_bytes=len(audio),
        usage=_normalize_transcription_usage(receiver.result()),
        time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
    )


async def transcribe_audio(
    model: OpenAIModel, request: TranscriptionRequest
) -> TranscriptionResult:
    """Transcribe one bounded audio file through the configured OpenAI model."""
    if "live" in model.model_name:
        return await _transcribe_realtime_audio(
            model,
            audio=request.audio,
            language=request.language,
        )
    if model.supports_streaming_transcription:
        return await _transcribe_file_audio_streaming_response(
            model,
            name=request.name,
            mime=request.mime,
            audio=request.audio,
            language=request.language,
        )
    return await transcribe_file_audio(
        model,
        name=request.name,
        mime=request.mime,
        audio=request.audio,
        language=request.language,
    )
