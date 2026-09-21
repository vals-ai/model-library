"""Streaming transcription over Meta's realtime ASR socket."""

import asyncio
import json
import math
from collections.abc import AsyncIterable

from pydantic import BaseModel, Field, StrictBool
from websockets.asyncio.client import connect

from model_library.base.output.transcription import TranscriptionResult
from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)

_META_STREAM_ENDPOINT = "wss://api.meta.ai/v1/asr/realtime"
_META_SAMPLE_RATE_HZ = 16_000
# The socket is closed once more than five seconds of audio is queued ahead of
# the engine, so audio has to leave at the rate it was recorded.
_META_MAX_BYTES_PER_SECOND = _META_SAMPLE_RATE_HZ * 2


class MetaTranscriptionEvent(BaseModel):
    # The handshake acknowledgement carries only the session id, no type.
    type: str = ""
    transcript: str = ""
    final: StrictBool = False
    audio_processed_ms: float | None = Field(default=None, alias="audioProcessedMs")
    message: str = ""


async def _receive_transcript(
    websocket: AsyncIterable[str | bytes],
    collector: TranscriptCollector,
) -> float | None:
    """Collect cumulative partials until the single final transcript arrives."""
    async for message in websocket:
        event = MetaTranscriptionEvent.model_validate_json(message)
        if event.type == "error":
            raise RuntimeError(
                f"Transcription failed: {event.message or 'unknown error'}"
            )
        if event.type != "transcript":
            continue
        collector.observe(event.transcript, is_final=event.final)
        if event.final:
            return (
                event.audio_processed_ms / 1_000
                if event.audio_processed_ms is not None
                else None
            )
    return None


async def transcribe_realtime(
    *, api_key: str, model_name: str, request: TranscriptionRequest
) -> TranscriptionResult:
    """Transcribe one complete audio file through the realtime ASR socket."""
    parsed = parse_mono_pcm16_wav(
        request.audio, required_sample_rate=_META_SAMPLE_RATE_HZ
    )
    collector = TranscriptCollector()
    async with connect(_META_STREAM_ENDPOINT, open_timeout=300.0) as websocket:
        async with asyncio.TaskGroup() as task_group:
            receiver = task_group.create_task(_receive_transcript(websocket, collector))
            await websocket.send(
                json.dumps(
                    {
                        "mode": "PUSH_TO_TALK",
                        "authorization": {"accessToken": f"Bearer {api_key}"},
                        "model": model_name,
                        "audioEncoding": "PCM_16KHZ",
                        "partialMode": "CUMULATIVE",
                    }
                )
            )
            await stream_audio_chunks(
                parsed.frames,
                chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                send=websocket.send,
                collector=collector,
                max_bytes_per_second=_META_MAX_BYTES_PER_SECOND,
            )
            await websocket.send(json.dumps({"type": "endStream"}))
    processed_duration_seconds = receiver.result()
    if processed_duration_seconds is None:
        processed_duration_seconds = len(parsed.frames) / _META_MAX_BYTES_PER_SECOND
    return build_transcription_result(
        text=collector.transcript(),
        audio_bytes=len(request.audio),
        billable_duration_seconds=math.floor(processed_duration_seconds),
        time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
    )
