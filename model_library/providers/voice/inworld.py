import asyncio
import base64
import json
from collections.abc import AsyncIterable
from typing import cast

from pydantic import BaseModel, Field, StrictBool
from typing_extensions import override
from websockets.asyncio.client import connect

from model_library import model_library_settings
from model_library.base import LLMConfig, TranscriptionOnly, TranscriptionResult
from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)
from model_library.register_models import register_provider

_INWORLD_STREAM_ENDPOINT = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
_INWORLD_SAMPLE_RATE = 16_000


class InworldTranscription(BaseModel):
    is_final: StrictBool = Field(alias="isFinal")
    transcript: str = ""


class InworldUsage(BaseModel):
    transcribed_audio_ms: float = Field(alias="transcribedAudioMs")


class InworldResult(BaseModel):
    transcription: InworldTranscription | None = None
    usage: InworldUsage | None = None


class InworldResponse(BaseModel):
    result: InworldResult | None = None
    error: dict[str, object] | None = None


@register_provider("inworld")
class InworldModel(TranscriptionOnly):
    """Complete-file transcription through Inworld's bidirectional STT stream."""

    provider_name = "inworld"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.INWORLD_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    @staticmethod
    def _event_message(event: dict[str, object]) -> str:
        return json.dumps(event, separators=(",", ":"))

    async def _receive_transcript(
        self,
        websocket: AsyncIterable[str | bytes],
        collector: TranscriptCollector,
    ) -> float:
        billable_duration_seconds: float | None = None
        async for message in websocket:
            response = InworldResponse.model_validate(
                cast(dict[str, object], json.loads(message))
            )
            if response.error is not None:
                detail = response.error.get("message", "unknown error")
                raise RuntimeError(f"Transcription failed: {detail}")
            result = response.result
            if result is None:
                continue
            if result.usage is not None:
                billable_duration_seconds = result.usage.transcribed_audio_ms / 1_000
            if result.transcription is not None:
                collector.observe(
                    result.transcription.transcript.strip(),
                    is_final=result.transcription.is_final,
                )
            # Inworld holds the socket open after the last turn, so stop reading
            # once both the final transcript and its usage have arrived.
            if collector.has_final and billable_duration_seconds is not None:
                break

        if billable_duration_seconds is None:
            raise RuntimeError("Transcription response did not include terminal usage")
        return billable_duration_seconds

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(
            request.audio,
            required_sample_rate=_INWORLD_SAMPLE_RATE,
        )
        endpoint = self.custom_endpoint or _INWORLD_STREAM_ENDPOINT
        headers = {"Authorization": f"Basic {self._api_key()}"}
        async with connect(
            endpoint,
            additional_headers=headers,
            open_timeout=300.0,
        ) as websocket:
            collector = TranscriptCollector()

            async def send(chunk: bytes) -> None:
                content = base64.b64encode(chunk).decode("ascii")
                await websocket.send(
                    self._event_message({"audioChunk": {"content": content}})
                )

            receiver: asyncio.Task[float] | None = None
            try:
                async with asyncio.TaskGroup() as task_group:
                    receiver = task_group.create_task(
                        self._receive_transcript(websocket, collector)
                    )
                    await websocket.send(
                        self._event_message(
                            {
                                "transcribeConfig": {
                                    "modelId": self.model_name,
                                    "audioEncoding": "LINEAR16",
                                    "sampleRateHertz": _INWORLD_SAMPLE_RATE,
                                    "numberOfChannels": 1,
                                    "language": request.language,
                                }
                            }
                        )
                    )
                    await stream_audio_chunks(
                        parsed.frames,
                        chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                        send=send,
                        collector=collector,
                    )
                    await websocket.send(self._event_message({"endTurn": {}}))
                    await websocket.send(self._event_message({"closeStream": {}}))
            except ExceptionGroup as group:
                if (
                    receiver is not None
                    and receiver.done()
                    and not receiver.cancelled()
                ):
                    receiver_error = receiver.exception()
                    if receiver_error is not None:
                        raise receiver_error from group
                raise
            assert receiver is not None
            billable_duration_seconds = receiver.result()
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            billable_duration_seconds=billable_duration_seconds,
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
