import asyncio
import json
from collections.abc import AsyncIterable
from urllib.parse import urlencode

from pydantic import BaseModel, StrictBool
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
from model_library.exceptions import ModelNoOutputError
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client

_RESON8_API_BASE_URL = "https://api.reson8.dev"
_RESON8_PRERECORDED_PATH = "/v1/speech-to-text/prerecorded"
_RESON8_REALTIME_PATH = "/v1/speech-to-text/realtime"
_RESON8_SAMPLE_RATE = 16_000
_RESON8_TERMINAL_TIMEOUT_SECONDS = 60.0


class Reson8Prerecorded(BaseModel):
    text: str


class Reson8RealtimeMessage(BaseModel):
    type: str
    text: str = ""
    is_final: StrictBool = True


@register_provider("reson8")
class Reson8Model(TranscriptionOnly):
    """Complete-file transcription through Reson8's prerecorded or realtime API.

    Reson8 serves one server-side model per endpoint, so the model name selects
    the endpoint: names ending in ``-realtime`` stream over WebSocket.
    """

    provider_name = "reson8"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.RESON8_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"ApiKey {self._api_key()}"}

    def _url(self, path: str, params: dict[str, str]) -> str:
        base = (self.custom_endpoint or _RESON8_API_BASE_URL).rstrip("/")
        query = f"?{urlencode(params)}" if params else ""
        return f"{base}{path}{query}"

    async def _transcribe_prerecorded(self, request: TranscriptionRequest) -> str:
        params = {"language": request.language} if request.language else {}
        async with default_httpx_client() as client:
            response = await client.post(
                self._url(_RESON8_PRERECORDED_PATH, params),
                headers={
                    **self._headers(),
                    "Content-Type": "application/octet-stream",
                },
                content=request.audio,
            )
        response.raise_for_status()
        text = Reson8Prerecorded.model_validate_json(response.content).text.strip()
        if not text:
            raise ModelNoOutputError(
                "Transcription response did not include transcript text"
            )

        return text

    @staticmethod
    async def _receive_transcript(
        websocket: AsyncIterable[str | bytes],
        collector: TranscriptCollector,
    ) -> None:
        async for raw in websocket:
            message = Reson8RealtimeMessage.model_validate_json(raw)
            if message.type == "flush_confirmation":
                return
            if message.type == "transcript":
                collector.observe(message.text, is_final=message.is_final)

    async def _transcribe_realtime(
        self, request: TranscriptionRequest, collector: TranscriptCollector
    ) -> None:
        parsed = parse_mono_pcm16_wav(
            request.audio, required_sample_rate=_RESON8_SAMPLE_RATE
        )
        params = {
            "encoding": "pcm_s16le",
            "sample_rate": str(_RESON8_SAMPLE_RATE),
            "channels": "1",
            "include_interim": "true",
        }
        if request.language:
            params["language"] = request.language
        url = self._url(_RESON8_REALTIME_PATH, params).replace("http", "ws", 1)
        async with connect(
            url, additional_headers=self._headers(), open_timeout=300.0
        ) as websocket:
            async with asyncio.TaskGroup() as task_group:
                receiver = task_group.create_task(
                    self._receive_transcript(websocket, collector)
                )
                await stream_audio_chunks(
                    parsed.frames,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=websocket.send,
                    collector=collector,
                )
                await websocket.send(json.dumps({"type": "flush_request"}))
                await asyncio.wait_for(receiver, _RESON8_TERMINAL_TIMEOUT_SECONDS)

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        if not self.model_name.endswith("-realtime"):
            return build_transcription_result(
                text=await self._transcribe_prerecorded(request),
                audio_bytes=len(request.audio),
            )
        collector = TranscriptCollector()
        await self._transcribe_realtime(request, collector)
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
