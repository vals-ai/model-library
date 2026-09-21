import asyncio
import json
from collections.abc import AsyncIterable, Mapping
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


_XAI_STT_ENDPOINT = "wss://api.x.ai/v1/stt"
_XAI_STT_SAMPLE_RATE = 16_000
_XAI_STT_MAX_BYTES_PER_SECOND = _XAI_STT_SAMPLE_RATE * 2


def _add_query_params(url: str, params: Mapping[str, str]) -> str:
    """Merge query parameters into a URL, overriding existing keys."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


@register_provider("xai")
class XAITranscriptionModel(TranscriptionOnly):
    provider_name = "xai"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.XAI_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    async def _receive_transcript(
        self,
        websocket: AsyncIterable[str | bytes],
        ready: asyncio.Event,
        collector: TranscriptCollector,
    ) -> str | None:
        pending_chunks: list[str] = []
        completed_utterances: list[str] = []
        try:
            async for message in websocket:
                payload = cast(dict[str, object], json.loads(message))
                event_type = payload.get("type")
                if event_type == "transcript.created":
                    ready.set()
                    continue
                if event_type == "error":
                    detail = payload.get("message")
                    message_text = (
                        detail if isinstance(detail, str) else "unknown error"
                    )
                    raise RuntimeError(f"Transcription failed: {message_text}")
                if event_type == "transcript.done":
                    text = payload.get("text")
                    done_text = text.strip() if isinstance(text, str) else ""
                    if done_text:
                        return done_text
                    break
                if event_type != "transcript.partial":
                    continue
                text = payload.get("text")
                transcript = text.strip() if isinstance(text, str) else ""
                collector.observe(transcript, is_final=False)
                if payload.get("speech_final") is True:
                    if transcript:
                        completed_utterances.append(transcript)
                        pending_chunks.clear()
                elif payload.get("is_final") is True and transcript:
                    pending_chunks.append(transcript)

            for transcript in (*completed_utterances, *pending_chunks):
                collector.observe(transcript, is_final=True)
            return None
        finally:
            ready.set()

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(
            request.audio, required_sample_rate=_XAI_STT_SAMPLE_RATE
        )
        params = {
            "model": self.model_name,
            "sample_rate": "16000",
            "encoding": "pcm",
            "interim_results": "true",
        }
        if request.language is not None:
            params["language"] = request.language
        endpoint = _add_query_params(
            self.custom_endpoint or _XAI_STT_ENDPOINT,
            params,
        )
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        async with connect(
            endpoint,
            additional_headers=headers,
            open_timeout=300.0,
        ) as websocket:
            ready = asyncio.Event()
            collector = TranscriptCollector()
            async with asyncio.TaskGroup() as task_group:
                receiver = task_group.create_task(
                    self._receive_transcript(websocket, ready, collector)
                )
                await ready.wait()
                if not receiver.done():
                    await stream_audio_chunks(
                        parsed.frames,
                        chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                        send=websocket.send,
                        collector=collector,
                        max_bytes_per_second=_XAI_STT_MAX_BYTES_PER_SECOND,
                    )
                    await websocket.send(json.dumps({"type": "audio.done"}))
        done_text = receiver.result()
        return build_transcription_result(
            text=done_text if done_text is not None else collector.transcript(),
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
