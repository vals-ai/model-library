from __future__ import annotations

import asyncio
from typing import cast

from mistralai.client import Mistral  # pyright: ignore[reportMissingImports]
from mistralai.client.models import (  # pyright: ignore[reportMissingImports]
    AudioFormat,
    RealtimeTranscriptionError,
    RealtimeTranscriptionSessionUpdated,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)
from mistralai.extra.realtime import (  # pyright: ignore[reportMissingImports]
    RealtimeConnection,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import TranscriptionOnly, TranscriptionResult
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


@register_provider("mistral")
class MistralTranscriptionModel(TranscriptionOnly):
    """Complete-file transcription through Voxtral batch or realtime APIs."""

    provider_name = "mistral"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.MISTRAL_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Mistral:
        if not self.has_client():
            assert api_key is not None
            self.assign_client(
                Mistral(
                    api_key=api_key,
                    async_client=default_httpx_client(),
                    server_url=base_url,
                    timeout_ms=300_000,
                )
            )
        return cast(Mistral, super().get_client())

    async def _receive_transcript(
        self,
        connection: RealtimeConnection,
        session_updated: asyncio.Event,
        collector: TranscriptCollector,
    ) -> None:
        async for event in connection:
            if isinstance(event, RealtimeTranscriptionSessionUpdated):
                session_updated.set()
                continue
            if isinstance(event, RealtimeTranscriptionError):
                detail = event.error.message
                message = detail if isinstance(detail, str) else "unknown error"
                raise RuntimeError(f"Transcription failed: {message}")
            if isinstance(event, TranscriptionStreamTextDelta):
                collector.observe(event.text, is_final=False)
                continue
            if isinstance(event, TranscriptionStreamDone):
                collector.observe(event.text, is_final=True)
                break

    async def _transcribe_offline(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        client = self.get_client()
        response = await client.audio.transcriptions.complete_async(
            model=self.model_name,
            file={"file_name": request.name, "content": request.audio},
            language=request.language,
        )
        if not response.text:
            raise ModelNoOutputError(
                "Transcription response did not include transcript text"
            )
        return build_transcription_result(
            text=response.text,
            audio_bytes=len(request.audio),
        )

    async def _transcribe_realtime(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(request.audio, required_sample_rate=16_000)
        connection = await self.get_client().audio.realtime.connect(
            model=self.model_name,
            audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16_000),
            server_url=self.custom_endpoint,
            timeout_ms=300_000,
        )
        try:
            session_updated = asyncio.Event()
            collector = TranscriptCollector()
            async with asyncio.TaskGroup() as task_group:
                _ = task_group.create_task(
                    self._receive_transcript(
                        connection,
                        session_updated,
                        collector,
                    )
                )
                await session_updated.wait()
                await stream_audio_chunks(
                    parsed.frames,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=connection.send_audio,
                    collector=collector,
                )
                await connection.flush_audio()
                await connection.end_audio()
        finally:
            await connection.close()
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        if "realtime" in self.model_name:
            return await self._transcribe_realtime(request)
        return await self._transcribe_offline(request)
