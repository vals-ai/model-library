import asyncio
from typing import Literal, cast

from cartesia import AsyncCartesia
from cartesia.resources.stt.manual_finalize import (
    AsyncManualFinalizeResourceConnection,
)
from cartesia.types import STTErrorResponse
from cartesia.types.stt import (
    STTManualFinalizeFlushDoneResponse,
    STTManualFinalizeTranscriptResponse,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import TranscriptionOnly, TranscriptionResult
from model_library.exceptions import ModelNoOutputError
from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)
from model_library.register_models import register_provider

_CARTESIA_SAMPLE_RATE = 16_000


@register_provider("cartesia")
class CartesiaModel(TranscriptionOnly):
    """Complete-file transcription using Cartesia Ink 2's manual-finalize stream."""

    provider_name = "cartesia"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.CARTESIA_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncCartesia:
        if not self.has_client():
            assert api_key is not None
            self.assign_client(
                AsyncCartesia(api_key=api_key, websocket_base_url=base_url)
            )
        return cast(AsyncCartesia, super().get_client())

    @staticmethod
    async def _receive_transcript(
        connection: AsyncManualFinalizeResourceConnection,
        collector: TranscriptCollector,
        final_segments: list[str],
        flush_done: asyncio.Event,
    ) -> None:
        try:
            async for event in connection:
                if isinstance(event, STTErrorResponse):
                    raise RuntimeError(f"Transcription failed: {event.message}")
                if isinstance(event, STTManualFinalizeFlushDoneResponse):
                    flush_done.set()
                    continue
                if not isinstance(event, STTManualFinalizeTranscriptResponse):
                    continue
                collector.observe(event.text, is_final=False)
                if event.is_final:
                    final_segments.append(event.text)
        finally:
            flush_done.set()

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(
            request.audio,
            required_sample_rate=_CARTESIA_SAMPLE_RATE,
        )
        async with self.get_client().stt.manual_finalize.websocket(
            encoding="pcm_s16le",
            model=self.model_name,
            sample_rate=_CARTESIA_SAMPLE_RATE,
            language=cast(Literal["en"], request.language),
        ) as connection:
            collector = TranscriptCollector()
            final_segments: list[str] = []
            flush_done = asyncio.Event()

            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(
                    self._receive_transcript(
                        connection, collector, final_segments, flush_done
                    )
                )
                await stream_audio_chunks(
                    parsed.frames,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=connection.send_raw,
                    collector=collector,
                )
                await connection.send("finalize")
                await flush_done.wait()
                await connection.send("close")
        text = "".join(final_segments)
        if not text:
            raise ModelNoOutputError(
                "Transcription response did not include a final transcript"
            )
        return build_transcription_result(
            text=text,
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
