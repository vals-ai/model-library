import asyncio
import io
import logging
from typing import cast

from google.genai import Client
from google.genai.interactions import Interaction
from google.genai.live import AsyncSession
from google.genai.types import (
    ActivityEnd,
    ActivityStart,
    AudioTranscriptionConfig,
    AutomaticActivityDetection,
    Blob,
    LiveConnectConfig,
    Modality,
    RealtimeInputConfig,
    UploadFileConfig,
)
from model_library.exceptions import ModelNoOutputError

from model_library.base import TranscriptionResult
from model_library.base.transcription import (
    PCM16_16KHZ_CHUNK_BYTES,
    NormalizedTranscriptionUsage,
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)

_GOOGLE_FINAL_TIMEOUT_SECONDS = 10.0
_GOOGLE_SAMPLE_RATE_HZ = 16_000
_GOOGLE_LIVE_REALTIME_MULTIPLE = 10
_GOOGLE_LIVE_MAX_BYTES_PER_SECOND = (
    _GOOGLE_SAMPLE_RATE_HZ * 2 * _GOOGLE_LIVE_REALTIME_MULTIPLE
)
_GOOGLE_AUDIO_MIME = f"audio/pcm;rate={_GOOGLE_SAMPLE_RATE_HZ}"

logger = logging.getLogger(__name__)


def _usage(interaction: Interaction) -> NormalizedTranscriptionUsage | None:
    usage = interaction.usage
    if usage is None:
        return None
    return NormalizedTranscriptionUsage(
        input_tokens=usage.total_input_tokens,
        output_tokens=usage.total_output_tokens,
    )


async def transcribe_interactions(
    client: Client, model_name: str, request: TranscriptionRequest
) -> TranscriptionResult:
    """Transcribe one complete audio file through the Interactions API."""
    audio_file = await client.aio.files.upload(
        file=io.BytesIO(request.audio),
        config=UploadFileConfig(mime_type=request.mime),
    )
    try:
        interaction = cast(
            Interaction,
            await client.aio.interactions.create(
                model=model_name,
                input=[
                    {
                        "type": "audio",
                        "uri": audio_file.uri,
                        "mime_type": audio_file.mime_type or request.mime,
                    }
                ],
            ),
        )
        text = interaction.output_text
        if not text or not text.strip():
            raise ModelNoOutputError(
                "Transcription response did not include transcript text"
            )
        result = build_transcription_result(
            text=text,
            audio_bytes=len(request.audio),
            usage=_usage(interaction),
        )
        return result
    finally:
        if audio_file.name is not None:
            try:
                _ = await client.aio.files.delete(name=audio_file.name)
            except Exception:
                logger.warning("Google transcription upload cleanup failed")


async def _receive_live_transcript(
    session: AsyncSession,
    collector: TranscriptCollector,
) -> NormalizedTranscriptionUsage | None:
    usage: NormalizedTranscriptionUsage | None = None
    async for message in session.receive():
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata is not None:
            usage = NormalizedTranscriptionUsage(
                input_tokens=usage_metadata.prompt_token_count,
                output_tokens=usage_metadata.response_token_count,
            )
        server_content = message.server_content
        if server_content is not None:
            interim_transcription = server_content.interim_input_transcription
            if interim_transcription is not None:
                collector.observe(interim_transcription.text or "", is_final=False)
            transcription = server_content.input_transcription
            if transcription is not None:
                collector.observe(transcription.text or "", is_final=True)
                if collector.has_final:
                    # Manual activity detection makes the file one speech turn.
                    break
    return usage


async def transcribe_live(
    client: Client, model_name: str, request: TranscriptionRequest
) -> TranscriptionResult:
    """Transcribe one complete audio file through a throttled Live API session."""
    parsed = parse_mono_pcm16_wav(
        request.audio, required_sample_rate=_GOOGLE_SAMPLE_RATE_HZ
    )
    config = LiveConnectConfig(
        response_modalities=[Modality.TEXT],
        input_audio_transcription=AudioTranscriptionConfig(),
        realtime_input_config=RealtimeInputConfig(
            automatic_activity_detection=AutomaticActivityDetection(disabled=True)
        ),
    )
    async with client.aio.live.connect(model=model_name, config=config) as session:
        collector = TranscriptCollector()
        async with (
            asyncio.timeout(None) as receive_timeout,
            asyncio.TaskGroup() as task_group,
        ):
            receiver = task_group.create_task(
                _receive_live_transcript(session, collector)
            )

            async def send(chunk: bytes) -> None:
                await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                    audio=Blob(data=chunk, mime_type=_GOOGLE_AUDIO_MIME)
                )

            await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                activity_start=ActivityStart()
            )
            await stream_audio_chunks(
                parsed.frames,
                chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                send=send,
                collector=collector,
                max_bytes_per_second=_GOOGLE_LIVE_MAX_BYTES_PER_SECOND,
            )
            await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                activity_end=ActivityEnd()
            )
            if not receiver.done():
                receive_timeout.reschedule(
                    asyncio.get_running_loop().time() + _GOOGLE_FINAL_TIMEOUT_SECONDS
                )

    return build_transcription_result(
        text=collector.transcript(),
        audio_bytes=len(request.audio),
        usage=receiver.result(),
        time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
    )
