import asyncio
from dataclasses import dataclass
from typing import cast

from deepgram.client import AsyncDeepgramClient
from deepgram.core.unchecked_base_model import construct_type
from deepgram.environment import (
    DeepgramClientEnvironment,
)
from deepgram.listen.v1.socket_client import AsyncV1SocketClient
from deepgram.listen.v1.types import ListenV1Metadata, ListenV1Results
from deepgram.listen.v2.socket_client import AsyncV2SocketClient
from deepgram.listen.v2.types import ListenV2TurnInfo
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
from model_library.utils import default_httpx_client


@dataclass(frozen=True, slots=True, kw_only=True)
class _DeepgramTranscript:
    text: str
    billable_duration_seconds: float


@register_provider("deepgram")
class DeepgramModel(TranscriptionOnly):
    """Complete-file transcription through Deepgram's Flux or Nova sockets."""

    provider_name = "deepgram"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.DEEPGRAM_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncDeepgramClient:
        if not self.has_client():
            assert api_key is not None
            kwargs: dict[str, object] = {
                "api_key": api_key,
                "timeout": 300.0,
                "max_retries": 0,
                "httpx_client": default_httpx_client(),
            }
            if base_url:
                kwargs["environment"] = DeepgramClientEnvironment(
                    base=base_url,
                    production=base_url,
                    agent=base_url,
                    agent_rest=base_url,
                )
            self.assign_client(AsyncDeepgramClient(**kwargs))
        return cast(AsyncDeepgramClient, super().get_client())

    async def _receive_v2_transcript(
        self,
        stream: AsyncV2SocketClient,
        collector: TranscriptCollector,
    ) -> _DeepgramTranscript:
        turns: dict[int, str] = {}
        open_turns: dict[int, tuple[str, float]] = {}
        billable_duration_seconds = 0.0
        async for response in stream:
            # The SDK yields raw dicts because its response union includes Any.
            if not isinstance(response, dict):
                continue
            event = cast(dict[str, object], response)
            if event.get("type") != "TurnInfo":
                continue
            turn = cast(
                ListenV2TurnInfo,
                construct_type(type_=ListenV2TurnInfo, object_=event),
            )
            transcript = turn.transcript.strip()
            audio_window_seconds = max(
                0.0, turn.audio_window_end - turn.audio_window_start
            )
            collector.observe(transcript, is_final=False)
            if turn.event == "EndOfTurn":
                open_turns.pop(turn.turn_index, None)
                billable_duration_seconds += audio_window_seconds
                if transcript:
                    turns[turn.turn_index] = transcript
            elif transcript:
                open_turns[turn.turn_index] = (transcript, audio_window_seconds)

        # The server may close the stream without ending the in-flight turn.
        for turn_index, (transcript, audio_window_seconds) in open_turns.items():
            turns[turn_index] = transcript
            billable_duration_seconds += audio_window_seconds

        text = " ".join(turns[index] for index in sorted(turns))
        if not text:
            raise ModelNoOutputError(
                "Transcription response did not include a completed turn"
            )
        return _DeepgramTranscript(
            text=text, billable_duration_seconds=billable_duration_seconds
        )

    async def _receive_v1_transcript(
        self,
        stream: AsyncV1SocketClient,
        collector: TranscriptCollector,
    ) -> _DeepgramTranscript:
        finalized_segments: dict[float, str] = {}
        metadata_duration_seconds: float | None = None
        async for response in stream:
            if isinstance(response, ListenV1Metadata):
                metadata_duration_seconds = response.duration
                continue
            if not isinstance(response, ListenV1Results):
                continue

            alternatives = response.channel.alternatives
            transcript = alternatives[0].transcript.strip() if alternatives else ""
            if not transcript:
                continue
            collector.observe(transcript, is_final=False)
            if response.is_final:
                finalized_segments[response.start] = transcript

        text = " ".join(
            finalized_segments[start] for start in sorted(finalized_segments)
        )
        if not text:
            raise ModelNoOutputError(
                "Transcription response did not include a final result"
            )
        if metadata_duration_seconds is None:
            raise RuntimeError(
                "Transcription response did not include terminal metadata"
            )
        return _DeepgramTranscript(
            text=text, billable_duration_seconds=metadata_duration_seconds
        )

    async def _stream_v1(
        self, request: TranscriptionRequest, collector: TranscriptCollector
    ) -> _DeepgramTranscript:
        audio = parse_mono_pcm16_wav(request.audio)
        async with self.get_client().listen.v1.connect(
            model=self.model_name,
            encoding="linear16",
            sample_rate=audio.sample_rate_hz,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=False,
            language=request.language,
        ) as stream:
            async with asyncio.TaskGroup() as task_group:
                receiver = task_group.create_task(
                    self._receive_v1_transcript(stream, collector)
                )
                await stream_audio_chunks(
                    audio.frames,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=stream.send_media,
                    collector=collector,
                )
                await stream.send_finalize()
                await stream.send_close_stream()
            return receiver.result()

    async def _stream_v2(
        self, request: TranscriptionRequest, collector: TranscriptCollector
    ) -> _DeepgramTranscript:
        audio = parse_mono_pcm16_wav(request.audio)
        async with self.get_client().listen.v2.connect(
            model=self.model_name,
            encoding="linear16",
            sample_rate=audio.sample_rate_hz,
        ) as stream:
            async with asyncio.TaskGroup() as task_group:
                receiver = task_group.create_task(
                    self._receive_v2_transcript(stream, collector)
                )
                await stream_audio_chunks(
                    audio.frames,
                    chunk_bytes=PCM16_16KHZ_CHUNK_BYTES,
                    send=stream.send_media,
                    collector=collector,
                )
                await stream.send_close_stream()
            return receiver.result()

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        collector = TranscriptCollector()
        if self.model_name.startswith("flux-"):
            transcript = await self._stream_v2(request, collector)
        else:
            transcript = await self._stream_v1(request, collector)
        return build_transcription_result(
            text=transcript.text,
            audio_bytes=len(request.audio),
            billable_duration_seconds=transcript.billable_duration_seconds,
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
