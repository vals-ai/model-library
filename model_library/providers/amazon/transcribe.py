# pyright: basic
import asyncio
import json
from typing import cast

from amazon_transcribe.auth import (  # pyright: ignore[reportMissingImports]
    StaticCredentialResolver,
)
from amazon_transcribe.client import (  # pyright: ignore[reportMissingImports]
    TranscribeStreamingClient,
)
from amazon_transcribe.endpoints import (  # pyright: ignore[reportMissingImports]
    StaticEndpointResolver,
)
from amazon_transcribe.model import (  # pyright: ignore[reportMissingImports]
    TranscriptEvent,
    TranscriptResultStream,
)
from boto3 import Session  # pyright: ignore[reportMissingImports]
from botocore.credentials import Credentials  # pyright: ignore[reportMissingImports]
from typing_extensions import override

from model_library.base import LLMConfig, TranscriptionOnly, TranscriptionResult
from model_library.base.transcription import (
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
    stream_audio_chunks,
)
from ._credentials import default_aws_api_key
from model_library.register_models import register_provider

_AWS_STREAM_CHUNK_SECONDS = 0.1
_AWS_MIN_SAMPLE_RATE_HZ = 8_000
_AWS_MAX_SAMPLE_RATE_HZ = 48_000


@register_provider("aws_transcribe")
class AWSTranscribeModel(TranscriptionOnly):
    """Complete-file transcription through the Amazon Transcribe duplex stream."""

    provider_name = "aws_transcribe"

    @override
    def _get_default_api_key(self) -> str:
        return default_aws_api_key()

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    def _region_and_credentials(
        self, api_key: str
    ) -> tuple[str, StaticCredentialResolver]:
        creds: dict[str, str] = (
            {} if api_key == "using-environment" else json.loads(api_key)
        )
        session = Session(
            aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=creds.get("AWS_SESSION_TOKEN"),
            region_name=creds.get("AWS_DEFAULT_REGION"),
        )
        region = cast(str | None, session.region_name)
        if not region:
            raise RuntimeError(
                "A standard region configuration is required for transcription"
            )
        credentials = cast(Credentials | None, session.get_credentials())
        if credentials is None:
            raise RuntimeError(
                "Cloud credentials are required for transcription. "
                "Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
                "or the standard AWS credential chain."
            )

        frozen_credentials = credentials.get_frozen_credentials()
        return region, StaticCredentialResolver(
            access_key_id=frozen_credentials.access_key,
            secret_access_key=frozen_credentials.secret_key,
            session_token=frozen_credentials.token,
        )

    async def _client(self) -> TranscribeStreamingClient:
        region, credential_resolver = await asyncio.to_thread(
            self._region_and_credentials, self._api_key()
        )
        endpoint_resolver = (
            StaticEndpointResolver(self.custom_endpoint)
            if self.custom_endpoint
            else None
        )
        return TranscribeStreamingClient(
            region=region,
            endpoint_resolver=endpoint_resolver,
            credential_resolver=credential_resolver,
        )

    async def _receive_transcript(
        self,
        output_stream: TranscriptResultStream,
        collector: TranscriptCollector,
    ) -> None:
        async for event in output_stream:
            if not isinstance(event, TranscriptEvent):
                continue
            for result in event.transcript.results:
                alternatives = result.alternatives
                if not alternatives:
                    continue
                collector.observe(
                    (alternatives[0].transcript or "").strip(),
                    is_final=not result.is_partial,
                )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(request.audio)
        if (
            not _AWS_MIN_SAMPLE_RATE_HZ
            <= parsed.sample_rate_hz
            <= _AWS_MAX_SAMPLE_RATE_HZ
        ):
            raise ValueError("AWS Transcribe requires 8–48 kHz mono PCM16 WAV audio")
        client = await self._client()
        stream = await client.start_stream_transcription(
            language_code=request.language or "en-US",
            media_sample_rate_hz=parsed.sample_rate_hz,
            media_encoding="pcm",
        )
        collector = TranscriptCollector()

        async def receive() -> None:
            await self._receive_transcript(stream.output_stream, collector)

        async def send(chunk: bytes) -> None:
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(receive())
            await stream_audio_chunks(
                parsed.frames,
                chunk_bytes=int(parsed.sample_rate_hz * 2 * _AWS_STREAM_CHUNK_SECONDS),
                send=send,
                collector=collector,
                max_bytes_per_second=parsed.sample_rate_hz * 2,
            )
            await stream.input_stream.end_stream()
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
