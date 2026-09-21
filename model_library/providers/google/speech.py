import json
from collections.abc import AsyncIterator
from functools import cached_property
from typing import cast

from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from google.cloud.speech_v2 import (
    AutoDetectDecodingConfig,
    RecognitionConfig,
    SpeechAsyncClient,
    StreamingRecognitionConfig,
    StreamingRecognitionFeatures,
    StreamingRecognizeRequest,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import TranscriptionOnly, TranscriptionResult
from model_library.base.transcription import (
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
)
from model_library.register_models import register_provider

_GOOGLE_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_GOOGLE_REGION = "us"
_GOOGLE_STREAM_CHUNK_BYTES = 24 * 1024


@register_provider("google_cloud")
class GoogleCloudModel(TranscriptionOnly):
    """Complete-file transcription through Google Cloud Speech-to-Text v2."""

    provider_name = "google_cloud"

    @override
    def _get_default_api_key(self) -> str:
        return json.dumps(
            {
                "GCP_PROJECT_ID": model_library_settings.GCP_PROJECT_ID,
                "GCP_CREDS": model_library_settings.GCP_CREDS,
            }
        )

    @cached_property
    def _credentials(self) -> tuple[str, Credentials]:
        creds = json.loads(self._api_key())
        return (
            creds["GCP_PROJECT_ID"],
            service_account.Credentials.from_service_account_info(  # pyright: ignore[reportUnknownMemberType]
                json.loads(creds["GCP_CREDS"]),
                scopes=[_GOOGLE_SCOPE],
            ),
        )

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> SpeechAsyncClient:
        if not self.has_client():
            assert api_key is not None
            _, credentials = self._credentials
            endpoint = base_url or f"{_GOOGLE_REGION}-speech.googleapis.com"
            self.assign_client(
                SpeechAsyncClient(
                    credentials=credentials,
                    client_options=ClientOptions(api_endpoint=endpoint),
                )
            )
        return cast(SpeechAsyncClient, super().get_client())

    @staticmethod
    def _streaming_config(language_code: str) -> StreamingRecognitionConfig:
        return StreamingRecognitionConfig(
            config=RecognitionConfig(
                auto_decoding_config=AutoDetectDecodingConfig(),
                language_codes=[language_code],
                model="chirp_3",
            ),
            streaming_features=StreamingRecognitionFeatures(interim_results=True),
        )

    async def _stream_requests(
        self,
        recognizer: str,
        config: StreamingRecognitionConfig,
        audio: bytes,
        collector: TranscriptCollector,
    ) -> AsyncIterator[StreamingRecognizeRequest]:
        yield StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=config,
        )
        collector.audio_started()
        for offset in range(0, len(audio), _GOOGLE_STREAM_CHUNK_BYTES):
            yield StreamingRecognizeRequest(
                audio=audio[offset : offset + _GOOGLE_STREAM_CHUNK_BYTES]
            )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        client = self.get_client()
        project_id, _ = self._credentials
        recognizer = f"projects/{project_id}/locations/{_GOOGLE_REGION}/recognizers/_"
        collector = TranscriptCollector()
        # Chirp auto-detects the container, so the full WAV is sent as-is.
        responses = await client.streaming_recognize(  # pyright: ignore[reportUnknownMemberType]
            requests=self._stream_requests(
                recognizer,
                self._streaming_config(request.language or "en-US"),
                request.audio,
                collector,
            ),
            timeout=300.0,
        )
        billable_duration_seconds: float | None = None
        async for response in responses:
            metadata = getattr(response, "metadata", None)
            billed_duration = getattr(metadata, "total_billed_duration", None)
            if billed_duration is not None:
                billed_seconds = float(getattr(billed_duration, "seconds", 0)) + (
                    float(getattr(billed_duration, "nanos", 0)) / 1_000_000_000
                )
                if billed_seconds > 0:
                    billable_duration_seconds = billed_seconds
            for result in response.results:
                if not result.alternatives:
                    continue
                collector.observe(
                    result.alternatives[0].transcript.strip(),
                    is_final=result.is_final,
                )
        return build_transcription_result(
            text=collector.transcript(),
            audio_bytes=len(request.audio),
            billable_duration_seconds=billable_duration_seconds,
            time_to_first_partial_seconds=collector.time_to_first_partial_seconds,
        )
