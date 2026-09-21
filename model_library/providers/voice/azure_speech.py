import asyncio
import threading
from contextlib import suppress

from azure.cognitiveservices.speech import (
    CancellationReason,
    SpeechConfig,
    SpeechRecognitionCanceledEventArgs,
    SpeechRecognitionEventArgs,
    SpeechRecognizer,
)
from azure.cognitiveservices.speech.audio import (
    AudioConfig,
    AudioStreamFormat,
    PushAudioInputStream,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import LLMConfig, TranscriptionOnly, TranscriptionResult
from model_library.base.transcription import (
    TranscriptCollector,
    TranscriptionRequest,
    build_transcription_result,
    parse_mono_pcm16_wav,
)
from model_library.register_models import register_provider

_DEFAULT_AZURE_REGION = "eastus"


@register_provider("azure_speech")
class AzureSpeechModel(TranscriptionOnly):
    """Complete-file transcription through Azure AI Speech continuous recognition."""

    provider_name = "azure_speech"

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.AZURE_SPEECH_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> None:
        return None

    def _recognize(
        self, frames: bytes, sample_rate: int, language: str
    ) -> tuple[str, float | None]:
        """Run one continuous recognition, returning its transcript and TTFP.

        The SDK invokes its callbacks on its own threads, so the results are
        collected here and read once `stopped` is set.
        """
        stream: PushAudioInputStream | None = None
        recognizer: SpeechRecognizer | None = None
        stream_closed = False
        stopped = threading.Event()
        collector = TranscriptCollector()
        failure: BaseException | None = None

        def on_recognizing(event: SpeechRecognitionEventArgs) -> None:
            collector.observe(event.result.text.strip(), is_final=False)

        def on_recognized(event: SpeechRecognitionEventArgs) -> None:
            collector.observe(event.result.text.strip(), is_final=True)

        def on_cancelled(event: SpeechRecognitionCanceledEventArgs) -> None:
            nonlocal failure
            details = event.cancellation_details
            if details.reason != CancellationReason.EndOfStream and failure is None:
                failure = RuntimeError(
                    f"Transcription cancelled: {details.error_details}"
                )
            stopped.set()

        def on_session_stopped(_event: object) -> None:
            stopped.set()

        def stop() -> None:
            if stream is not None and not stream_closed:
                stream.close()
            if recognizer is not None:
                recognizer.stop_continuous_recognition_async().get()

        try:
            speech_config = (
                SpeechConfig(
                    subscription=self._api_key(), endpoint=self.custom_endpoint
                )
                if self.custom_endpoint
                else SpeechConfig(
                    subscription=self._api_key(),
                    region=model_library_settings.get(
                        "AZURE_SPEECH_REGION", _DEFAULT_AZURE_REGION
                    ),
                )
            )
            audio_format = AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16,
                channels=1,
            )
            stream = PushAudioInputStream(stream_format=audio_format)
            audio_config = AudioConfig(stream=stream)
            recognizer = SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
                language=language,
            )
            recognizer.recognizing.connect(  # pyright: ignore[reportUnknownMemberType]
                on_recognizing
            )
            recognizer.recognized.connect(  # pyright: ignore[reportUnknownMemberType]
                on_recognized
            )
            recognizer.canceled.connect(  # pyright: ignore[reportUnknownMemberType]
                on_cancelled
            )
            recognizer.session_stopped.connect(  # pyright: ignore[reportUnknownMemberType]
                on_session_stopped
            )
            recognizer.start_continuous_recognition_async().get()
            collector.audio_started()
            stream.write(frames)
            stream.close()
            stream_closed = True
            stopped.wait()
            if failure is not None:
                raise failure
            transcript = collector.transcript()
        except BaseException:
            with suppress(BaseException):
                stop()
            raise
        stop()
        return transcript, collector.time_to_first_partial_seconds

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        parsed = parse_mono_pcm16_wav(request.audio)
        # Keep the SDK's native threads alive when the asyncio task is cancelled.
        transcript, time_to_first_partial_seconds = await asyncio.shield(
            asyncio.to_thread(
                self._recognize,
                parsed.frames,
                parsed.sample_rate_hz,
                request.language or "en-US",
            )
        )
        return build_transcription_result(
            text=transcript,
            audio_bytes=len(request.audio),
            time_to_first_partial_seconds=time_to_first_partial_seconds,
        )
