"""Unit tests for Azure AI Speech transcription.

Run: uv run pytest tests/unit/providers/test_azure_speech.py
Covers region and endpoint selection, buffered final transcripts, and TTFP.
"""

import asyncio
import threading
from collections.abc import Callable
from types import SimpleNamespace

import azure.cognitiveservices.speech as speechsdk  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
from pydantic import SecretStr  # pyright: ignore[reportMissingImports]

from examples.data.audio import tone_wav
from model_library.base import LLMConfig, TranscriptionResult
from model_library.providers.voice import azure_speech
from model_library.providers.voice.azure_speech import AzureSpeechModel


def _model(*, custom_endpoint: str | None = None) -> AzureSpeechModel:
    return AzureSpeechModel(
        "universal-language-model",
        config=LLMConfig(
            supports_transcription=True,
            custom_api_key=SecretStr("azure-key"),
            custom_endpoint=custom_endpoint,
            registry_key="azure_speech/universal-language-model",
        ),
    )


async def _transcribe(model: AzureSpeechModel) -> TranscriptionResult:
    return await model.transcribe_audio(
        name="audio.wav",
        mime="audio/wav",
        audio=tone_wav(),
        language="en-US",
    )


def _patch_sdk(
    monkeypatch: pytest.MonkeyPatch,
    recognizer: object,
    stream: object,
    *,
    region: str | None = None,
    speech_config_calls: list[dict[str, str]] | None = None,
) -> None:
    monkeypatch.setattr(
        azure_speech,
        "model_library_settings",
        SimpleNamespace(
            get=lambda name, default: (
                region
                if name == "AZURE_SPEECH_REGION" and region is not None
                else default
            )
        ),
    )

    def speech_config(**kwargs: str) -> object:
        if speech_config_calls is not None:
            speech_config_calls.append(kwargs)
        return object()

    monkeypatch.setattr(azure_speech, "SpeechConfig", speech_config)
    monkeypatch.setattr(azure_speech, "AudioStreamFormat", lambda **_kwargs: object())
    monkeypatch.setattr(azure_speech, "PushAudioInputStream", lambda **_kwargs: stream)
    monkeypatch.setattr(azure_speech, "AudioConfig", lambda **_kwargs: object())
    monkeypatch.setattr(azure_speech, "SpeechRecognizer", lambda **_kwargs: recognizer)


class MockSignal:
    """Callback registration and emission for a mocked Azure SDK signal."""

    def __init__(self) -> None:
        self.callback: Callable[[object], None] | None = None

    def connect(self, callback: Callable[[object], None]) -> None:
        self.callback = callback

    def emit(self, event: object) -> None:
        assert self.callback is not None
        self.callback(event)


class MockOperation:
    """Synchronous Azure SDK operation wrapper."""

    def get(self) -> None:
        return None


class MockResult:
    """Recognized text carried by a mocked Azure callback."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockEvent:
    """Azure callback event containing a recognition result."""

    def __init__(self, text: str) -> None:
        self.result = MockResult(text)


class MockCancellationDetails:
    """Cancellation details carried by a mocked Azure callback."""

    def __init__(self, reason: speechsdk.CancellationReason, error: str = "") -> None:
        self.reason = reason
        self.error_details = error


class MockCancellationEvent:
    """Azure callback event containing cancellation details."""

    def __init__(self, reason: speechsdk.CancellationReason, error: str = "") -> None:
        self.cancellation_details = MockCancellationDetails(reason, error)


class MockRecognizer:
    """Continuous recognizer that emits partial and final events."""

    def __init__(self) -> None:
        self.recognizing = MockSignal()
        self.recognized = MockSignal()
        self.canceled = MockSignal()
        self.session_stopped = MockSignal()
        self.started = False
        self.stopped = False
        self.stop_count = 0
        self.stop_finished = threading.Event()
        self.events: list[str] = []

    def start_continuous_recognition_async(self) -> MockOperation:
        self.started = True
        return MockOperation()

    def stop_continuous_recognition_async(self) -> MockOperation:
        self.stopped = True
        self.stop_count += 1
        self.events.append("recognition_stopped")
        self.stop_finished.set()
        return MockOperation()


class PassivePushStream:
    """Push stream that only records cleanup."""

    def __init__(self) -> None:
        self.closed = False
        self.close_count = 0

    def write(self, _chunk: bytes) -> None:
        return None

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class MockPushStream:
    """Push stream that emits Azure events as audio is closed."""

    def __init__(self, recognizer: MockRecognizer) -> None:
        self.recognizer = recognizer
        self.chunks: list[bytes] = []
        self.closed = False
        self.close_count = 0

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        self.recognizer.recognizing.emit(MockEvent("hello"))

    def close(self) -> None:
        self.close_count += 1
        if not self.closed:
            self.closed = True
            self.recognizer.events.append("stream_closed")
            self.recognizer.recognized.emit(MockEvent("hello world"))
            self.recognizer.events.append("first_final")
            self.recognizer.recognized.emit(MockEvent("from Azure"))
            self.recognizer.events.append("second_final")
            self.recognizer.session_stopped.emit(MockEvent(""))
            self.recognizer.events.append("session_stopped")


class BlockingWritePushStream(MockPushStream):
    """Push stream whose write waits for test release."""

    def __init__(
        self,
        recognizer: MockRecognizer,
        write_started: threading.Event,
        release_write: threading.Event,
    ) -> None:
        super().__init__(recognizer)
        self.write_started = write_started
        self.release_write = release_write

    def write(self, chunk: bytes) -> None:
        self.write_started.set()
        assert self.release_write.wait(2)
        super().write(chunk)


@pytest.mark.parametrize(
    ("region", "custom_endpoint", "expected_speech_config"),
    [
        (
            "westus3",
            None,
            {"subscription": "azure-key", "region": "westus3"},
        ),
        (
            None,
            None,
            {"subscription": "azure-key", "region": "eastus"},
        ),
        (
            "westus3",
            "https://speech.example.test/",
            {
                "subscription": "azure-key",
                "endpoint": "https://speech.example.test/",
            },
        ),
    ],
)
@pytest.mark.unit
async def test_azure_speech_selects_region_or_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    region: str | None,
    custom_endpoint: str | None,
    expected_speech_config: dict[str, str],
) -> None:
    recognizer = MockRecognizer()
    stream = MockPushStream(recognizer)
    speech_config_calls: list[dict[str, str]] = []
    model = _model(custom_endpoint=custom_endpoint)
    _patch_sdk(
        monkeypatch,
        recognizer,
        stream,
        region=region,
        speech_config_calls=speech_config_calls,
    )

    result = await _transcribe(model)

    assert result.text == "hello world from Azure"
    assert speech_config_calls == [expected_speech_config]


@pytest.mark.unit
async def test_azure_speech_drains_final_segments_before_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer = MockRecognizer()
    stream = MockPushStream(recognizer)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    result = await _transcribe(model)

    assert result.text == "hello world from Azure"
    assert result.metadata.time_to_first_partial_seconds is not None
    assert stream.chunks
    assert stream.closed
    assert recognizer.started
    assert recognizer.stopped
    assert stream.close_count == 1
    assert recognizer.stop_count == 1
    assert recognizer.events == [
        "stream_closed",
        "first_final",
        "second_final",
        "session_stopped",
        "recognition_stopped",
    ]


class EndOfStreamPushStream(MockPushStream):
    def close(self) -> None:
        self.close_count += 1
        if not self.closed:
            self.closed = True
            self.recognizer.events.append("stream_closed")
            self.recognizer.recognized.emit(MockEvent("complete transcript"))
            self.recognizer.events.append("final")
            self.recognizer.canceled.emit(
                MockCancellationEvent(speechsdk.CancellationReason.EndOfStream)
            )
            self.recognizer.events.append("end_of_stream")


@pytest.mark.unit
async def test_azure_speech_end_of_stream_cancellation_completes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer = MockRecognizer()
    stream = EndOfStreamPushStream(recognizer)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    result = await _transcribe(model)

    assert result.text == "complete transcript"
    assert recognizer.events == [
        "stream_closed",
        "final",
        "end_of_stream",
        "recognition_stopped",
    ]
    assert stream.close_count == 1
    assert recognizer.stop_count == 1


class ErrorPushStream(MockPushStream):
    def close(self) -> None:
        self.close_count += 1
        if not self.closed:
            self.closed = True
            self.recognizer.events.append("stream_closed")
            self.recognizer.canceled.emit(
                MockCancellationEvent(
                    speechsdk.CancellationReason.Error, "authentication failed"
                )
            )
            self.recognizer.events.append("error")


@pytest.mark.unit
async def test_azure_speech_real_cancellation_error_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer = MockRecognizer()
    stream = ErrorPushStream(recognizer)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    with pytest.raises(RuntimeError, match="authentication failed"):
        await _transcribe(model)

    assert recognizer.events == [
        "stream_closed",
        "error",
        "recognition_stopped",
    ]
    assert stream.close_count == 1
    assert recognizer.stop_count == 1


class FailingOperation(MockOperation):
    """SDK operation that fails synchronously when joined."""

    def get(self) -> None:
        raise RuntimeError("startup failed")


class FailingStartRecognizer(MockRecognizer):
    """Recognizer whose continuous-recognition startup fails."""

    def start_continuous_recognition_async(self) -> MockOperation:
        return FailingOperation()


@pytest.mark.unit
async def test_azure_speech_startup_failure_closes_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a failed SDK startup cannot leak its push stream."""
    recognizer = FailingStartRecognizer()
    stream = PassivePushStream()
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    with pytest.raises(RuntimeError, match="startup failed"):
        await _transcribe(model)

    assert stream.closed
    assert stream.close_count == 1
    assert recognizer.stop_count == 1


@pytest.mark.unit
async def test_azure_speech_cancellation_leaves_worker_cleanup_to_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cancellation does not interrupt the worker's native operation."""
    write_started = threading.Event()
    release_write = threading.Event()
    recognizer = MockRecognizer()
    stream = BlockingWritePushStream(recognizer, write_started, release_write)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)
    transcription = asyncio.create_task(
        model.transcribe_audio(
            name="audio.wav", mime="audio/wav", audio=tone_wav(), language="en-US"
        )
    )

    try:
        assert await asyncio.to_thread(write_started.wait, 1)
        transcription.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transcription
    finally:
        release_write.set()

    assert await asyncio.to_thread(recognizer.stop_finished.wait, 1)
    assert stream.closed
    assert stream.close_count == 1
    assert recognizer.stop_count == 1


class StopFailureOperation(MockOperation):
    """SDK stop operation that fails when joined."""

    def get(self) -> None:
        raise RuntimeError("stop failed")


class FailingStopRecognizer(MockRecognizer):
    """Recognizer whose cleanup stop operation fails."""

    def stop_continuous_recognition_async(self) -> MockOperation:
        self.stopped = True
        self.stop_count += 1
        return StopFailureOperation()


@pytest.mark.unit
async def test_azure_speech_recognition_failure_precedes_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cleanup cannot mask Azure's recognition error."""
    recognizer = FailingStopRecognizer()
    stream = ErrorPushStream(recognizer)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    with pytest.raises(RuntimeError, match="authentication failed"):
        await _transcribe(model)

    assert stream.close_count == 1
    assert recognizer.stop_count == 1


@pytest.mark.unit
async def test_azure_speech_stop_failure_surfaces_without_prior_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify an otherwise successful request preserves its stop failure."""
    recognizer = FailingStopRecognizer()
    stream = MockPushStream(recognizer)
    model = _model()
    _patch_sdk(monkeypatch, recognizer, stream)

    with pytest.raises(RuntimeError, match="stop failed"):
        await _transcribe(model)

    assert stream.close_count == 1
    assert recognizer.stop_count == 1
