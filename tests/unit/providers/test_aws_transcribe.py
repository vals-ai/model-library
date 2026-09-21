"""Tests for the Amazon Transcribe Streaming SDK boundary."""

import asyncio
import io
import json
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from amazon_transcribe.auth import StaticCredentialResolver
from amazon_transcribe.endpoints import StaticEndpointResolver
from amazon_transcribe.model import (
    Alternative,
    Result,
    Transcript,
    TranscriptEvent,
)

from tests.unit.providers.transcription_test_support import (
    ScriptedStream as MockOutputStream,
)
from examples.data.audio import tone_wav
from model_library.base import LLMConfig
from model_library.providers.amazon import transcribe as amazon
from model_library.providers.amazon.transcribe import AWSTranscribeModel


class MockInputStream:
    def __init__(self) -> None:
        self.events: list[bytes] = []
        self.ended = False

    async def send_audio_event(self, *, audio_chunk: bytes) -> None:
        self.events.append(audio_chunk)

    async def end_stream(self) -> None:
        self.ended = True


class MockDuplexStream:
    def __init__(
        self, output_stream: MockOutputStream, input_stream: MockInputStream
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream


class MockClient:
    def __init__(self, stream: MockDuplexStream) -> None:
        self.stream = stream
        self.inputs: list[dict[str, object]] = []

    async def start_stream_transcription(self, **kwargs: object) -> MockDuplexStream:
        self.inputs.append(kwargs)
        return self.stream


def test_aws_credentials_use_bedrock_style_credentials_json() -> None:
    api_key = json.dumps(
        {
            "AWS_ACCESS_KEY_ID": "settings-key",
            "AWS_SECRET_ACCESS_KEY": "settings-secret",
            "AWS_SESSION_TOKEN": "settings-token",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )

    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(
            custom_endpoint="https://transcribe.example.com",
            registry_key="aws_transcribe/streaming",
        ),
    )

    region, resolver = model._region_and_credentials(  # pyright: ignore[reportPrivateUsage]
        api_key
    )

    assert region == "us-east-1"
    credentials = asyncio.run(resolver.get_credentials())
    assert credentials is not None
    assert credentials.access_key_id == "settings-key"
    assert credentials.secret_access_key == "settings-secret"
    assert credentials.session_token == "settings-token"


def test_aws_credentials_fall_back_to_default_chain() -> None:
    credentials = MagicMock()
    credentials.get_frozen_credentials.return_value = MagicMock(
        access_key="access-key",
        secret_key="secret-key",
        token=None,
    )
    session = MagicMock(region_name="profile-or-default-region")
    session.get_credentials.return_value = credentials

    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(registry_key="aws_transcribe/streaming"),
    )

    with patch.object(amazon, "Session", MagicMock(return_value=session)):
        region, resolver = model._region_and_credentials(  # pyright: ignore[reportPrivateUsage]
            "using-environment"
        )

    assert region == "profile-or-default-region"
    resolved = asyncio.run(resolver.get_credentials())
    assert resolved is not None
    assert resolved.access_key_id == "access-key"
    assert resolved.session_token is None


def _event(text: str, *, is_partial: bool) -> TranscriptEvent:
    return TranscriptEvent(
        Transcript(
            results=[
                Result(
                    alternatives=[Alternative(text, [], None)],
                    is_partial=is_partial,
                )
            ]
        )
    )


async def test_aws_transcribe_streaming_sdk_contract() -> None:
    audio = tone_wav(duration_seconds=2.5)
    input_stream = MockInputStream()
    output_stream = MockOutputStream(
        [_event("hello", is_partial=True), _event("hello world", is_partial=False)]
    )
    stream = MockDuplexStream(output_stream, input_stream)
    client = MockClient(stream)
    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(
            supports_transcription=True, registry_key="aws_transcribe/streaming"
        ),
    )

    with (
        patch.object(model, "_client", AsyncMock(return_value=client)),
        patch(
            "model_library.providers.amazon.transcribe.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch(
            "model_library.base.base.perf_counter",
            side_effect=[0.0, 1.0],
        ),
    ):
        result = await asyncio.wait_for(
            model.transcribe_audio(
                name="audio.wav",
                mime="audio/wav",
                audio=audio,
                language="en",
            ),
            timeout=1.0,
        )

    with wave.open(io.BytesIO(audio), "rb") as source:
        expected_audio = source.readframes(source.getnframes())
        expected_sample_rate = source.getframerate()

    sent_audio = b"".join(input_stream.events)
    stream_input = client.inputs[0]

    assert stream_input["language_code"] == "en-US"
    assert stream_input["media_sample_rate_hz"] == expected_sample_rate
    assert stream_input["media_encoding"] == "pcm"
    assert sent_audio == expected_audio
    assert result.text == "hello world"
    assert result.metadata.time_to_first_partial_seconds is not None
    assert result.metadata.request_duration_seconds == 1.0
    assert len(input_stream.events) == 25
    assert all(len(chunk) == 3_200 for chunk in input_stream.events)
    assert input_stream.ended


@pytest.mark.parametrize("sample_rate", [7_999, 48_001])
async def test_aws_transcribe_rejects_unsupported_sample_rate(
    sample_rate: int,
) -> None:
    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(
            supports_transcription=True, registry_key="aws_transcribe/streaming"
        ),
    )

    with pytest.raises(ValueError, match="requires 8–48 kHz"):
        await model.transcribe_audio(
            name="audio.wav",
            mime="audio/wav",
            audio=tone_wav(sample_rate=sample_rate),
            language="en",
        )


async def test_aws_client_uses_custom_endpoint() -> None:
    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(
            custom_endpoint="https://transcribe.example.com",
            registry_key="aws_transcribe/streaming",
        ),
    )
    resolver = StaticCredentialResolver("access-key", "secret-key")

    with (
        patch.object(
            model,
            "_region_and_credentials",
            return_value=("us-east-1", resolver),
        ),
        patch.object(amazon, "TranscribeStreamingClient") as client_constructor,
    ):
        await model._client()

    client_constructor.assert_called_once()
    kwargs = client_constructor.call_args.kwargs
    endpoint_resolver = kwargs["endpoint_resolver"]
    assert isinstance(endpoint_resolver, StaticEndpointResolver)
    assert (
        await endpoint_resolver.resolve("us-west-2") == "https://transcribe.example.com"
    )
    assert kwargs["region"] == "us-east-1"
    assert kwargs["credential_resolver"] is resolver


def test_aws_credentials_require_region() -> None:
    session = MagicMock(region_name=None)
    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(registry_key="aws_transcribe/streaming"),
    )

    with (
        patch.object(amazon, "Session", MagicMock(return_value=session)),
        pytest.raises(
            RuntimeError,
            match="^A standard region configuration is required for transcription$",
        ),
    ):
        model._region_and_credentials("using-environment")  # pyright: ignore[reportPrivateUsage]


def test_aws_credentials_require_credentials() -> None:
    session = MagicMock(region_name="us-east-1")
    session.get_credentials.return_value = None
    model = AWSTranscribeModel(
        "streaming",
        config=LLMConfig(registry_key="aws_transcribe/streaming"),
    )

    with (
        patch.object(amazon, "Session", MagicMock(return_value=session)),
        pytest.raises(
            RuntimeError,
            match=(
                "^Cloud credentials are required for transcription\\. "
                "Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
                "or the standard AWS credential chain\\.$"
            ),
        ),
    ):
        model._region_and_credentials("using-environment")  # pyright: ignore[reportPrivateUsage]
