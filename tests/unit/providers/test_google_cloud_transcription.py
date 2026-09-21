"""Google Cloud Speech transcription adapter tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import google.cloud.speech_v2 as speech_v2
import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig, TranscriptionOnly
from model_library.providers.google import speech
from model_library.providers.google.speech import GoogleCloudModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    clear_transcription_client_registry as clear_transcription_client_registry,
)

_TRANSCRIPTION_ONLY_GET_CLIENT = TranscriptionOnly.get_client
_GOOGLE_CLOUD_GET_CLIENT = GoogleCloudModel.get_client
_GOOGLE_API_KEY = json.dumps(
    {
        "GCP_PROJECT_ID": "project",
        "GCP_CREDS": json.dumps({"type": "service_account"}),
    }
)


class _ResponseStream:
    def __init__(self, responses: list[speech_v2.StreamingRecognizeResponse]) -> None:
        self._responses = responses
        self.requests: list[speech_v2.StreamingRecognizeRequest] = []
        self.request_iterator: object | None = None

    async def stream(self):
        assert self.request_iterator is not None
        async for request in self.request_iterator:  # type: ignore[union-attr]
            self.requests.append(request)
        for response in self._responses:
            yield response


class _Credentials:
    pass


def _config() -> LLMConfig:
    return LLMConfig(
        supports_transcription=True,
        custom_api_key=SecretStr(_GOOGLE_API_KEY),
        registry_key="google_cloud/chirp_3",
    )


def _response(
    transcript: str,
    *,
    is_final: bool,
    billed_duration_seconds: int | None = None,
) -> speech_v2.StreamingRecognizeResponse:
    metadata = (
        {"total_billed_duration": {"seconds": billed_duration_seconds}}
        if billed_duration_seconds is not None
        else None
    )
    return speech_v2.StreamingRecognizeResponse(
        results=[
            speech_v2.StreamingRecognitionResult(
                alternatives=[
                    speech_v2.SpeechRecognitionAlternative(transcript=transcript)
                ],
                is_final=is_final,
            )
        ],
        metadata=metadata,
    )


async def _model_result(stream: _ResponseStream, *, language: str = "en-US"):
    model = GoogleCloudModel("chirp_3", config=_config())
    client = MagicMock()

    async def streaming_recognize(*, requests: object, timeout: float):
        assert timeout == 300.0
        stream.request_iterator = requests
        return stream.stream()

    client.streaming_recognize = AsyncMock(side_effect=streaming_recognize)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with (
        patch.object(
            speech.service_account.Credentials,
            "from_service_account_info",
            return_value=_Credentials(),
        ) as credentials_factory,
        patch(
            "model_library.providers.google.speech.SpeechAsyncClient",
            return_value=client,
        ),
        patch.object(model, "get_client", return_value=client),
        patch("model_library.base.base.perf_counter", side_effect=[0.0, 2.0]),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language=language
        )
    credentials_factory.assert_called_once_with(
        {"type": "service_account"},
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return result, client


async def test_google_cloud_streaming_tracks_ttfp_and_final_transcript() -> None:
    stream = _ResponseStream(
        [
            _response("hello", is_final=False),
            _response("hello world", is_final=True, billed_duration_seconds=2),
        ]
    )

    result, client = await _model_result(stream)

    assert result.text == "hello world"
    assert result.metadata.request_duration_seconds == 2.0
    assert result.metadata.billable_duration_seconds == 2.0
    assert result.metadata.time_to_first_partial_seconds is not None
    assert client.streaming_recognize.await_args.kwargs["timeout"] == 300.0
    config_request, *audio_requests = stream.requests
    assert config_request.recognizer == "projects/project/locations/us/recognizers/_"
    assert config_request.streaming_config.config.language_codes == ["en-US"]
    assert config_request.streaming_config.config.model == "chirp_3"
    assert config_request.streaming_config.streaming_features.interim_results
    assert not config_request.audio
    assert b"".join(request.audio for request in audio_requests) == AUDIO
    client.__aexit__.assert_not_awaited()


@pytest.mark.parametrize(
    ("custom_endpoint", "expected_endpoint"),
    [
        ("https://proxy.example/google/v2", "https://proxy.example/google/v2"),
        (None, "us-speech.googleapis.com"),
    ],
)
def test_google_cloud_uses_custom_or_regional_endpoint(
    custom_endpoint: str | None, expected_endpoint: str
) -> None:
    model_config = _config()
    model_config.custom_endpoint = custom_endpoint
    client = MagicMock()

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(GoogleCloudModel, "get_client", _GOOGLE_CLOUD_GET_CLIENT),
        patch.object(
            speech.service_account.Credentials,
            "from_service_account_info",
            return_value=_Credentials(),
        ),
        patch.object(speech, "SpeechAsyncClient", return_value=client) as constructor,
    ):
        model = GoogleCloudModel("chirp_3", config=model_config)
        assert model.get_client() is client

    assert (
        constructor.call_args.kwargs["client_options"].api_endpoint == expected_endpoint
    )


async def test_google_cloud_reuses_client_for_multiple_transcriptions() -> None:
    streams = [
        _ResponseStream([_response("first", is_final=True)]),
        _ResponseStream([_response("second", is_final=True)]),
    ]
    client = MagicMock()

    async def streaming_recognize(*, requests: object, timeout: float):
        assert timeout == 300.0
        stream = streams.pop(0)
        stream.request_iterator = requests
        return stream.stream()

    client.streaming_recognize = AsyncMock(side_effect=streaming_recognize)
    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(GoogleCloudModel, "get_client", _GOOGLE_CLOUD_GET_CLIENT),
        patch.object(
            speech.service_account.Credentials,
            "from_service_account_info",
            return_value=_Credentials(),
        ) as credentials_factory,
        patch(
            "model_library.providers.google.speech.SpeechAsyncClient",
            return_value=client,
        ) as constructor,
    ):
        different_config = _config()
        different_config.custom_endpoint = "other-speech.googleapis.com"
        first_model = GoogleCloudModel("chirp_3", config=_config())
        second_model = GoogleCloudModel("chirp_3", config=_config())
        different_model = GoogleCloudModel("chirp_3", config=different_config)

        assert first_model.get_client() is second_model.get_client()
        assert first_model._client_registry_key != different_model._client_registry_key  # pyright: ignore[reportPrivateUsage]

        result = await first_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en-US"
        )
        second_result = await second_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en-US"
        )

    assert result.text == "first"
    assert second_result.text == "second"
    assert constructor.call_count == 2
    assert credentials_factory.call_count == 3


async def test_google_cloud_client_exits_when_streaming_fails() -> None:
    model = GoogleCloudModel("chirp_3", config=_config())
    client = MagicMock()
    client.streaming_recognize = AsyncMock(side_effect=RuntimeError("stream failed"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(
            speech.service_account.Credentials,
            "from_service_account_info",
            return_value=_Credentials(),
        ),
        patch.object(model, "get_client", return_value=client),
        pytest.raises(RuntimeError, match="stream failed"),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)

    client.__aexit__.assert_not_awaited()
