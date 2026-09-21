"""Mistral transcription adapter tests."""

from unittest.mock import AsyncMock, MagicMock, patch

from mistralai.client.models import (
    RealtimeTranscriptionSessionUpdated,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)
import pytest
from pydantic import SecretStr

from model_library.base import TranscriptionOnly
from model_library.exceptions import ModelNoOutputError
from model_library.providers.mistral import MistralModel, MistralTranscriptionModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    ScriptedStream,
    clear_transcription_client_registry as clear_transcription_client_registry,
    config,
)

_TRANSCRIPTION_ONLY_GET_CLIENT = TranscriptionOnly.get_client
_MISTRAL_TRANSCRIPTION_GET_CLIENT = MistralTranscriptionModel.get_client


class _RealtimeConnection(ScriptedStream):
    def __init__(self, responses: list[object]) -> None:
        super().__init__(
            [RealtimeTranscriptionSessionUpdated.model_construct(), *responses]
        )
        self.calls: list[str] = []
        self.audio: list[bytes] = []
        self.closed = False

    async def send_audio(self, chunk: bytes) -> None:
        self.calls.append("send_audio")
        self._record_send(chunk)
        self.audio.append(bytes(chunk))

    async def flush_audio(self) -> None:
        self.calls.append("flush_audio")

    async def end_audio(self) -> None:
        self.calls.append("end_audio")

    async def close(self) -> None:
        self.closed = True


def _client(connection: _RealtimeConnection) -> MagicMock:
    client = MagicMock()
    client.audio.realtime.connect = AsyncMock(return_value=connection)
    return client


async def test_mistral_offline_uses_file_transcription() -> None:
    response = MagicMock(text="hello world")
    client = MagicMock()
    client.audio.transcriptions.complete_async = AsyncMock(return_value=response)
    async_client = MagicMock()
    async_client.aclose = AsyncMock()
    model_config = config("mistral/voxtral-mini-2602")
    model_config.custom_endpoint = (
        "https://proxy.example/mistral/v1/audio/transcriptions"
    )
    different_config = config("mistral/voxtral-mini-2602")
    different_config.custom_endpoint = model_config.custom_endpoint
    different_config.custom_api_key = SecretStr("other-key")

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(
            MistralTranscriptionModel,
            "get_client",
            _MISTRAL_TRANSCRIPTION_GET_CLIENT,
        ),
        patch(
            "model_library.providers.mistral.transcribe.Mistral",
            return_value=client,
        ) as constructor,
        patch(
            "model_library.providers.mistral.transcribe.default_httpx_client",
            return_value=async_client,
        ),
    ):
        first_model = MistralTranscriptionModel(
            "voxtral-mini-2602", config=model_config
        )
        second_model = MistralTranscriptionModel(
            "voxtral-mini-2602", config=model_config
        )
        different_model = MistralTranscriptionModel(
            "voxtral-mini-2602", config=different_config
        )
        chat_model = MistralModel("mistral-medium", config=model_config)

        assert first_model.get_client() is second_model.get_client()
        assert first_model._client_registry_key != different_model._client_registry_key  # pyright: ignore[reportPrivateUsage]
        assert first_model._client_registry_key[0] == "mistral.transcription"  # pyright: ignore[reportPrivateUsage]
        assert chat_model._client_registry_key[0] == "mistral"  # pyright: ignore[reportPrivateUsage]
        assert first_model._client_registry_key != chat_model._client_registry_key  # pyright: ignore[reportPrivateUsage]

        result = await first_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )
        second_result = await second_model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    assert result.text == "hello world"
    assert second_result.text == "hello world"
    assert result.metadata.time_to_first_partial_seconds is None
    assert result.metadata.cost_usd == pytest.approx(0.003 * 0.75 / 60)
    assert constructor.call_count == 2
    assert constructor.call_args_list[0].kwargs == {
        "api_key": "provider-key",
        "async_client": async_client,
        "server_url": "https://proxy.example/mistral/v1/audio/transcriptions",
        "timeout_ms": 300_000,
    }
    assert client.audio.transcriptions.complete_async.await_count == 2
    client.audio.transcriptions.complete_async.assert_awaited_with(
        model="voxtral-mini-2602",
        file={"file_name": "clip.wav", "content": AUDIO},
        language="en",
    )
    async_client.aclose.assert_not_awaited()


async def test_mistral_realtime_requires_done_text() -> None:
    connection = _RealtimeConnection(
        [
            TranscriptionStreamTextDelta.model_construct(text="provisional"),
            TranscriptionStreamDone.model_construct(text=""),
        ]
    )
    model = MistralTranscriptionModel(
        "voxtral-mini-transcribe-realtime-2602",
        config=config("mistral/voxtral-mini-transcribe-realtime-2602"),
    )

    with (
        patch.object(model, "get_client", return_value=_client(connection)),
        pytest.raises(ModelNoOutputError, match="did not include a final transcript"),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)
