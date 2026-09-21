"""Deepgram transcription adapter tests."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from deepgram.listen.v1.types import ListenV1Metadata, ListenV1Results
import pytest

from examples.data.audio import speech_wav, long_speech_wav
from model_library.base import TranscriptionOnly
from model_library.base.transcription import parse_mono_pcm16_wav
from model_library.providers.voice.deepgram import DeepgramModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection,
    ScriptedStream,
    config,
)

_TRANSCRIPTION_ONLY_GET_CLIENT = TranscriptionOnly.get_client
_DEEPGRAM_GET_CLIENT = DeepgramModel.get_client


def _deepgram_turn(
    transcript: str, event: str, turn_index: int = 0
) -> dict[str, object]:
    return {
        "type": "TurnInfo",
        "request_id": "request-1",
        "sequence_id": turn_index,
        "event": event,
        "turn_index": turn_index,
        "audio_window_start": 0.0,
        "audio_window_end": 0.5,
        "transcript": transcript,
        "words": [],
        "end_of_turn_confidence": 1.0,
    }


def _deepgram_v1_result(
    transcript: str, *, is_final: bool, duration: float
) -> ListenV1Results:
    return ListenV1Results.model_validate(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": duration,
            "start": 0.0,
            "is_final": is_final,
            "speech_final": is_final,
            "channel": {
                "alternatives": [
                    {"transcript": transcript, "confidence": 0.99, "words": []}
                ]
            },
            "metadata": {
                "request_id": "request-1",
                "model_info": {"name": "nova-3", "version": "1", "arch": "nova-3"},
                "model_uuid": "model-1",
            },
        }
    )


class _DeepgramV2Stream(ScriptedStream):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(cast(list[object], responses))
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[object]:
        await self.closed.wait()
        for response in self._script:
            yield response

    async def send_media(self, chunk: bytes) -> None:
        self._record_send(chunk)

    async def send_close_stream(self) -> None:
        self._record_send("close")
        self.closed.set()


class _DeepgramV1Stream(ScriptedStream):
    def __init__(
        self,
        interim_results: list[ListenV1Results],
        final_results: list[ListenV1Results],
        *,
        include_metadata: bool = True,
    ) -> None:
        super().__init__([*interim_results, *final_results])
        self._include_metadata = include_metadata

    async def send_media(self, chunk: bytes) -> None:
        self._record_send(chunk)

    async def send_finalize(self) -> None:
        self._record_send("finalize")

    async def send_close_stream(self) -> None:
        self._record_send("close")

    async def __aiter__(self) -> AsyncIterator[ListenV1Results | ListenV1Metadata]:
        for response in self._script:
            if isinstance(response, ListenV1Results):
                yield response
        if self._include_metadata:
            yield ListenV1Metadata.model_validate(
                {
                    "type": "Metadata",
                    "transaction_key": "transaction-1",
                    "request_id": "request-1",
                    "sha256": "hash",
                    "created": "2026-08-21T00:00:00Z",
                    "duration": 0.75,
                    "channels": 1,
                }
            )


def test_deepgram_passes_custom_endpoint_verbatim() -> None:
    model_config = config("deepgram/flux-general-en")
    model_config.custom_endpoint = "https://host/v1/listen/"
    different_config = config("deepgram/flux-general-en")
    different_config.custom_endpoint = "https://other-host/v1/listen/"
    httpx_client = MagicMock()
    client = MagicMock()

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(DeepgramModel, "get_client", _DEEPGRAM_GET_CLIENT),
        patch(
            "model_library.providers.voice.deepgram.AsyncDeepgramClient",
            return_value=client,
        ) as constructor,
        patch(
            "model_library.providers.voice.deepgram.default_httpx_client",
            return_value=httpx_client,
        ),
    ):
        first = DeepgramModel("flux-general-en", config=model_config)
        second = DeepgramModel("nova-3", config=model_config)
        different = DeepgramModel("flux-general-en", config=different_config)

        assert first.get_client() is second.get_client()
        assert first._client_registry_key != different._client_registry_key  # pyright: ignore[reportPrivateUsage]

    assert constructor.call_count == 2
    environment = constructor.call_args_list[0].kwargs["environment"]
    assert environment.base == "https://host/v1/listen/"


@pytest.mark.parametrize("audio", [speech_wav(), long_speech_wav()], ids=["short", "long"])
async def test_deepgram_flux_keeps_open_turn_when_stream_closes_without_end(
    audio: bytes,
) -> None:
    stream = _DeepgramV2Stream(
        [
            _deepgram_turn("first turn", "EndOfTurn", turn_index=0),
            _deepgram_turn("second", "Update", turn_index=1),
            _deepgram_turn("second turn", "Update", turn_index=1),
        ]
    )
    model = DeepgramModel("flux-general-en", config=config("deepgram/flux-general-en"))
    client = MagicMock(spec=["listen"])
    client.listen.v2.connect.return_value = Connection(stream)

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=audio, language="en"
        )

    assert result.text == "first turn second turn"
    assert result.metadata.billable_duration_seconds == 1.0
    assert stream.sends[-1] == "close"
    sent_audio = b"".join(chunk for chunk in stream.sends if isinstance(chunk, bytes))
    assert sent_audio == parse_mono_pcm16_wav(audio).frames


async def test_deepgram_nova_maps_metadata_duration_to_billing() -> None:
    stream = _DeepgramV1Stream(
        [], [_deepgram_v1_result("hello", is_final=True, duration=0.75)]
    )
    client = MagicMock(spec=["listen"])
    client.listen.v1.connect.return_value = Connection(stream)
    model = DeepgramModel("nova-3", config=config("deepgram/nova-3-streaming"))

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "hello"
    assert result.metadata.billable_duration_seconds == 0.75
    assert result.metadata.cost_usd == pytest.approx(0.0048 * 0.75 / 60)


async def test_deepgram_nova_requires_terminal_metadata() -> None:
    stream = _DeepgramV1Stream(
        [],
        [_deepgram_v1_result("hello world", is_final=True, duration=0.75)],
        include_metadata=False,
    )
    client = MagicMock(spec=["listen"])
    client.listen.v1.connect.return_value = Connection(stream)
    httpx_client = MagicMock()
    httpx_client.aclose = AsyncMock()

    with (
        patch.object(TranscriptionOnly, "get_client", _TRANSCRIPTION_ONLY_GET_CLIENT),
        patch.object(DeepgramModel, "get_client", _DEEPGRAM_GET_CLIENT),
        patch(
            "model_library.providers.voice.deepgram.AsyncDeepgramClient",
            return_value=client,
        ),
        patch(
            "model_library.providers.voice.deepgram.default_httpx_client",
            return_value=httpx_client,
        ),
        pytest.raises(RuntimeError, match="terminal metadata"),
    ):
        model = DeepgramModel("nova-3", config=config("deepgram/nova-3-streaming"))
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    httpx_client.aclose.assert_not_awaited()
