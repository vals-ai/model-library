"""Meta transcription adapter tests."""

import json
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from model_library.providers.delegates.meta import MetaModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection,
    ScriptedStream,
    config,
)


class _MetaStream(ScriptedStream):
    async def send(self, message: str | bytes) -> None:
        self._record_send(message)

    async def __aiter__(self) -> AsyncIterator[str]:
        yield json.dumps({"sessionId": "session-1"})
        for event in self._script:
            yield json.dumps(event)


def _model() -> MetaModel:
    return MetaModel(
        "muse-voice-transcribe-1.0", config=config("meta/muse_voice_transcribe")
    )


def test_meta_masks_api_key_in_repr() -> None:
    assert "provider-key" not in repr(_model())


async def test_meta_rejects_custom_endpoint_before_opening_socket() -> None:
    model_config = config("meta/muse_voice_transcribe")
    model_config.custom_endpoint = "https://custom.example/v1"
    model = MetaModel("muse-voice-transcribe-1.0", config=model_config)

    with (
        patch("model_library.providers.voice.meta.connect") as connect,
        pytest.raises(ValueError, match="custom_endpoint is not supported"),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)

    connect.assert_not_called()


async def test_meta_keeps_only_the_final_cumulative_transcript() -> None:
    stream = _MetaStream(
        [
            {"type": "transcript", "transcript": "the weather", "final": False},
            {"type": "transcript", "transcript": "the weather is", "final": False},
            {
                "type": "transcript",
                "transcript": "the weather is mild.",
                "final": True,
                "audioProcessedMs": 2_400,
            },
        ]
    )

    with patch(
        "model_library.providers.voice.meta.connect",
        return_value=Connection(stream),
    ):
        result = await _model().transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "the weather is mild."
    assert result.metadata.audio_duration_seconds == pytest.approx(0.75)
    assert result.metadata.billable_duration_seconds == 2
    assert result.metadata.cost_usd == pytest.approx(0.003 * 2 / 60)

    handshake = stream.sends[0]
    assert isinstance(handshake, str)
    assert json.loads(handshake)["authorization"] == {
        "accessToken": "Bearer provider-key"
    }

    end_stream = stream.sends[-1]
    assert isinstance(end_stream, str)
    assert json.loads(end_stream) == {"type": "endStream"}


async def test_meta_floors_local_duration_when_provider_omits_it() -> None:
    stream = _MetaStream(
        [
            {
                "type": "transcript",
                "transcript": "the weather is mild.",
                "final": True,
            }
        ]
    )

    with patch(
        "model_library.providers.voice.meta.connect",
        return_value=Connection(stream),
    ):
        result = await _model().transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.metadata.audio_duration_seconds == pytest.approx(0.75)
    assert result.metadata.billable_duration_seconds == 0
    assert result.metadata.cost_usd == 0


async def test_meta_surfaces_error_event() -> None:
    stream = _MetaStream([{"type": "error", "message": "unsupported encoding"}])

    with (
        patch(
            "model_library.providers.voice.meta.connect",
            return_value=Connection(stream),
        ),
        pytest.raises(RuntimeError, match="Transcription failed: unsupported encoding"),
    ):
        await _model().transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)
