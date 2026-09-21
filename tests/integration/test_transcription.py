"""Integration tests for speech-to-text providers."""

import pytest

from examples.data.audio import long_speech_wav, speech_wav
from model_library.register_models import get_transcription_registry
from model_library.registry_utils import get_transcription_model

_STT_MODELS = tuple(
    (
        model_key,
        bool(config.transcription_streaming),
    )
    for model_key, config in get_transcription_registry().items()
    if not config.metadata.deprecated
)
# Realtime engines need the longer utterance to endpoint.
@pytest.mark.parametrize(
    ("model_key", "supports_streaming", "long_clip"),
    [
        pytest.param(
            model_key,
            supports_streaming,
            long_clip,
            id=f"{model_key}-{'long' if long_clip else 'short'}",
        )
        for model_key, supports_streaming in _STT_MODELS
        for long_clip in ([True] if supports_streaming else [False, True])
    ],
)
async def test_transcribes_clip(
    model_key: str, supports_streaming: bool, long_clip: bool
) -> None:
    """Speech clips return transcripts, measurements, and streamed partial timing."""
    audio = long_speech_wav() if long_clip else speech_wav()
    result = await get_transcription_model(model_key).transcribe_audio(
        name="speech.wav",
        mime="audio/wav",
        audio=audio,
        language="en",
    )

    assert result.text.strip()
    assert result.metadata.audio_bytes == len(audio)
    assert result.metadata.request_duration_seconds > 0
    if long_clip:
        assert result.metadata.audio_duration_seconds == pytest.approx(8.07, abs=0.1)
    if supports_streaming:
        time_to_first_partial = result.metadata.time_to_first_partial_seconds
        assert time_to_first_partial is not None
        assert 0 < time_to_first_partial < result.metadata.request_duration_seconds
