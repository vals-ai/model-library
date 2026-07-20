"""
Integration tests for audio input handling (real API when available).
"""

from model_library.registry_utils import get_registry_model
from tests.test_helpers import get_example_audio_bytes_input


async def test_google_audio_bytes():
    """
    Verify Gemini accepts inline audio through model.query.

    Test cases:
    - A generated WAV clip is passed as FileWithBytes.
    - Gemini returns text identifying the clip as a tone.
    """
    model = get_registry_model("google/gemini-2.5-flash")
    assert model.supports_audio

    result = await model.query(get_example_audio_bytes_input())

    assert "tone" in result.output_text_str.lower()
