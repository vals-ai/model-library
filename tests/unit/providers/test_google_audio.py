"""Tests for Gemini audio file inputs.

Run: pytest tests/unit/providers/test_google_audio.py

Covers native Google provider request building for byte-backed audio input files.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)
from pydantic import SecretStr

from model_library.base import LLMConfig, TextInput
from model_library.base.input import FileWithBytes
from model_library.providers.google.google import GoogleModel


class TestGoogleAudioInput:
    """Audio file parsing for native Gemini requests."""

    async def test_query_sends_audio_file_as_inline_file_part(self) -> None:
        """
        Verify model.query sends Gemini audio input as inline file data.

        Test cases:
        - Audio MIME is preserved on the Gemini part.
        - Audio bytes are sent as inline data.
        - Query returns the provider text response.
        """

        async def stream() -> AsyncIterator[GenerateContentResponse]:
            yield GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(parts=[Part(text="tone")]),
                        finish_reason=FinishReason.STOP,
                    )
                ],
                usage_metadata=GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    candidates_token_count=1,
                ),
            )

        audio_bytes = b"fake mp3 bytes"
        client = MagicMock()
        client.aio.models.generate_content_stream = AsyncMock(return_value=stream())

        with patch.object(GoogleModel, "get_client", return_value=client):
            model = GoogleModel(
                "gemini-test",
                config=LLMConfig(custom_api_key=SecretStr("test-key")),
            )
            result = await model.query(
                [
                    TextInput(text="Transcribe this audio."),
                    FileWithBytes(
                        type="file",
                        name="clip.mp3",
                        mime="audio/mpeg",
                        data=audio_bytes,
                    ),
                ],
            )

        body = client.aio.models.generate_content_stream.call_args.kwargs
        content = body["contents"][0]
        assert content.parts[0].text == "Transcribe this audio."

        audio_part = content.parts[1]
        assert audio_part.inline_data is not None
        assert audio_part.inline_data.mime_type == "audio/mpeg"
        assert audio_part.inline_data.data == audio_bytes
        assert result.output_text == "tone"
