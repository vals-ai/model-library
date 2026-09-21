"""Unit tests for OpenAI transcription transports.

Run: uv run pytest tests/unit/providers/test_openai_transcription.py
Covers complete-file, streaming-response, and realtime transcription behavior.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai import omit  # pyright: ignore[reportMissingImports]
from openai.types.audio.transcription import (  # pyright: ignore[reportMissingImports]
    Transcription,
    UsageDuration,
    UsageTokens,
)
from openai.types.audio.transcription_diarized import (  # pyright: ignore[reportMissingImports]
    UsageDuration as DiarizedUsageDuration,
    UsageTokens as DiarizedUsageTokens,
)
from openai.types.audio.transcription_text_delta_event import (  # pyright: ignore[reportMissingImports]
    TranscriptionTextDeltaEvent,
)
from openai.types.audio.transcription_text_done_event import (  # pyright: ignore[reportMissingImports]
    TranscriptionTextDoneEvent,
    Usage as TranscriptionUsage,
)
from openai.types.audio.transcription_verbose import (  # pyright: ignore[reportMissingImports]
    Usage as VerboseUsage,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (  # pyright: ignore[reportMissingImports]
    ConversationItemInputAudioTranscriptionCompletedEvent,
    UsageTranscriptTextUsageDuration,
    UsageTranscriptTextUsageTokens,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (  # pyright: ignore[reportMissingImports]
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
import pytest  # pyright: ignore[reportMissingImports]

from model_library import raw_model
from model_library.base import LLMConfig
from model_library.exceptions import ModelNoOutputError
from model_library.providers.openai import OpenAIModel
from model_library.providers.openai.voice import (
    OpenAITranscriptionUsage,
    _normalize_transcription_usage,  # pyright: ignore[reportPrivateUsage]
)
from model_library.registry_utils import get_registry_config
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection as _RealtimeConnectionManager,
)


class _RealtimeConnection:
    def __init__(
        self,
        responses: list[dict[str, object]],
        *,
        precommit_responses: list[dict[str, object]] | None = None,
    ) -> None:
        self._responses = responses
        self._precommit_responses = precommit_responses or []
        self._session_updated = asyncio.Event()
        self._committed = asyncio.Event()
        self.operations: list[str] = []
        self.session = MagicMock()
        self.session.update = AsyncMock(side_effect=self._update_session)
        self.input_audio_buffer = MagicMock()
        self.input_audio_buffer.append = AsyncMock(side_effect=self._append_audio)
        self.input_audio_buffer.commit = AsyncMock(side_effect=self._commit_audio)

    async def _update_session(self, **_kwargs: object) -> None:
        self._session_updated.set()

    async def _append_audio(self, **_kwargs: object) -> None:
        self.operations.append("append")

    async def _commit_audio(self) -> None:
        self.operations.append("commit")
        self._committed.set()

    @staticmethod
    def _event(payload: dict[str, object]) -> object:
        event_type = payload.get("type")
        if event_type == "conversation.item.input_audio_transcription.delta":
            return ConversationItemInputAudioTranscriptionDeltaEvent(
                event_id="event", content_index=0, **payload
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            usage_values = payload["usage"]
            assert isinstance(usage_values, dict)
            usage = (
                UsageTranscriptTextUsageDuration(**usage_values)
                if usage_values.get("type") == "duration"
                else UsageTranscriptTextUsageTokens(type="tokens", **usage_values)
            )
            return ConversationItemInputAudioTranscriptionCompletedEvent(
                event_id="event",
                content_index=0,
                **{**payload, "usage": usage},
            )
        error = payload.get("error")
        if isinstance(error, dict):
            payload = {**payload, "error": SimpleNamespace(**error)}
        return SimpleNamespace(**payload)

    async def __aiter__(self):
        yield self._event({"type": "session.created"})
        await self._session_updated.wait()
        yield self._event({"type": "session.updated"})
        for response in self._precommit_responses:
            yield self._event(response)
        await self._committed.wait()
        yield self._event(
            {
                "type": "input_audio_buffer.committed",
                "item_id": "committed-item",
            }
        )
        for response in self._responses:
            yield self._event(response)


class _FileTranscriptionStream:
    def __init__(
        self,
        events: list[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent],
    ) -> None:
        self._events = events
        self.closed = False

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


def _model() -> OpenAIModel:
    model = OpenAIModel(
        "gpt-live-transcribe",
        config=LLMConfig(
            supports_transcription=True,
            registry_key="openai/gpt-live-transcribe",
        ),
    )
    model._metadata = get_registry_config(  # pyright: ignore[reportPrivateUsage]
        "openai/gpt-live-transcribe"
    )
    return model


def _file_model(
    model_name: str = "gpt-4o-transcribe",
    *,
    registry_key: str | None = None,
    supports_streaming_transcription: bool = False,
) -> OpenAIModel:
    registry_key = registry_key or f"openai/{model_name}"
    model = OpenAIModel(
        model_name,
        config=LLMConfig(
            supports_transcription=True,
            supports_streaming_transcription=supports_streaming_transcription,
            registry_key=registry_key,
        ),
    )
    model._metadata = get_registry_config(  # pyright: ignore[reportPrivateUsage]
        registry_key
    )
    return model


@pytest.mark.parametrize(
    "usage",
    [
        UsageDuration(type="duration", seconds=2.5),
        DiarizedUsageDuration(type="duration", seconds=2.5),
        VerboseUsage(type="duration", seconds=2.5),
        UsageTranscriptTextUsageDuration(type="duration", seconds=2.5),
    ],
)
def test_normalize_openai_duration_usage_by_discriminator(
    usage: OpenAITranscriptionUsage,
) -> None:
    normalized = _normalize_transcription_usage(usage)

    assert normalized is not None
    assert normalized.billable_duration_seconds == 2.5


@pytest.mark.parametrize(
    "usage",
    [
        UsageTokens(type="tokens", input_tokens=12, output_tokens=3, total_tokens=999),
        DiarizedUsageTokens(
            type="tokens", input_tokens=12, output_tokens=3, total_tokens=999
        ),
        TranscriptionUsage(
            type="tokens", input_tokens=12, output_tokens=3, total_tokens=999
        ),
        UsageTranscriptTextUsageTokens(
            type="tokens", input_tokens=12, output_tokens=3, total_tokens=999
        ),
    ],
)
def test_normalize_openai_token_usage_by_discriminator(
    usage: OpenAITranscriptionUsage,
) -> None:
    normalized = _normalize_transcription_usage(usage)

    assert normalized is not None
    assert normalized.input_tokens == 12
    assert normalized.output_tokens == 3


async def test_openai_file_transcription_tracks_usage_and_registry_cost() -> None:
    response = Transcription(
        text="hello world",
        usage=UsageTokens(
            type="tokens",
            input_tokens=12,
            output_tokens=3,
            total_tokens=999,
            input_token_details={"audio_tokens": 10, "text_tokens": 2},
        ),
    )
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=response)
    model = _file_model()
    assert not model.supports_batch

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", AUDIO, "audio/wav"),
        model="gpt-4o-transcribe",
        language="en",
        response_format="json",
    )
    assert result.text == "hello world"
    assert result.metadata.audio_bytes == len(AUDIO)
    assert result.metadata.audio_duration_seconds == 0.75
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 3
    assert result.metadata.total_tokens == 15
    assert result.metadata.audio_tokens == 10
    assert result.metadata.text_tokens == 2
    assert result.metadata.cost_usd == 0.00006
    assert result.metadata.time_to_first_partial_seconds is None


@pytest.mark.parametrize("language", [None, "en"])
@pytest.mark.parametrize("use_raw_model", [False, True], ids=["direct", "raw"])
async def test_openai_file_transcription_language(
    use_raw_model: bool,
    language: str | None,
) -> None:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=Transcription(text="hello world")
    )
    config = LLMConfig(supports_transcription=True)
    model = (
        raw_model("openai/gpt-4o-transcribe", config=config)
        if use_raw_model
        else OpenAIModel("gpt-4o-transcribe", config=config)
    )
    assert isinstance(model, OpenAIModel)

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language=language
        )

    assert result.text == "hello world"
    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", AUDIO, "audio/wav"),
        model="gpt-4o-transcribe",
        language=language if language is not None else omit,
        response_format="json",
    )


async def test_openai_file_transcription_tracks_duration_usage() -> None:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=Transcription(
            text="hello world",
            usage=UsageDuration(type="duration", seconds=12.5),
        )
    )
    model = _file_model()

    with patch.object(model, "get_client", return_value=client):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.metadata.billable_duration_seconds == 12.5


@pytest.mark.parametrize("text", ["", "   "])
async def test_openai_file_transcription_rejects_empty_text(text: str) -> None:
    """Verify the complete-file transport matches other empty-output contracts."""
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=Transcription(text=text)
    )
    model = _file_model()

    with (
        patch.object(model, "get_client", return_value=client),
        pytest.raises(ModelNoOutputError, match="did not include transcript text"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )


async def test_openai_mini_streaming_response_tracks_usage() -> None:
    stream = _FileTranscriptionStream(
        [
            TranscriptionTextDeltaEvent(
                type="transcript.text.delta", delta="partial text"
            ),
            TranscriptionTextDoneEvent(
                type="transcript.text.done",
                text="authoritative final text",
                usage=TranscriptionUsage(
                    type="tokens",
                    input_tokens=12,
                    output_tokens=3,
                    total_tokens=15,
                    input_token_details={"audio_tokens": 10, "text_tokens": 2},
                ),
            ),
        ]
    )
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=stream)
    model = _file_model(
        "gpt-4o-mini-transcribe",
        registry_key="openai/gpt-4o-mini-transcribe-streaming-response",
        supports_streaming_transcription=True,
    )

    with (
        patch.object(model, "get_client", return_value=client),
        patch(
            "model_library.providers.openai.voice.time.perf_counter",
            side_effect=[0.0, 0.25, 1.0],
        ),
        patch(
            "model_library.base.base.perf_counter",
            side_effect=[0.0, 1.0],
        ),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("clip.wav", AUDIO, "audio/wav"),
        model="gpt-4o-mini-transcribe",
        language="en",
        response_format="json",
        stream=True,
    )
    assert result.text == "authoritative final text"
    assert result.metadata.audio_bytes == len(AUDIO)
    assert result.metadata.audio_duration_seconds == 0.75
    assert result.metadata.input_tokens == 12
    assert result.metadata.output_tokens == 3
    assert result.metadata.total_tokens == 15
    assert result.metadata.audio_tokens == 10
    assert result.metadata.text_tokens == 2
    assert result.metadata.cost_usd == pytest.approx(0.00003)
    assert result.metadata.request_duration_seconds == 1.0
    assert stream.closed


async def test_openai_mini_rejects_stream_without_done_event() -> None:
    stream = _FileTranscriptionStream(
        [TranscriptionTextDeltaEvent(type="transcript.text.delta", delta="partial")]
    )
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=stream)
    model = _file_model(
        "gpt-4o-mini-transcribe",
        registry_key="openai/gpt-4o-mini-transcribe-streaming-response",
        supports_streaming_transcription=True,
    )

    with (
        patch.object(model, "get_client", return_value=client),
        pytest.raises(ModelNoOutputError, match="did not include transcript text"),
    ):
        await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO, language="en"
        )

    assert stream.closed


async def test_openai_realtime_transcription_uses_manually_committed_item() -> None:
    connection = _RealtimeConnection(
        [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "vad-item",
                "transcript": "premature result",
                "usage": {"type": "duration", "seconds": 0.1},
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "committed-item",
                "transcript": "complete ordered transcript",
                "usage": {"type": "duration", "seconds": 1.25},
            },
        ],
        precommit_responses=[
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "vad-item",
                "delta": "premature",
            },
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "committed-item",
                "delta": "complete",
            },
        ],
    )
    client = MagicMock()
    client.realtime.connect.return_value = _RealtimeConnectionManager(connection)
    model = _model()

    with (
        patch.object(model, "get_client", return_value=client),
        patch("model_library.providers.openai.voice.asyncio.sleep", new=AsyncMock()),
    ):
        result = await model.transcribe_audio(
            name="clip.wav", mime="audio/wav", audio=AUDIO
        )

    assert result.text == "complete ordered transcript"
    assert result.metadata.billable_duration_seconds == 1.25
    assert result.metadata.time_to_first_partial_seconds is not None


async def test_openai_realtime_transcription_requires_pcm16_wav() -> None:
    model = _model()

    with pytest.raises(ValueError, match="WAV audio payload"):
        await model.transcribe_audio(
            name="clip.mp3", mime="audio/mpeg", audio=b"not-a-wav"
        )


@pytest.mark.parametrize(
    "response",
    [
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"message": "bad audio"},
        },
        {"type": "error", "error": {"message": "session failed"}},
    ],
)
async def test_openai_realtime_transcription_surfaces_errors(
    response: dict[str, object],
) -> None:
    connection = _RealtimeConnection([response])
    client = MagicMock()
    client.realtime.connect.return_value = _RealtimeConnectionManager(connection)
    model = _model()

    with (
        patch.object(model, "get_client", return_value=client),
        patch("model_library.providers.openai.voice.asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError, match="failed: "),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)
