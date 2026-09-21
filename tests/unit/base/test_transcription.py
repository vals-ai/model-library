"""Unit tests for the shared transcription module.

Run: uv run pytest tests/unit/base/test_transcription.py
"""

import asyncio
import io
import struct
import time
import wave

import pytest  # pyright: ignore[reportMissingImports]

from model_library.exceptions import ModelNoOutputError
from model_library.base.transcription import (
    MonoPcm16Audio,
    TranscriptCollector,
    parse_mono_pcm16_wav,
    resolve_language,
    stream_audio_chunks,
)


def _wav(
    frames: bytes = b"\x01\x00\x02\x00",
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return buffer.getvalue()


def _with_format_code(audio: bytes, format_code: int) -> bytes:
    modified = bytearray(audio)
    struct.pack_into("<H", modified, 20, format_code)
    return bytes(modified)


@pytest.mark.parametrize(
    ("language", "registry_key", "expected"),
    [
        (None, "cohere/cohere-transcribe-03-2026", "en"),
        ("es-MX", "cohere/cohere-transcribe-03-2026", "es"),
        ("en", "google_cloud/chirp_3", "en-US"),
        ("es-MX", "google_cloud/chirp_3", "es-MX"),
        (None, "google_cloud/chirp_3", "en-US"),
        (None, None, None),
        (None, "openai/gpt-4o", None),
        ("en", None, "en"),
    ],
)
def test_resolve_language(
    language: str | None,
    registry_key: str | None,
    expected: str | None,
) -> None:
    assert resolve_language(language, registry_key) == expected


class TestParseMonoPcm16Wav:
    def test_extracts_named_audio_fields(self) -> None:
        assert parse_mono_pcm16_wav(_wav()) == MonoPcm16Audio(
            frames=b"\x01\x00\x02\x00",
            sample_rate_hz=16_000,
        )

    @pytest.mark.parametrize(
        ("audio", "message"),
        [
            (b"not-a-wav", "valid WAV"),
            (_wav()[:-1], "complete WAV"),
            (_wav(b""), "must not be empty"),
            (_wav(channels=2), "mono PCM16"),
            (_wav(sample_width=1), "mono PCM16"),
            (_with_format_code(_wav(), 3), "valid WAV"),
        ],
    )
    def test_rejects_invalid_protocol_audio(self, audio: bytes, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_mono_pcm16_wav(audio)

    def test_rejects_zero_sample_rate(self) -> None:
        audio = bytearray(_wav())
        struct.pack_into("<I", audio, 24, 0)
        struct.pack_into("<I", audio, 28, 0)

        with pytest.raises(ValueError, match="positive sample rate"):
            parse_mono_pcm16_wav(bytes(audio))

    def test_rejects_wrong_required_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="24 kHz mono PCM16"):
            parse_mono_pcm16_wav(_wav(), required_sample_rate=24_000)


class TestTranscriptCollector:
    def test_final_first_event_stamps_time_to_first_transcript(self) -> None:
        collector = TranscriptCollector()
        collector.audio_started()

        collector.observe("hello", is_final=True)
        stamped = collector.time_to_first_partial_seconds
        assert stamped is not None
        collector.observe("hello world", is_final=False)

        assert collector.has_final
        assert collector.transcript() == "hello"
        assert collector.time_to_first_partial_seconds == stamped

    def test_joins_final_segments_and_stamps_first_partial(self) -> None:
        collector = TranscriptCollector()
        collector.audio_started()

        collector.observe("", is_final=False)
        assert collector.time_to_first_partial_seconds is None

        collector.observe("hel", is_final=False)
        stamped = collector.time_to_first_partial_seconds
        assert stamped is not None
        collector.observe("hello", is_final=False)
        collector.observe("hello", is_final=True)
        collector.observe("world", is_final=True)

        assert collector.has_final
        assert collector.transcript() == "hello world"
        assert collector.time_to_first_partial_seconds == stamped

    def test_padded_segments_join_without_duplicate_spaces(self) -> None:
        collector = TranscriptCollector()
        collector.audio_started()
        collector.observe(" hello", is_final=True)
        collector.observe(" world ", is_final=True)
        collector.observe(" .", is_final=True)

        assert collector.transcript() == "hello world."

    def test_transcript_requires_a_final_segment(self) -> None:
        collector = TranscriptCollector()
        collector.audio_started()
        collector.observe("partial only", is_final=False)

        assert not collector.has_final
        with pytest.raises(
            ModelNoOutputError, match="did not include a final transcript"
        ):
            collector.transcript()


async def test_throttled_send_does_not_accumulate_send_latency() -> None:
    """Slow sends must not push the stream below the requested rate."""

    async def send(_chunk: bytes) -> None:
        await asyncio.sleep(0.1)

    started = time.perf_counter()
    await stream_audio_chunks(
        b"\x00" * 400,
        chunk_bytes=100,
        send=send,
        collector=TranscriptCollector(),
        max_bytes_per_second=1_000,
    )
    # 400 bytes at 1000 B/s is 0.4s of audio, which the sends alone cover.
    assert time.perf_counter() - started < 0.55


async def test_send_timeout_bounds_one_blocked_transport_write() -> None:
    async def send(_chunk: bytes) -> None:
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await stream_audio_chunks(
            b"\x00" * 100,
            chunk_bytes=100,
            send=send,
            collector=TranscriptCollector(),
            send_timeout_seconds=0.01,
        )
