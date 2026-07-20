import io
import math
import struct
import wave


def tone_wav(
    *,
    frequency_hz: int = 440,
    duration_seconds: float = 0.75,
    sample_rate: int = 16_000,
) -> bytes:
    amplitude = 12_000
    frame_count = int(sample_rate * duration_seconds)
    frames = bytearray()

    for frame_index in range(frame_count):
        sample = int(
            amplitude * math.sin(2 * math.pi * frequency_hz * frame_index / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))

    return buffer.getvalue()
