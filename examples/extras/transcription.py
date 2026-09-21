# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio

from model_library.registry_utils import get_registry_model

from examples.data.audio import speech_wav
from examples.setup import console_log, setup


async def transcribe(model_str: str) -> None:
    """Transcribe a short clip with a registry-configured model."""
    console_log("\n--- Transcription ---\n")

    model = get_registry_model(model_str)
    console_log(
        f"Supports transcription: {model.supports_transcription}, "
        f"temperature: {model.supports_temperature}"
    )

    audio = speech_wav()
    result = await model.transcribe_audio(
        name="clip.wav",
        mime="audio/wav",
        audio=audio,
        language="en",
    )

    console_log(f"Text: {result.text}")
    console_log(f"Metadata: {result.metadata.model_dump(exclude_none=True)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a short audio clip")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-4o-transcribe",
        help="Transcription model (default: openai/gpt-4o-transcribe)",
    )
    args = parser.parse_args()

    await transcribe(args.model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
