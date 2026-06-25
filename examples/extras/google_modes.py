# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio

from pydantic import SecretStr

from model_library.base import LLMConfig, TextInput
from model_library.registry_utils import get_raw_model

from examples.setup import console_log, setup


async def google_delegate_to_openai_completions() -> None:
    """Route Google through OpenAI chat completions via native=False."""
    console_log("\n--- Google Delegate -> OpenAI Chat Completions ---\n")

    # custom_endpoint / custom_api_key are optional — defaults point at
    # Google's OpenAI-compat endpoint using GOOGLE_API_KEY from env.
    config = LLMConfig(
        native=False,
        max_tokens=64,
        temperature=0.5,
        top_p=0.9,
        custom_endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_raw_model("google/gemini-2.5-flash", config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


async def google_native_with_top_k() -> None:
    """Run Google through its native genai path with top_k."""
    console_log("\n--- Google Native (top_k) ---\n")

    config = LLMConfig(
        max_tokens=64,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
    )
    model = get_raw_model("google/gemini-2.5-flash", config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Google-specific extras")
    parser.add_argument(
        "mode",
        choices=["delegate", "native"],
        help="Google demo to run",
    )
    args = parser.parse_args()

    if args.mode == "delegate":
        await google_delegate_to_openai_completions()
    elif args.mode == "native":
        await google_native_with_top_k()


if __name__ == "__main__":
    setup()
    asyncio.run(main())
