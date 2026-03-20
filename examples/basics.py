import asyncio
import base64
from typing import Any, Coroutine

from model_library.base import (
    LLM,
    FileWithBase64,
    LLMConfig,
    TextInput,
)
from model_library.registry_utils import get_registry_model

from .data.files import secret_pdf
from .data.images import red_image
from .setup import console_log, setup


async def basic(model: LLM):
    console_log("\n--- Basic ---\n")

    # or ...query("string") instead of [Input]
    await model.query(
        [
            TextInput(
                text="What is QSBS? Explain your thinking in detail and make it concise"
            )
        ],
    )


async def system_prompt(model: LLM):
    console_log("\n--- System Prompt ---\n")

    await model.query(
        [TextInput(text="Hello, how are you?")],
        system_prompt="You are a pirate, answer in the speaking style of a pirate. Keeps responses under 10 words",
        # any argument passed here, besides system_prompt, will be passed directly to the model
    )


async def image(model: LLM):
    console_log("\n--- Image ---\n")

    red_image_content = red_image()

    await model.query(
        [
            TextInput(text="What color is the image?"),
            FileWithBase64(
                type="image",
                name="red_image.png",
                mime="png",
                base64=base64.b64encode(red_image_content).decode("utf-8"),
            ),
        ]
    )


async def file(model: LLM):
    console_log("\n--- File ---\n")

    secret_file_content = secret_pdf()
    mime = "application/pdf"

    await model.query(
        [
            TextInput(text="What is the secret?"),
            FileWithBase64(
                type="file",
                name="file_base64.pdf",
                mime=mime,
                base64=base64.b64encode(secret_file_content).decode("utf-8"),
            ),
        ],
    )


async def custom_params(model: LLM):
    from google.genai import types

    # google show thinking
    await model.query(
        "Hello",
        thinking_config=types.ThinkingConfig(
            thinking_budget=24576, include_thoughts=True
        ),
    )
    # openai show thinking
    await model.query(
        "Hello",
        reasoning={"effort": "low", "summary": "auto"},
    )

    # openai verbosity
    await model.query("Hello", text={"verbosity": "low"})


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run basic examples with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="google/gemini-2.5-flash",
        type=str,
        help="Model endpoint (default: google/gemini-2.5-flash)",
    )
    args = parser.parse_args()

    # NOTE: or get_raw_model() to skip loading yaml config
    model = get_registry_model(args.model, LLMConfig(temperature=0.7, top_p=0.95))
    model.logger.info(model)

    tasks: list[Coroutine[Any, Any, None]] = []

    tasks.append(basic(model))
    tasks.append(system_prompt(model))
    if model.supports_images:
        tasks.append(image(model))
    if model.supports_files:
        tasks.append(file(model))

    for task in tasks:
        await task


if __name__ == "__main__":
    setup()
    asyncio.run(main())
