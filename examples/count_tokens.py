import asyncio
import logging

from model_library import set_logging
from model_library.base import (
    LLM,
    TextInput,
    ToolBody,
    ToolDefinition,
)
from model_library.registry_utils import get_registry_model

from .setup import console_log, setup


async def count_tokens(model: LLM):
    console_log("\n--- Count Tokens ---\n")

    tools = [
        ToolDefinition(
            name="get_weather",
            body=ToolBody(
                name="get_weather",
                description="Get current temperature in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    },
                },
                required=["location"],
            ),
        ),
        ToolDefinition(
            name="get_danger",
            body=ToolBody(
                name="get_danger",
                description="Get current danger in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    },
                },
                required=["location"],
            ),
        ),
    ]

    tokens = await model.count_tokens(
        [TextInput(text="What is the weather in San Francisco right now?")],
        tools=tools,
        system_prompt="You must make exactly 0 or 1 tool calls per answer. You must not make more than 1 tool call per answer.",
    )

    console_log(f"Tokens: {tokens}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Example of counting tokens")
    parser.add_argument(
        "model",
        nargs="?",
        default="google/gemini-2.5-flash",
        type=str,
        help="Model endpoint (default: google/gemini-2.5-flash)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    set_logging(enable=True, level=logging.DEBUG)

    await count_tokens(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
