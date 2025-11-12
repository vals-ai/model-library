import asyncio

from model_library.base import (
    LLM,
    TextInput,
    ToolBody,
    ToolDefinition,
    ToolResult,
)
from model_library.registry_utils import get_registry_model

from .setup import console_log, setup


async def tool_calls(model: LLM):
    console_log("\n--- Tool Calls ---\n")

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

    output1 = None
    while (
        not output1
        or not output1.tool_calls
        or output1.tool_calls[0].name != "get_weather"
    ):
        output1 = await model.query(
            [TextInput(text="What is the weather in San Francisco right now?")],
            tools=tools,
            system_prompt="You must make exactly 0 or 1 tool calls per answer. You must not make more than 1 tool call per answer.",
        )
    print(f"\nTool Calls: {output1.tool_calls}\n")

    output2 = None
    while (
        not output2
        or not output2.tool_calls
        or output2.tool_calls[0].name != "get_danger"
    ):
        output2 = await model.query(
            [
                # order matters! ToolResult should come first
                ToolResult(tool_call=output1.tool_calls[0], result="25C"),
                TextInput(
                    text="Also, includes some weird emojies in your answer (at least 8 of them). If the weather is under 30C, can you also check what the danger is there?"
                ),
            ],
            history=output1.history,
            tools=tools,
        )
    print(f"\nTool Calls: {output2.tool_calls}\n")

    output3 = await model.query(
        [
            ToolResult(tool_call=output2.tool_calls[0], result="low"),
        ],
        history=output2.history,
        tools=tools,
    )
    print(f"\nTool Calls: {output3.tool_calls}\n")
    # print(f"History: {output3.history}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run tool call example with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    if not model.supports_tools:
        raise Exception("Model does not support tools")

    await tool_calls(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
