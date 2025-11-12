import argparse
import asyncio

from model_library.base import LLM, ToolDefinition
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup


async def deep_research(model: LLM):
    console_log("\n--- Deep Research ---\n")

    tools = [
        ToolDefinition(name="web_search_preview", body={"type": "web_search_preview"}),
        ToolDefinition(
            name="code_interpreter",
            body={
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": []},
            },
        ),
    ]

    kwargs = {}
    if model.provider == "openai":
        kwargs["background"] = True
        kwargs["reasoning"] = {"summary": "auto"}

    response = await model.query(
        "Summarize the latest challenges in deploying fusion energy at utility scale",
        **kwargs,  # type: ignore
        tools=tools,
    )
    for citation in response.extras.citations:
        print(citation, "\n")


async def main():
    parser = argparse.ArgumentParser(description="Run a Deep Research example query")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/o4-mini-deep-research-2025-06-26",
        type=str,
        help="Model endpoint (default: openai/o4-mini-deep-research-2025-06-26)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    if not model.supports_tools:
        raise Exception("Model does not support tools")

    await deep_research(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
