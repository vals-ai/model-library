# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio

from model_library.base import LLM, ToolDefinition
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup


async def deep_research(model: LLM):
    console_log("\n--- Deep Research ---\n")
    if model.gateway_mode:
        raise Exception(
            "Deep research example uses provider-specific OpenAI query parameters "
            "and is direct-provider only; unset MODEL_GATEWAY_URL to run it."
        )

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
    model.instance_logger.info(model)

    if not model.supports_tools:
        raise Exception("Model does not support tools")

    await deep_research(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
