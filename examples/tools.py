# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import asyncio

from model_library.base import LLM, SystemInput, TextInput
from model_library.base.output import QueryResult
from model_library.registry_utils import get_registry_model

from examples.quickstart import basic_agent
from examples.setup import console_log, setup, sync_model_metadata
from examples.utils import GetWeather


async def single_tool_call(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    max_attempts: int = 3,
) -> QueryResult | None:
    """Minimal bounded tool-call probe for validators."""
    if not quiet:
        console_log("\n--- Single Tool Call ---\n")

    try:
        output: QueryResult | None = None
        for _ in range(max_attempts):
            result = await model.query(
                [
                    SystemInput(
                        text="You must call the get_weather tool exactly once. Do not answer directly."
                    ),
                    TextInput(text="What is the weather in Tokyo right now?"),
                ],
                tools=[GetWeather().definition],
            )
            output = result
            if result.tool_calls:
                if not quiet:
                    print(f"\nTool Calls: {result.tool_calls}\n")
                return result
        return output
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run tool-use demos with a model")
    parser.add_argument(
        "model",
        type=str,
        help="Model endpoint",
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "direct", "both"],
        default="agent",
        help="Demo to run (default: agent)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.instance_logger.info(model)

    await sync_model_metadata(model)
    if not model.supports_tools:
        raise Exception("Model does not support tools")

    if args.mode in {"direct", "both"}:
        await single_tool_call(model)
    if args.mode in {"agent", "both"}:
        await basic_agent(model, max_turns=5, max_seconds=90)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
