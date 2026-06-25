# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import logging

from model_library.agent import Agent, AgentConfig, AgentResult, TimeLimit, TurnLimit
from model_library.base import LLM, LLMConfig, SystemInput, TextInput
from model_library.base.output import QueryResult
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup, sync_model_metadata
from examples.utils import GetWeather, SaveNote


async def basic_query(
    model: LLM, *, quiet: bool = False, raise_errors: bool = False
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- Text Query ---\n")

    try:
        result = await model.query(
            [TextInput(text="What is QSBS? Explain it concisely.")],
        )
        if not quiet:
            console_log(result.output_text_str)
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def system_prompt(
    model: LLM, *, quiet: bool = False, raise_errors: bool = False
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- System Prompt ---\n")

    try:
        result = await model.query(
            [
                SystemInput(
                    text="You are a pirate. Answer in a pirate style under 10 words."
                ),
                TextInput(text="Hello, how are you?"),
            ],
        )
        if not quiet:
            console_log(result.output_text_str)
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def basic_agent(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    max_turns: int | None = None,
    max_seconds: float | None = None,
    logger: logging.Logger | None = None,
) -> AgentResult | None:
    if not quiet:
        console_log("\n--- Basic Agent ---\n")

    try:
        agent = Agent(
            name="quickstart",
            llm=model,
            tools=[GetWeather(), SaveNote()],
            config=AgentConfig(
                turn_limit=TurnLimit(max_turns=max_turns) if max_turns else None,
                time_limit=TimeLimit(max_seconds=max_seconds) if max_seconds else None,
                max_tool_calls_per_turn=1 if max_turns or max_seconds else None,
            ),
        )
        result = await agent.run(
            [
                TextInput(
                    text="What's the weather in Tokyo? Save it as a note called 'tokyo_weather'."
                )
            ],
            question_id="quickstart_agent",
            logger=logger,
        )

        if not quiet:
            console_log(f"Answer: {result.final_answer}")
            console_log(
                f"Turns: {result.total_turns}, Tool calls: {result.tool_calls_count}"
            )
            console_log(f"Logs: {result.output_dir}")
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main quickstart demo")
    parser.add_argument(
        "model",
        nargs="?",
        default="google/gemini-2.5-flash",
        type=str,
        help="Model endpoint (default: google/gemini-2.5-flash)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model, LLMConfig(temperature=0.7, top_p=0.95))
    model.instance_logger.info(model)
    await sync_model_metadata(model)

    await basic_query(model)
    await system_prompt(model)
    if model.supports_tools:
        await basic_agent(model, max_turns=5, max_seconds=90)
    else:
        console_log("Skipping agent demo: model does not support tools", color="yellow")


if __name__ == "__main__":
    setup()
    asyncio.run(main())
