"""Agent example: tool-augmented conversation loop.

Demonstrates creating tools, configuring an agent, and running it.
The agent decides which tools to call and when to stop.
"""

import asyncio
from typing import Any

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentTurn,
    ErrorTurn,
    SerializableException,
    TimeLimit,
    TurnLimit,
    TurnResult,
)
from model_library.agent.hooks import default_determine_answer
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import InputItem, TextInput
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup
from ..utils import GetWeather, SaveNote

# --- Run ---


async def basic_agent(model: LLM):
    """Minimal agent: LLM + tools, run until the model stops calling tools."""
    console_log("\n--- Basic Agent ---\n")

    agent = Agent(
        name="basic",
        llm=model,
        tools=[GetWeather(), SaveNote()],
        config=AgentConfig(turn_limit=None, time_limit=None),
    )
    result = await agent.run(
        [
            TextInput(
                text="What's the weather in Tokyo? Save it as a note called 'tokyo_weather'."
            )
        ],
        question_id="question_1",
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"Turns: {result.total_turns}, Tool calls: {result.tool_calls_count}")
    console_log(f"Logs: {result.output_dir}")


async def agent_with_submit_tool(model: LLM):
    """Agent with a submit tool that signals completion via done=True."""
    console_log("\n--- Agent with Submit Tool ---\n")

    agent = Agent(
        name="submit",
        llm=model,
        tools=[GetWeather(), SubmitTool()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=10), time_limit=None),
    )
    result = await agent.run(
        [
            TextInput(
                text="Check the weather in San Francisco, then submit a one-sentence summary."
            )
        ],
        question_id="question_1",
    )

    console_log(f"Final answer: {result.final_answer}")
    console_log(f"Logs: {result.output_dir}")


async def agent_with_bash(model: LLM):
    """Agent with bash: can run shell commands to answer questions."""
    console_log("\n--- Agent with Bash ---\n")

    from model_library.agent.tools.bash import BashTool

    agent = Agent(
        name="bash",
        llm=model,
        tools=[BashTool(working_dir="/tmp"), SubmitTool()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
    )
    result = await agent.run(
        [
            TextInput(
                text="Use bash to list the files in /tmp, then submit a summary of what you found."
            )
        ],
        question_id="question_1",
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"Turns: {result.total_turns}, Tool calls: {result.tool_calls_count}")
    console_log(f"Logs: {result.output_dir}")


def _turn_message(turn_number: int, max_turns: int) -> InputItem | None:
    remaining = max_turns - turn_number

    if remaining <= 3:
        return TextInput(
            text=f"[Turn {turn_number}/{max_turns} — {remaining} remaining. Wrap up soon.]"
        )

    return TextInput(text=f"[Turn {turn_number}/{max_turns}]")


def _time_message(elapsed_seconds: float, max_seconds: float) -> InputItem | None:
    remaining = max_seconds - elapsed_seconds

    if remaining < 30:
        return TextInput(text=f"[{remaining:.0f}s remaining — submit now.]")

    return None


def _build_hooks_agent(model: LLM) -> tuple[Agent, dict[str, Any]]:
    """Build an agent with all hook types for reuse across examples."""

    def stop_after_turns(turn_result: TurnResult) -> bool:
        if turn_result.turn_number >= 4:
            console_log("Turn limit reached via hook")
            return True
        return False

    def answer_from_state(
        state: dict[str, Any],
        turns: list[AgentTurn | ErrorTurn],
        final_error: SerializableException | None,
    ) -> str:
        if "tokyo_weather" in state:
            return f"From state: {state['tokyo_weather']}"
        return default_determine_answer(state, turns, final_error)

    state: dict[str, Any] = {}
    agent = Agent(
        name="hooks",
        llm=model,
        tools=[GetWeather(), SaveNote(), SubmitTool()],
        config=AgentConfig(
            turn_limit=TurnLimit(
                max_turns=10,
                turn_message=_turn_message,
            ),
            time_limit=TimeLimit(
                max_seconds=120,
                time_message=_time_message,
            ),
        ),
        hooks=AgentHooks(
            should_stop=stop_after_turns,
            determine_answer=answer_from_state,
        ),
    )

    return agent, state


async def agent_with_hooks(model: LLM):
    """Agent with lifecycle hooks: should_stop, determine_answer, turn_message, time_message."""
    console_log("\n--- Agent with Hooks ---\n")

    agent, state = _build_hooks_agent(model)
    result = await agent.run(
        [TextInput(text="Get the weather in Tokyo and save it as 'tokyo_weather'.")],
        question_id="question_1",
        state=state,
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"State keys: {list(state.keys())}")
    console_log(f"Logs: {result.output_dir}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agent examples")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)

    # await basic_agent(model)
    # await agent_with_bash(model)
    await agent_with_submit_tool(model)
    # await agent_with_hooks(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
