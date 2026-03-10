"""Agent example: tool-augmented conversation loop.

Demonstrates creating tools, configuring an agent, and running it.
The agent decides which tools to call and when to stop.
"""

import asyncio
import json
import logging
from typing import Any

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentTurn,
    ErrorTurn,
    SerializableException,
    Tool,
    ToolOutput,
    TurnLimit,
    TurnResult,
)
from model_library.agent.hooks import default_determine_answer
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from .setup import console_log, setup

# --- Tools ---


class GetWeather(Tool):
    """Returns fake weather data for a city."""

    name = "get_weather"
    description = "Get current weather for a city. Returns temperature and conditions."
    parameters = {
        "city": {
            "type": "string",
            "description": "City name, e.g. 'San Francisco'",
        },
    }

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        city = args.get("city", "unknown")
        weather = {"city": city, "temperature": "18C", "conditions": "foggy"}
        return ToolOutput(output=json.dumps(weather))


class SaveNote(Tool):
    """Saves a note to shared state. Demonstrates state passing between tools."""

    name = "save_note"
    description = "Save a note to the session. Other tools can read it later."
    parameters = {
        "key": {"type": "string", "description": "Note identifier"},
        "content": {"type": "string", "description": "Note content"},
    }

    def __init__(self, custom_logic: str | None = None):
        self.custom_logic = custom_logic

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        key, content = args["key"], args["content"]
        state[key] = content
        return ToolOutput(output=f"Saved note '{key}'.")


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


async def agent_with_web_search(model: LLM):
    """Agent with web search: uses TavilyWebSearch to answer questions about the real world."""
    console_log("\n--- Agent with Web Search ---\n")

    from model_library.agent.tools.web_search import TavilyWebSearch

    agent = Agent(
        name="web_search",
        llm=model,
        tools=[TavilyWebSearch(max_end_date="2025-04-07"), SubmitTool()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
    )
    result = await agent.run(
        [
            TextInput(
                text="Search for the latest news about AI regulation, then submit a brief summary."
            )
        ],
        question_id="question_1",
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"Turns: {result.total_turns}, Tool calls: {result.tool_calls_count}")


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


async def agent_with_hooks(model: LLM):
    """Agent with lifecycle hooks: should_stop and determine_answer."""
    console_log("\n--- Agent with Hooks ---\n")

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

    agent = Agent(
        name="hooks",
        llm=model,
        tools=[GetWeather(), SaveNote()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
        hooks=AgentHooks(
            should_stop=stop_after_turns,
            determine_answer=answer_from_state,
        ),
    )

    state: dict[str, Any] = {}
    result = await agent.run(
        [TextInput(text="Get the weather in Tokyo and save it as 'tokyo_weather'.")],
        question_id="question_1",
        state=state,
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"State keys: {list(state.keys())}")


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
    # await agent_with_bash(model, logger)
    await agent_with_submit_tool(model)
    # await agent_with_web_search(model)
    # await agent_with_hooks(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
