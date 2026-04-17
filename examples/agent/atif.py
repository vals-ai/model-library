"""Agent example: ATIF trajectory export.

Demonstrates running an agent with ATIF v1.6 trajectory export enabled.
The trajectory is written to trajectory_atif.json in the agent's output directory.
"""

import asyncio
import json

from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup
from ..utils import GetWeather, SaveNote


async def agent_with_atif(model: LLM):
    """Agent with ATIF trajectory export enabled.

    Writes trajectory_atif.json to the agent's output directory after the run.
    """
    console_log("\n--- Agent with ATIF Export ---\n")

    from model_library.agent import (
        Agent,
        AgentConfig,
        TurnLimit,
    )

    agent = Agent(
        name="atif-example",
        llm=model,
        tools=[GetWeather(), SaveNote(), SubmitTool()],
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=10),
            time_limit=None,
        ),
    )
    result = await agent.run(
        [TextInput(text="Get the weather in Tokyo and save it as 'tokyo_weather'.")],
        question_id="question_1",
        atif_export=True,
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"Logs: {result.output_dir}")

    atif_path = result.output_dir / "trajectory_atif.json"
    trajectory = json.loads(atif_path.read_text())
    console_log(f"\nATIF trajectory: {atif_path}")
    console_log(f"  schema_version: {trajectory['schema_version']}")
    console_log(f"  steps: {trajectory['final_metrics']['total_steps']}")
    console_log(
        f"  prompt_tokens: {trajectory['final_metrics']['total_prompt_tokens']}"
    )
    console_log(
        f"  completion_tokens: {trajectory['final_metrics']['total_completion_tokens']}"
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agent with ATIF export")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    await agent_with_atif(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
