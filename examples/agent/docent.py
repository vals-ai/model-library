"""Agent example: Docent ingestion.

Demonstrates running an agent with Docent transcript ingestion enabled.
Hook-injected messages (turn_message, time_message) are captured in the
Docent transcript so you can see exactly what the model was prompted with.

Requires the docent extra: uv pip install model-library[docent]
"""

import asyncio
import uuid

from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup
from ..utils import GetWeather, SaveNote


async def agent_with_docent(model: LLM):
    """Agent with Docent ingestion enabled.

    Hook-injected messages (turn_message, time_message) are captured in the
    Docent transcript so you can see exactly what the model was prompted with.

    Known limitation: history truncation via before_query (e.g. truncate_oldest)
    is not reflected in the Docent transcript. Truncated turns still appear
    even though the model no longer saw them.
    """
    console_log("\n--- Agent with Docent Ingestion ---\n")

    from docent import Docent

    from model_library.agent import (
        Agent,
        AgentConfig,
        TimeLimit,
        TurnLimit,
    )

    run_id = f"example-{uuid.uuid4().hex[:8]}"
    Docent().create_collection(run_id)

    agent = Agent(
        name="docent-example",
        llm=model,
        tools=[GetWeather(), SaveNote(), SubmitTool()],
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=10),
            time_limit=TimeLimit(max_seconds=120),
        ),
    )
    result = await agent.run(
        [TextInput(text="Get the weather in Tokyo and save it as 'tokyo_weather'.")],
        question_id="question_1",
        run_id=run_id,
        docent_ingest=True,
    )

    console_log(f"Answer: {result.final_answer}")
    console_log(f"Logs: {result.output_dir}")
    console_log(f"Docent dashboard: https://docent.transluce.org/dashboard/{run_id}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agent with Docent ingestion")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    await agent_with_docent(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
