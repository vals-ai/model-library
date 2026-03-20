"""Resume an agent run from a specific turn.

Runs an agent, then resumes from a mid-run turn by loading the exact history
and state that existed at that point and starting a new agent from there.

Useful for comparing how two models complete the same task given identical
prior context, or for re-running the tail of a failed/interrupted run.
"""

import asyncio
import json

from model_library.agent import Agent, AgentConfig, TurnLimit
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup
from ..utils import GetWeather


async def run_and_resume(run_model: LLM, resume_model: LLM) -> None:
    tools = [GetWeather(), SubmitTool()]
    config = AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None)

    # --- Run the agent ---
    console_log("--- Original run ---")
    agent = Agent(name="resume_demo", llm=run_model, tools=tools, config=config)
    result = await agent.run(
        [
            TextInput(
                text="Check the weather in Tokyo and Paris. Which city is warmer? Submit a one-sentence answer."
            )
        ],
        question_id="q1",
    )

    # --- Pick the turn to resume from ---
    turns_dir = result.output_dir / "turns"
    turn_dirs = sorted(turns_dir.glob("turn_*"))
    if len(turn_dirs) < 2:
        console_log("Not enough turns to resume from.")
        return

    # Resume from turn 2 — it has at least one tool result in history
    turn_dir = turn_dirs[1]

    # --- Load checkpoint ---
    history = LLM.deserialize_input(turn_dir / "history.bin")
    state = json.loads((turn_dir / "state.json").read_text())

    # --- Resume with a different model ---
    console_log("\n--- Resume run ---")
    resume_agent = Agent(
        name="resume_demo", llm=resume_model, tools=tools, config=config
    )
    resume_result = await resume_agent.run(history, state=state, question_id="resume")

    # --- Compare ---
    console_log(f"\n{'':20} {'Original':35} {'Resume'}")
    console_log(f"{'Model':<20} {run_model.model_name:<35} {resume_model.model_name}")
    console_log(
        f"{'Total turns':<20} {result.total_turns:<35} {resume_result.total_turns} (from {turn_dir.name})"
    )
    console_log(f"\nOriginal answer: {result.final_answer}")
    console_log(f"Resume answer:   {resume_result.final_answer}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="openai/gpt-5-nano-2025-08-07")
    parser.add_argument(
        "resume_model", nargs="?", default="openai/gpt-5-mini-2025-08-07"
    )
    args = parser.parse_args()

    run_model = get_registry_model(args.model)
    resume_model = get_registry_model(args.resume_model)
    await run_and_resume(run_model, resume_model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
