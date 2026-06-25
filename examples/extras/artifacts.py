"""Agent artifact examples: ATIF export, replay, and resume."""

# ruff: noqa: E402
from __future__ import annotations

# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import json

from model_library.agent import Agent, AgentConfig, AgentTurn, TurnLimit
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput, ToolCall, ToolDefinition
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup
from examples.utils import GetWeather, SaveNote


async def agent_with_atif(model: LLM) -> None:
    """Run an agent with ATIF trajectory export enabled."""
    console_log("\n--- Agent with ATIF Export ---\n")

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


def fmt_calls(calls: list[ToolCall]) -> list[str]:
    return [f"{c.name}({c.parsed_args})" for c in calls]


async def run_and_replay(run_model: LLM, replay_model: LLM) -> None:
    """Run an agent, then replay one turn with another model."""
    console_log("--- Original run ---")
    agent = Agent(
        name="replay_demo",
        llm=run_model,
        tools=[GetWeather(), SubmitTool()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
    )
    result = await agent.run(
        [
            TextInput(
                text="Check the weather in Tokyo and Paris. Which city is warmer? Submit a one-sentence answer."
            )
        ],
        question_id="q1",
    )

    turns_dir = result.output_dir / "turns"
    turn_dirs = sorted(turns_dir.glob("turn_*"))
    if not turn_dirs:
        console_log("No turns to replay.")
        return

    # Skip the last turn (submit) and replay the prior tool-result turn when present.
    turn_dir = turn_dirs[-2] if len(turn_dirs) >= 2 else turn_dirs[-1]

    history = LLM.deserialize_input(turn_dir / "history.json")
    config = json.loads((turns_dir / "init" / "config.json").read_text())
    original = AgentTurn.model_validate_json((turn_dir / "result.json").read_text())

    tool_defs = [ToolDefinition.model_validate(td) for td in config.get("tools", [])]

    console_log(f"\n--- Replay ({turn_dir.name}) ---")
    response = await replay_model.query(
        input=history, tools=tool_defs, question_id="replay"
    )

    orig_meta = original.query_result.metadata
    replay_meta = response.metadata

    console_log(f"\n{'':15} {'Original':30} {'Replay'}")
    console_log(f"{'Model':<15} {run_model.model_name:<30} {replay_model.model_name}")
    console_log(
        f"{'Tokens in':<15} {orig_meta.total_input_tokens:<30} {replay_meta.total_input_tokens}"
    )
    console_log(
        f"{'Tokens out':<15} {orig_meta.total_output_tokens:<30} {replay_meta.total_output_tokens}"
    )
    console_log(
        f"{'Duration':<15} {f'{original.effective_duration_seconds:.2f}s':<30} {replay_meta.default_duration_seconds:.2f}s"
    )
    console_log(
        f"\nOriginal tool calls: {fmt_calls([r.tool_call for r in original.tool_call_records])}"
    )
    console_log(f"Replay tool calls:   {fmt_calls(response.tool_calls)}")
    console_log(f"\nOriginal text: {original.query_result.output_text}")
    console_log(f"Replay text:   {response.output_text}")


async def run_and_resume(run_model: LLM, resume_model: LLM) -> None:
    """Run an agent, then resume from a mid-run turn with another model."""
    tools = [GetWeather(), SubmitTool()]
    config = AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None)

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

    turns_dir = result.output_dir / "turns"
    turn_dirs = sorted(turns_dir.glob("turn_*"))
    if len(turn_dirs) < 2:
        console_log("Not enough turns to resume from.")
        return

    # Resume from turn 2 — it has at least one tool result in history.
    turn_dir = turn_dirs[1]

    history = LLM.deserialize_input(turn_dir / "history.json")
    state = json.loads((turn_dir / "state.json").read_text())

    console_log("\n--- Resume run ---")
    resume_agent = Agent(
        name="resume_demo", llm=resume_model, tools=tools, config=config
    )
    resume_result = await resume_agent.run(history, state=state, question_id="resume")

    console_log(f"\n{'':20} {'Original':35} {'Resume'}")
    console_log(f"{'Model':<20} {run_model.model_name:<35} {resume_model.model_name}")
    console_log(
        f"{'Total turns':<20} {result.total_turns:<35} {resume_result.total_turns} (from {turn_dir.name})"
    )
    console_log(f"\nOriginal answer: {result.final_answer}")
    console_log(f"Resume answer:   {resume_result.final_answer}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent artifact extras")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    atif_parser = subparsers.add_parser("atif")
    atif_parser.add_argument("model", nargs="?", default="openai/gpt-5-nano-2025-08-07")

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "model", nargs="?", default="openai/gpt-5-nano-2025-08-07"
    )
    replay_parser.add_argument(
        "replay_model", nargs="?", default="openai/gpt-5-mini-2025-08-07"
    )

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument(
        "model", nargs="?", default="openai/gpt-5-nano-2025-08-07"
    )
    resume_parser.add_argument(
        "resume_model", nargs="?", default="openai/gpt-5-mini-2025-08-07"
    )

    args = parser.parse_args()

    if args.mode == "atif":
        await agent_with_atif(get_registry_model(args.model))
    elif args.mode == "replay":
        await run_and_replay(
            get_registry_model(args.model), get_registry_model(args.replay_model)
        )
    elif args.mode == "resume":
        await run_and_resume(
            get_registry_model(args.model), get_registry_model(args.resume_model)
        )


if __name__ == "__main__":
    setup()
    asyncio.run(main())
