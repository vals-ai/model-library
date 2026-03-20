"""Replay a turn from an agent run.

Runs an agent, then replays one of its turns by loading the exact history
that was sent to the LLM and re-querying with the same input.

Useful for debugging (reproduce a specific turn), regression testing
(check if the model still makes the same decision), or experimenting with
a different model on the same history.
"""

import asyncio
import json

from model_library.agent import Agent, AgentConfig, AgentTurn, TurnLimit
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import TextInput, ToolCall, ToolDefinition
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup
from ..utils import GetWeather


async def run_and_replay(run_model: LLM, replay_model: LLM) -> None:
    # --- Run the agent ---
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

    # --- Pick a turn to replay ---
    turns_dir = result.output_dir / "turns"
    turn_dirs = sorted(turns_dir.glob("turn_*"))
    if not turn_dirs:
        console_log("No turns to replay.")
        return

    # Skip the last turn (submit) — replay the one before it, which has tool results in history
    turn_dir = turn_dirs[-2] if len(turn_dirs) >= 2 else turn_dirs[-1]

    # --- Load history and config ---
    history = LLM.deserialize_input(turn_dir / "history.bin")
    config = json.loads((turns_dir / "init" / "config.json").read_text())
    original = AgentTurn.model_validate_json((turn_dir / "result.json").read_text())

    tool_defs = [ToolDefinition.model_validate(td) for td in config.get("tools", [])]

    # --- Replay with a different model ---
    console_log(f"\n--- Replay ({turn_dir.name}) ---")
    response = await replay_model.query(
        input=history, tools=tool_defs, question_id="replay"
    )

    # --- Compare ---
    orig_meta = original.query_result.metadata
    replay_meta = response.metadata

    def fmt_calls(calls: list[ToolCall]) -> list[str]:
        return [f"{c.name}({c.parsed_args})" for c in calls]

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


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="openai/gpt-5-nano-2025-08-07")
    parser.add_argument(
        "replay_model", nargs="?", default="openai/gpt-5-mini-2025-08-07"
    )
    args = parser.parse_args()

    run_model = get_registry_model(args.model)
    replay_model = get_registry_model(args.replay_model)
    await run_and_replay(run_model, replay_model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
