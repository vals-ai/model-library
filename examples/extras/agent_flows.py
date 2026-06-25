"""Advanced agent flow examples.

This module keeps the related agent demos in one runnable file:
submit/bashing hooks, history compaction, and conductor evaluation.
"""

# ruff: noqa: E402
from __future__ import annotations

# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentTurn,
    ConductorAgent,
    ConductorConfig,
    ConductorResult,
    ErrorTurn,
    HistoryCompaction,
    SerializableException,
    TimeLimit,
    Tool,
    ToolOutput,
    TurnLimit,
    TurnResult,
    llm_summary_compactor,
)
from model_library.agent.hooks import default_determine_answer
from model_library.agent.tools.bash import BashTool
from model_library.agent.tools.submit import SubmitTool
from model_library.base import LLM
from model_library.base.input import InputItem, SystemInput, TextInput
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup
from examples.utils import GetWeather, SaveNote

SUMMARY_MARKER = "COMPACTED_OK:"


async def agent_with_submit_tool(model: LLM) -> None:
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


async def agent_with_bash(model: LLM) -> None:
    """Agent with bash: can run shell commands to answer questions."""
    console_log("\n--- Agent with Bash ---\n")

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


async def agent_with_hooks(model: LLM) -> None:
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


class LargeNotesTool(Tool):
    name = "fetch_notes"
    description = "Return large mocked incident notes."
    parameters = {"section": {"type": "string"}}

    async def execute(
        self,
        args: dict[str, Any],
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> ToolOutput:
        section = args["section"]
        notes = "\n".join(
            f"{section}-fact-{i}: service A emitted retry warnings after deploy 42."
            for i in range(600)
        )
        return ToolOutput(output=notes)


class StatusTool(Tool):
    name = "inspect_status"
    description = "Return current mocked service status."
    parameters = {"service": {"type": "string"}}

    async def execute(
        self,
        args: dict[str, Any],
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> ToolOutput:
        service = args["service"]
        return ToolOutput(
            output=(
                f"{service} warning rate returned to baseline 15 minutes after "
                "deploy 42. No data loss or customer-visible outage was detected."
            )
        )


class MitigationTool(Tool):
    name = "lookup_mitigation"
    description = "Return the mocked mitigation used for an incident."
    parameters = {"deploy": {"type": "string"}}

    async def execute(
        self,
        args: dict[str, Any],
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> ToolOutput:
        deploy = args["deploy"]
        return ToolOutput(
            output=(
                f"For {deploy}, operators paused the canary and temporarily "
                "increased retry backoff. Rollback was not required."
            )
        )


def _print_result(label: str, result: Any) -> None:
    console_log(f"\n--- {label} ---")
    console_log(f"Final answer: {result.final_answer}")
    console_log(f"Total turns:  {result.total_turns}")
    console_log(f"Tool usage:   {result.tool_usage}")
    console_log(f"Compactions:  {len(result.compactions)}")
    for i, c in enumerate(result.compactions, start=1):
        status = "ok" if c.success else f"failed: {c.error.type if c.error else '?'}"
        console_log(
            f"  [{i}] turn={c.turn_number} tokens={c.input_token_estimate} {status}"
        )
    console_log(f"Logs: {result.output_dir}")


def _assert_workflow_result(label: str, result: Any) -> None:
    final_answer = (result.final_answer or "").strip()
    if not result.success:
        raise RuntimeError(f"{label} failed: {result.final_error}")
    if not final_answer:
        raise RuntimeError(f"{label} did not produce a final answer")
    for tool_name in ("fetch_notes", "inspect_status", "lookup_mitigation"):
        if result.tool_usage.get(tool_name, 0) < 1:
            raise RuntimeError(f"{label} did not call {tool_name}")
    if not any(c.success for c in result.compactions):
        raise RuntimeError(f"{label} did not record a successful compaction")
    if not any(SUMMARY_MARKER in (c.summary or "") for c in result.compactions):
        raise RuntimeError(f"{label} did not use the compaction prompt")


async def threshold_demo(llm: LLM) -> None:
    """Compaction triggered by threshold_tokens crossing on pending tool output."""
    compaction_prompt = (
        "This is a compaction-only step. Do not continue the incident workflow, "
        "do not answer the user, and do not emit tool-call JSON.\n"
        f"The first characters of your response must be {SUMMARY_MARKER}\n"
        "Then summarize the conversation above for the agent. Include the "
        "user goal, key facts from tool outputs, completed tool calls, and the "
        "remaining ordered tool-call plan."
    )
    config_compaction = HistoryCompaction(threshold_tokens=2_000)
    agent = Agent(
        name="compaction-threshold",
        llm=llm,
        tools=[LargeNotesTool(), StatusTool(), MitigationTool()],
        log_dir=Path("logs"),
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=6),
            time_limit=None,
            max_tool_calls_per_turn=1,
            history_compaction=config_compaction,
        ),
        hooks=AgentHooks(
            compaction=llm_summary_compactor(
                llm, config_compaction, prompt=compaction_prompt
            ),
        ),
    )

    result = await agent.run(
        [
            SystemInput(text="Be concise."),
            TextInput(
                text=(
                    "Follow this workflow exactly, with only one tool call per turn: "
                    "first call fetch_notes for section A; after that, call "
                    "inspect_status for service A; after that, call lookup_mitigation "
                    "for deploy 42. Do not answer until all three tools have returned. "
                    "Then give one sentence beginning with INCIDENT_SUMMARY:"
                )
            ),
        ],
        question_id="threshold-demo",
        atif_export=True,
    )
    _assert_workflow_result("Threshold-triggered compaction", result)
    _print_result("Threshold-triggered compaction", result)


class LookupEligibilityTool(Tool):
    """Dummy tool: looks up SNAP eligibility based on household info."""

    name = "lookup_eligibility"
    description = "Look up SNAP eligibility based on household size and income."
    parameters: dict[str, Any] = {
        "household_size": {
            "type": "integer",
            "description": "Number of people in household",
        },
        "monthly_income": {
            "type": "number",
            "description": "Gross monthly income in dollars",
        },
    }

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        size = args.get("household_size", 1)
        income = args.get("monthly_income", 0)

        # Simplified 2024 gross income limits.
        limits = {1: 1580, 2: 2137, 3: 2694, 4: 3250, 5: 3807}
        limit = limits.get(size, 3250 + (size - 4) * 557)

        eligible = income <= limit
        return ToolOutput(
            output=f"Household of {size} with ${income}/mo income: "
            f"{'ELIGIBLE' if eligible else 'NOT ELIGIBLE'} "
            f"(gross income limit: ${limit}/mo)"
        )


class LookupOfficeTool(Tool):
    """Dummy tool: finds the nearest SNAP office."""

    name = "lookup_office"
    description = "Find the nearest SNAP office by state."
    parameters: dict[str, Any] = {
        "state": {
            "type": "string",
            "description": "Two-letter state code (e.g. CA, NY)",
        },
    }

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        state_code = args.get("state", "unknown").upper()

        return ToolOutput(
            output=f"Nearest SNAP office in {state_code}: "
            f"Department of Social Services, 123 Main St. "
            f"Phone: 1-800-555-0100. Hours: Mon-Fri 8am-5pm. "
            f"You can also apply online at {state_code.lower()}.gov/snap"
        )


class EndConversationTool(Tool):
    """Signal that the auditor is done with the conversation."""

    name = "end_conversation"
    description = (
        "Call this tool when you are done with the conversation. "
        "Pass a brief summary of what you learned as the answer. "
        "The conversation will end after this tool is called."
    )
    parameters: dict[str, Any] = {
        "answer": {
            "type": "string",
            "description": "Summary of the conversation.",
        }
    }
    required: list[str] = ["answer"]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        return ToolOutput(output=args["answer"], done=True)


def build_transcript(result: ConductorResult) -> str:
    """Format a ConductorResult into a plain-text transcript for evaluation."""
    lines: list[str] = []

    for message in result.messages:
        lines.append(f"[{message.role.capitalize()}]: {message.result.final_answer}")
        lines.append("")

    return "\n".join(lines)


async def snap_benefits_evaluation(
    auditor_model: LLM, target_model: LLM, evaluator_model: LLM
) -> None:
    """Run a SNAP benefits conversation, then evaluate the transcript with a separate LLM."""
    console_log("\n--- SNAP Benefits Evaluation ---\n")

    auditor_system_prompt = SystemInput(
        text=(
            "You are role-playing as a real person in a conversation. Stay in character at all times. "
            "You are NOT an assistant or AI — you are the person described below.\n\n"
            "YOUR CHARACTER: You are a single parent with two children (ages 4 and 7) "
            "who recently lost your job. You live in California and your last monthly income "
            "was about $2,000. You are not very familiar with government programs. "
            "A friend mentioned something called food stamps but you don't know the details.\n\n"
            "YOUR GOAL: Find out if you qualify for food assistance and how to apply. "
            "Ask natural questions like a real person would — mention your state, "
            "household size, and income when relevant so the assistant can look things up for you. "
            "When you feel your questions have been answered, call the end_conversation tool "
            "with a brief summary of what you learned."
        )
    )

    target_system_prompt = SystemInput(
        text=(
            "You are a helpful assistant that answers questions about SNAP "
            "(Supplemental Nutrition Assistance Program) benefits. Provide accurate, "
            "clear information about eligibility, application processes, and benefit amounts. "
            "Be empathetic and use plain language. "
            "Use the lookup_eligibility and lookup_office tools to provide specific answers."
        )
    )

    auditor = Agent(
        name="auditor",
        llm=auditor_model,
        tools=[EndConversationTool()],
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=3),
            time_limit=None,
        ),
    )

    target = Agent(
        name="target",
        llm=target_model,
        tools=[LookupEligibilityTool(), LookupOfficeTool()],
        config=AgentConfig(turn_limit=TurnLimit(max_turns=3), time_limit=None),
    )

    conductor = ConductorAgent(
        auditor=auditor,
        target=target,
        auditor_system_prompt=auditor_system_prompt,
        target_system_prompt=target_system_prompt,
        name="snap-benefits",
        config=ConductorConfig(max_exchanges=5),
    )

    result = await conductor.run(question_id="snap_benefits")

    console_log(f"Stop reason: {result.stop_reason}")
    console_log(f"Messages: {len(result.messages)}")
    console_log(f"Logs: {result.output_dir}")

    for message in result.messages:
        text = message.result.final_answer
        console_log(f"[{message.role.upper()}]:\n\n{text}\n")

    transcript = build_transcript(result)
    evaluation = await evaluator_model.query(
        input=[
            SystemInput(
                text=(
                    "You are evaluating a benefits assistance agent. "
                    "Score the agent on a scale of 1-5 for each criterion: "
                    "helpfulness, accuracy, empathy, and clarity. "
                    "Provide a brief justification for each score and an overall assessment."
                )
            ),
            TextInput(
                text=f"Here is the conversation transcript to evaluate:\n\n{transcript}"
            ),
        ]
    )

    console_log(f"\n--- Evaluator Assessment ---\n{evaluation.output_text}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced agent flow extras")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    for name in ["submit", "bash", "hooks", "compaction"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("model")

    conductor = subparsers.add_parser("conductor")
    conductor.add_argument("auditor_model")
    conductor.add_argument("target_model")
    conductor.add_argument("evaluator_model")

    args = parser.parse_args()

    if args.mode == "submit":
        await agent_with_submit_tool(get_registry_model(args.model))
    elif args.mode == "bash":
        await agent_with_bash(get_registry_model(args.model))
    elif args.mode == "hooks":
        await agent_with_hooks(get_registry_model(args.model))
    elif args.mode == "compaction":
        await threshold_demo(get_registry_model(args.model))
    elif args.mode == "conductor":
        await snap_benefits_evaluation(
            get_registry_model(args.auditor_model),
            get_registry_model(args.target_model),
            get_registry_model(args.evaluator_model),
        )


if __name__ == "__main__":
    setup()
    asyncio.run(main())
