from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    group: str
    command: str
    description: str


GROUPS = [
    "Core examples",
    "Model release checks",
    "Agent and tool loops",
    "Starter demos",
    "One-off feature examples",
    "Setup and compatibility asides",
]

COMMANDS = [
    Command(
        "Core examples",
        "uv run python examples/validate_model.py <model>",
        "capability report for a model",
    ),
    Command(
        "Model release checks",
        "uv run python examples/diagnostics/rate_limit.py <model>",
        "rate-limit headers/probe",
    ),
    Command(
        "Model release checks",
        "uv run python examples/diagnostics/token_retry.py <model>",
        "retry/queue behavior benchmark",
    ),
    Command(
        "Agent and tool loops",
        "uv run python examples/tools.py <model> [--mode agent|direct|both]",
        "basic agent tool use (direct probe is opt-in)",
    ),
    Command(
        "Agent and tool loops",
        "uv run python examples/extras/agent_flows.py <submit | bash | hooks | compaction> <model>",
        "advanced agent loop demos",
    ),
    Command(
        "Agent and tool loops",
        "uv run python examples/extras/agent_flows.py conductor <auditor> <target> <evaluator>",
        "conductor evaluation flow",
    ),
    Command(
        "Starter demos",
        "uv run python examples/quickstart.py",
        "minimal text query and system prompt",
    ),
    Command(
        "Starter demos", "uv run python examples/inputs.py", "image and file inputs"
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/structured_output.py [--mode pydantic|json-schema|both]",
        "structured output schema styles",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/count_tokens.py",
        "token counting",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/prompt_caching.py",
        "prompt caching with cached-token report",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/batch.py",
        "batch API",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/deep_research_openai.py",
        "OpenAI deep research",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/artifacts.py <atif | replay | resume>",
        "agent artifacts and run replay/resume",
    ),
    Command(
        "One-off feature examples",
        "uv run python examples/extras/load.py <embeddings | stress>",
        "embedding and query load demos",
    ),
    Command(
        "Setup and compatibility asides",
        "uv run python examples/extras/provider_setup.py <endpoint | registry-endpoint | custom-retrier>",
        "custom endpoints and retrier demo",
    ),
    Command(
        "Setup and compatibility asides",
        "uv run python examples/extras/google_modes.py <delegate | native>",
        "Google OpenAI-compatible and native paths",
    ),
]


def _print_group(title: str, commands: list[Command], *, width: int) -> None:
    if not commands:
        return

    print(f"\n{title}")
    print("-" * len(title))
    for command in commands:
        print(f"  {command.command:<{width}}  -  {command.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List runnable model-library examples",
        add_help=True,
    )
    parser.parse_args()

    print("Model Library examples")
    print(
        "Run commands from the repository root. Commands may make live provider calls."
    )
    print("Core examples come first; release checks are the recurring model probes.")

    width = max(len(command.command) for command in COMMANDS)
    for group in GROUPS:
        commands = [command for command in COMMANDS if command.group == group]
        _print_group(group, commands, width=width)
