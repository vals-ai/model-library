from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from collections.abc import Callable
from datetime import date

from model_library.agent.agent import Agent, AgentResult
from model_library.agent.config import AgentConfig
from model_library.agent.tool import Tool
from model_library.agent.tools import BashTool, StopTool, SubmitTool, TavilyWebSearch
from dotenv import load_dotenv

from model_library.base import TextInput
from model_library.registry_utils import get_registry_model
from model_library.utils import create_file_logger

TOOL_REGISTRY: dict[str, Callable[[], Tool]] = {
    "stop": StopTool,
    "submit": SubmitTool,
    "web_search": lambda: TavilyWebSearch(max_end_date=date.today().isoformat()),
    "bash": lambda: BashTool(working_dir=str(Path.cwd())),
}


def get_tool(name: str) -> Tool:
    """Instantiate a built-in tool by name."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool {name!r}. Available: {list(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]()


def list_tools() -> list[str]:
    """Return available built-in tool names."""
    return list(TOOL_REGISTRY)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model_library.agent",
        description="Run the model_library agent from the command line.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Registry model key, e.g. anthropic/claude-sonnet-4-6",
    )
    parser.add_argument(
        "--problem-statement",
        required=True,
        help="Path to problem statement file, or '-' for stdin",
    )
    parser.add_argument(
        "--tools",
        default="",
        help=f"Comma-separated built-in tool names. Available: {list_tools()}",
    )
    parser.add_argument(
        "--max-turns", type=int, default=100, help="Maximum agent turns (default: 100)"
    )
    parser.add_argument(
        "--max-time-seconds",
        type=float,
        default=28800,
        help="Wall-clock time limit in seconds (default: 28800 = 8h)",
    )
    parser.add_argument(
        "--log-file", default="agent.log", help="Log file path (default: agent.log)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file to load (default: .env)",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        default=False,
        help="Also log to the console (stderr)",
    )
    parser.add_argument(
        "--output", default=None, help="Write AgentResult JSON to this path"
    )
    return parser


def _collect_tools(args: argparse.Namespace) -> list[Tool]:
    tools: list[Tool] = []
    if args.tools:
        for name in args.tools.split(","):
            tools.append(get_tool(name))  # invalid names will fail
    return tools


def _read_problem_statement(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


async def _run(args: argparse.Namespace) -> AgentResult:
    llm = get_registry_model(args.model)
    tools = _collect_tools(args)
    problem_statement = _read_problem_statement(args.problem_statement)
    config = AgentConfig(
        max_turns=args.max_turns, max_time_seconds=args.max_time_seconds
    )

    with create_file_logger(
        "model_library.agent.cli",
        args.log_file,
        level=getattr(logging, args.log_level),
        console=args.console,
    ) as logger:
        agent = Agent(llm=llm, tools=tools, logger=logger, config=config)
        result = await agent.run([TextInput(text=problem_statement)])

    return result


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    load_dotenv(args.env_file)

    result = asyncio.run(_run(args))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2))

    print(result.final_answer)


if __name__ == "__main__":
    main()
