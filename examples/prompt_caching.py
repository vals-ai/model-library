#!/usr/bin/env python
import argparse
import asyncio
from typing import Any, Coroutine

from model_library.base import LLM, ToolBody, ToolDefinition
from model_library.registry_utils import get_registry_model

from .setup import setup


async def run(model: LLM) -> None:
    system_prefix = (
        "You are a helpful assistant. " + ("Legal agreement terms. " * 4000)
    ).strip()
    task_spec = "<TASK_INSTRUCTIONS>\n" + ("Full agreement text. " * 2500).strip()

    tools: list[ToolDefinition] = [
        ToolDefinition(
            name="get_clause_text",
            body=ToolBody(
                name="get_clause_text",
                description="Extract the full text of a named clause from the agreement context.",
                properties={"clause_name": {"type": "string"}},
                required=["clause_name"],
            ),
        ),
        ToolDefinition(
            name="analyze_clause",
            body=ToolBody(
                name="analyze_clause",
                description=(
                    "Analyze a clause for risks or obligations in a given jurisdiction; "
                    "useful for termination, indemnification, confidentiality, etc."
                ),
                properties={
                    "clause_name": {"type": "string"},
                    "jurisdiction": {"type": "string"},
                },
                required=["clause_name", "jurisdiction"],
            ),
        ),
    ]

    async def query_with_logging(tag: str, question: str) -> None:
        user_prompt = f"{task_spec}\n\nQUESTION: {question}"
        await model.query(
            input=user_prompt,
            system_prompt=system_prefix,
            tools=tools,
        )

    await query_with_logging(
        "first_query",
        "List the key obligations and summarize the indemnification clause.",
    )
    await query_with_logging(
        "second_query",
        "Evaluate termination clause enforceability under California law.",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt caching demo")
    parser.add_argument(
        "model",
        nargs="?",
        default="anthropic/claude-haiku-4-5-20251001",
        type=str,
        help="Model endpoint (default: anthropic/claude-haiku-4-5-20251001)",
    )
    args = parser.parse_args()

    model: LLM = get_registry_model(args.model)
    model.logger.info(model)

    tasks: list[Coroutine[Any, Any, None]] = []
    tasks.append(run(model))
    for task in tasks:
        await task


if __name__ == "__main__":
    setup()
    asyncio.run(main())
