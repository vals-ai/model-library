#!/usr/bin/env python
# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import json
from typing import cast

from model_library.base import LLM, SystemInput, TextInput
from model_library.base.output import QueryResult
from model_library.registry_utils import get_registry_model

from examples.setup import setup, sync_model_metadata


def _run_report(label: str, result: QueryResult) -> dict[str, object]:
    metadata = result.metadata
    cached_tokens = metadata.cache_read_tokens or 0
    cache_write_tokens = metadata.cache_write_tokens or 0
    return {
        "label": label,
        "answer": result.output_text_str,
        "cached_tokens": cached_tokens,
        "tokens": {
            "input": metadata.in_tokens,
            "output": metadata.out_tokens,
            "reasoning": metadata.reasoning_tokens,
            "cached": cached_tokens,
            "cache_read": cached_tokens,
            "cache_write": cache_write_tokens,
            "total_input": metadata.total_input_tokens,
            "total_output": metadata.total_output_tokens,
        },
        "cache": {
            "cached_tokens": cached_tokens,
            "read_tokens": cached_tokens,
            "write_tokens": cache_write_tokens,
            "used_cache": cached_tokens > 0,
            "wrote_cache": cache_write_tokens > 0,
        },
        "cost": metadata.cost.model_dump(mode="json") if metadata.cost else None,
        "duration_seconds": metadata.duration_seconds,
        "performance": metadata.performance.model_dump(mode="json"),
        "metadata_extra": metadata.extra,
    }


async def _query(model: LLM, cached_prefix: str, task_spec: str) -> QueryResult:
    return await model.query(
        input=[
            SystemInput(text=cached_prefix),
            TextInput(
                text=(
                    f"{task_spec}\n\n"
                    "QUESTION: List the key obligations and summarize the "
                    "indemnification clause in three concise bullets."
                )
            ),
        ],
    )


async def run(model: LLM) -> dict[str, object]:
    await sync_model_metadata(model)

    cached_prefix = (
        "You are a careful legal-analysis assistant. "
        + ("Reusable agreement context for prompt-cache demonstration. " * 5000)
    ).strip()
    task_spec = (
        "<AGREEMENT_CONTEXT>\n"
        + (
            "The vendor must protect confidential data, maintain audit logs, "
            "notify the customer of security incidents, indemnify the customer "
            "for third-party IP claims, and limit liability except for fraud, "
            "confidentiality breaches, and indemnification claims. " * 1500
        ).strip()
    )

    first = await _query(model, cached_prefix, task_spec)
    second = await _query(model, cached_prefix, task_spec)

    first_cached_tokens = first.metadata.cache_read_tokens or 0
    second_cached_tokens = second.metadata.cache_read_tokens or 0
    cached_tokens = second_cached_tokens
    registry_key = cast(object, getattr(model, "registry_key", None))
    status = "pass" if cached_tokens > 0 else "warn"
    caching_result = {
        "name": "prompt cache read",
        "status": status,
        "cached_tokens": cached_tokens,
        "detail": (
            "second run reported cached tokens"
            if cached_tokens > 0
            else "provider returned 0 cached tokens for the second identical run"
        ),
    }

    return {
        "model": {
            "provider": model.provider,
            "model_name": model.model_name,
            "registry_key": registry_key,
        },
        "prompt": {
            "cached_prefix_chars": len(cached_prefix),
            "task_chars": len(task_spec),
            "runs": 2,
            "why_two_runs": "first identical call can warm/write cache; second identical call can report cached tokens",
        },
        "sections": {"Caching": caching_result},
        "category_result": {"category": "Caching", **caching_result},
        "cache_summary": {
            "cached_tokens": cached_tokens,
            "first_cached_tokens": first_cached_tokens,
            "second_cached_tokens": second_cached_tokens,
            "second_read_more_than_first": second_cached_tokens > first_cached_tokens,
        },
        "runs": [
            _run_report("first: cache warm-up", first),
            _run_report("second: cache read", second),
        ],
    }


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

    model = get_registry_model(args.model)
    report = await run(model)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    setup(disable_logging=True)
    asyncio.run(main())
