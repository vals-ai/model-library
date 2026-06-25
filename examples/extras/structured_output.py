# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import asyncio
from typing import Any

from pydantic import BaseModel

from model_library import model
from model_library.base import LLM, TextInput

from examples.setup import console_log, setup

RECIPE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "prep_time_minutes": {"type": "integer"},
        "servings": {"type": "integer"},
    },
    "required": ["name", "ingredients", "steps", "prep_time_minutes", "servings"],
    "additionalProperties": False,
}


class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int
    servings: int


async def pydantic_structured_output(llm: LLM):
    """Structured output using a Pydantic model as output_schema."""
    console_log(f"\n--- Pydantic Structured Output ({llm.model_name}) ---\n")

    result = await llm.query(
        [TextInput(text="Give me a recipe for chocolate chip cookies")],
        output_schema=Recipe,
    )

    assert isinstance(result.output_parsed, Recipe)
    console_log(f"Recipe: {result.output_parsed.name}")
    console_log(f"Ingredients: {result.output_parsed.ingredients}")
    console_log(f"Prep time: {result.output_parsed.prep_time_minutes} minutes")
    console_log(f"Servings: {result.output_parsed.servings}")


async def json_schema_structured_output(llm: LLM):
    """Structured output using a raw JSON schema dict as output_schema."""
    console_log(f"\n--- JSON Schema Structured Output ({llm.model_name}) ---\n")

    result = await llm.query(
        [TextInput(text="Give me a recipe for banana bread")],
        output_schema=RECIPE_JSON_SCHEMA,
    )

    assert isinstance(result.output_parsed, dict)
    console_log(f"Recipe: {result.output_parsed['name']}")
    console_log(f"Ingredients: {result.output_parsed['ingredients']}")
    console_log(f"Prep time: {result.output_parsed['prep_time_minutes']} minutes")
    console_log(f"Servings: {result.output_parsed['servings']}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run structured output example with a model"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="anthropic/claude-sonnet-4-6",
        type=str,
        help="Model endpoint (default: anthropic/claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--mode",
        choices=["pydantic", "json-schema", "both"],
        default="pydantic",
        help="Structured output style to run (default: pydantic)",
    )
    args = parser.parse_args()
    llm: LLM = model(args.model)
    if args.mode in {"pydantic", "both"}:
        await pydantic_structured_output(llm)
    if args.mode in {"json-schema", "both"}:
        await json_schema_structured_output(llm)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
