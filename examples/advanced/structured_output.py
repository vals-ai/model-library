import asyncio

from pydantic import BaseModel

from model_library import model
from model_library.base import LLM, TextInput

from examples.setup import console_log, setup


class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int
    servings: int


async def basic_structured_output(llm: LLM):
    """Basic example of structured output using query_json"""
    console_log(f"\n--- Basic Structured Output ({llm.model_name}) ---\n")

    await llm.query_json(
        [TextInput(text="Give me a recipe for chocolate chip cookies")],
        pydantic_model=Recipe,
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run structured output example with a model"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()
    llm: LLM = model(args.model)
    await basic_structured_output(llm)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
