# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import asyncio

from model_library.base import LLM, SystemInput, TextInput
from model_library.providers.azure import AzureOpenAIModel
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup


async def embeddings(model: OpenAIModel | AzureOpenAIModel, size: int):
    from tqdm import tqdm

    console_log("\n--- Embeddings ---\n")

    async def run_single():
        return await model.get_embedding(
            "What is QSBS? Explain your thinking in detail and make it concise"
        )

    tasks = [run_single() for _ in range(size)]

    lengths: list[int] = []
    for f in tqdm(asyncio.as_completed(tasks), total=size):
        result = await f
        lengths.append(len(result))

    assert all(result == lengths[0] for result in lengths)

    print(f"Produced {size} embeddings of length [{lengths[0]}]")


async def stress(model: LLM, size: int):
    from tqdm import tqdm

    console_log("\n--- Stress ---\n")

    async def run_single():
        prompt = "Tell me a bedtime story. Keep it under 50 words"
        return await model.query(
            [
                SystemInput(
                    text="You are a pirate, answer in the speaking style of a pirate. Keeps responses under 10 words"
                ),
                TextInput(text=prompt * 50),
            ],
        )

    tasks = [run_single() for _ in range(size)]

    out_tokens: list[int] = []

    for f in tqdm(asyncio.as_completed(tasks), total=size):
        result = await f
        out_tokens.append(result.metadata.out_tokens)

    print(f"Out tokens: {out_tokens}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run load-oriented extras")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    embeddings_parser = subparsers.add_parser("embeddings")
    embeddings_parser.add_argument(
        "provider", nargs="?", default="openai", choices=["openai", "azure"]
    )
    embeddings_parser.add_argument("-s", "--size", type=int, default=500)

    stress_parser = subparsers.add_parser("stress")
    stress_parser.add_argument(
        "model", nargs="?", default="openai/gpt-5-nano-2025-08-07"
    )
    stress_parser.add_argument("-s", "--size", type=int, default=100)

    args = parser.parse_args()

    if args.mode == "embeddings":
        match args.provider:
            case "openai":
                model = OpenAIModel("dummy")
            case "azure":
                model = AzureOpenAIModel("dummy")
            case _:
                raise Exception("Invalid provider")
        await embeddings(model, size=args.size)
    elif args.mode == "stress":
        await stress(get_registry_model(args.model), args.size)


if __name__ == "__main__":
    setup(disable_logging=True)
    asyncio.run(main())
