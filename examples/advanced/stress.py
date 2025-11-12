import asyncio

from model_library.base import LLM
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup


async def stress(model: LLM, size: int):
    from tqdm import tqdm

    console_log("\n--- Stress ---\n")

    async def run_single():
        prompt = "Tell me a bedtime story. Keep it under 50 words"
        return await model.query(
            prompt * 50,
            system_prompt="You are a pirate, answer in the speaking style of a pirate. Keeps responses under 10 words",
        )

    tasks = [run_single() for _ in range(size)]

    out_tokens: list[int] = []

    for f in tqdm(asyncio.as_completed(tasks), total=size):
        result = await f
        out_tokens.append(result.metadata.out_tokens)

    print(f"Out tokens: {out_tokens}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run stress example with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=100,
        help="Number of tasks (default: 100)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=500_000,
        help="Number of tokens (default: 500_000)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    await stress(model, args.size)


if __name__ == "__main__":
    setup(disable_logging=True)
    asyncio.run(main())
