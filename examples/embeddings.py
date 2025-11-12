import asyncio

from model_library.providers.azure import AzureOpenAIModel
from model_library.providers.openai import OpenAIModel

from .setup import console_log, setup


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


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run embeddings examples with a provider"
    )
    parser.add_argument(
        "provider",
        nargs="?",
        default="openai",
        choices=["openai", "azure"],
        type=str,
        help="Provider (default: openai)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=500,
        help="Number of tasks (default: 500)",
    )
    args = parser.parse_args()

    match args.provider:
        case "openai":
            model = OpenAIModel("dummy")
        case "azure":
            model = AzureOpenAIModel("dummy")
        case _:
            raise Exception("Invalid provider")
    model.logger.info(model)

    await embeddings(model, size=args.size)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
