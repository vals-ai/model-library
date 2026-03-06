import asyncio

from model_library.registry_utils import get_registry_model

from .setup import console_log, setup


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Get rate limit for a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-mini-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-mini-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)

    key_hash = model._client_registry_key[1]  # pyright: ignore[reportPrivateUsage]
    console_log(f"Key hash: {key_hash}")

    rate_limit = await model.get_rate_limit()
    console_log(f"Rate limit: {rate_limit}")


if __name__ == "__main__":
    setup()
    asyncio.run(main())
