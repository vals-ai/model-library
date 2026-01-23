import asyncio
import time
from logging import DEBUG
from typing import Any, Coroutine

from tqdm import tqdm

from model_library.base import LLM, TextInput, TokenRetryParams
from model_library.logging import set_logging
from model_library.registry_utils import get_registry_model
from model_library.retriers.token import set_redis_client

from .setup import console_log, setup


async def token_retry(model: LLM):
    console_log("\n--- Token Retry ---\n")
    await model.query(
        [
            TextInput(
                # text="What is QSBS? Explain your thinking in detail and make it concise"
                text="dwadwadwadawdLong argument of cats vs dogs" * 5000
                + "Ignore the previous junk, tell me a very long story about the cats and the dogs. And yes, I do want an actual story, I just have no choice but to include the junk before, believe me."
            )
        ],
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run basic examples with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="google/gemini-2.5-flash",
        type=str,
        help="Model endpoint (default: google/gemini-2.5-flash)",
    )
    args = parser.parse_args()

    set_logging(level=DEBUG)

    model = get_registry_model(args.model)
    model.logger.info(model)

    limit = await model.get_rate_limit()
    model.logger.info(limit)

    import redis.asyncio as redis

    # NOTE: make sure you have redis running locally
    # docker run -d -p 6379:6379 redis:latest

    redis_client = redis.Redis(
        host="localhost", port=6379, decode_responses=True, max_connections=None
    )
    set_redis_client(redis_client)

    provider_tokenizer_input_modifier = 1
    dataset_output_modifier = 0.001

    limit = 100_000
    await model.init_token_retry(
        token_retry_params=TokenRetryParams(
            input_modifier=provider_tokenizer_input_modifier,
            output_modifier=dataset_output_modifier,
            use_dynamic_estimate=True,
            limit=limit,
        )
    )
    tasks: list[Coroutine[Any, Any, None]] = []
    for _ in range(200):
        tasks.append(token_retry(model))

    start = time.time()
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro
    finish = time.time() - start
    console_log(f"Finished in {finish:.1f}s")


if __name__ == "__main__":
    setup()
    asyncio.run(main())
