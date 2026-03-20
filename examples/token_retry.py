import asyncio
import logging
import time
import uuid
from typing import Any, Coroutine

from tqdm import tqdm

from model_library.base import LLM, TextInput, TokenRetryParams
from model_library.logging import set_logging
from model_library.registry_utils import get_registry_model
from model_library.retriers.token import benchmark_queue, set_redis_client

from .setup import console_log, setup


async def token_retry(model: LLM):
    await model.query(
        [TextInput(text="Tell me a 200 word story about a cat and a dog")],
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run basic examples with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-mini-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-mini-2025-08-07)",
    )
    parser.add_argument(
        "--no-benchmark-queue",
        action="store_true",
        help="Disable benchmark queue (runs execute concurrently without serialization)",
    )
    args = parser.parse_args()

    set_logging(level=logging.ERROR)

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

    requests_per_run = 50
    num_runs = 4

    async def run(run_id: str, n: int):
        """A single benchmark run — waits for its queue slot, then fires requests."""
        short_id = run_id[:8]
        bar = tqdm(
            total=n,
            desc=f"{short_id} waiting",
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
        )

        async with benchmark_queue(
            model_registry_key=model._client_registry_key_model_specific,  # pyright: ignore[reportPrivateUsage]
            run_id=run_id,
            logger=model.logger,
            enabled=not args.no_benchmark_queue,
            total_requests=n if not args.no_benchmark_queue else None,
        ):
            bar.set_description(f"{short_id} running")

            tasks: list[Coroutine[Any, Any, None]] = []
            for _ in range(n):
                tasks.append(token_retry(model))

            start = time.time()
            for coro in asyncio.as_completed(tasks):
                await coro
                bar.update(1)

            bar.set_description(f"{short_id} done ({time.time() - start:.1f}s)")
            bar.close()

    # benchmark_queue serializes concurrent runs for the same model so they
    # don't compete for the same TPM budget. Each run waits its turn, then
    # releases early once all its requests have been dispatched.

    console_log("\n--- Token Retry ---\n")
    console_log(
        "Runs start once all previous requests are dispatched (not completed)\n"
    )

    runs = [run(uuid.uuid4().hex, requests_per_run) for _ in range(num_runs)]
    await asyncio.gather(*runs)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
