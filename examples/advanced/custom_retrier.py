import asyncio
import logging
from typing import Any, Awaitable, Callable

from model_library.base import (
    LLM,
    RetrierType,
    TextInput,
)
from model_library.exceptions import (
    BackoffRetryException,
    RetryException,
    retry_llm_call,
)
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup


def is_context_length_error(error_str: str) -> bool:
    """
    Very simple context length error detection
    """
    return any(
        keyword in error_str
        for keyword in [
            "context length",
            "context window",
            "too many tokens",
            "maximum context length",
            "context_length_exceeded",
            "prompt is too long",
            "input is too long",
            "exceeds the context window",
        ]
    )


def custom_retrier(logger: logging.Logger) -> RetrierType:
    """
    Custom retrier that raised BackoffRetryException for context length errors
    Custom retries takes in a logger. It replaces the backoff retrier. Immediate retries still function.
    """

    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        """
        Decorator must return wrapper function
        """

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # detect context length errors and retry with backoff
                if is_context_length_error(str(e).lower()):
                    logger.warning(f"Context length error detected: {e}")
                    # for simplicty, we don't actually retry
                    raise BackoffRetryException(f"Context length error: {e}")

                raise

        return wrapper

    return decorator


async def custom_retrier_context(model: LLM):
    """
    Large input that will trigger context length errors
    """

    console_log("\n--- Custom Retrier ---\n")

    model.custom_retrier = custom_retrier
    try:
        await model.query(
            [
                TextInput(
                    text="What is QSBS? Explain your thinking in detail and make it concise. "
                    * 1000
                )
                for _ in range(20)
            ],  # 20 messages, each message has large input (x1000)
        )
    except BackoffRetryException:
        console_log("Custom retrier raised BackoffRetryException")
    except Exception:
        raise Exception("Custom retrier did not raised BackoffRetryException")


async def custom_retrier_callback(model: LLM):
    """
    Add a callback to the our default retrier
    """

    console_log("\n--- Custom Retrier Callback ---\n")

    def callback(tries: int, exception: Exception | None, elapsed: float, wait: float):
        print(f"Logging retry #{tries}")
        if tries > 1:
            raise Exception("Reached retry 2")

    def custom_retrier(logger: logging.Logger):
        return retry_llm_call(
            logger,
            max_tries=3,
            max_time=500,
            backoff_callback=callback,
        )

    model.custom_retrier = custom_retrier

    def simulate_retry(*args: object, **kwargs: object):
        raise RetryException("Simulated failure")

    model._query_impl = simulate_retry  # pyright: ignore[reportPrivateUsage]
    try:
        await model.query("Ping!")
    except Exception as e:
        console_log(f"Caugh exception: {e}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a model with a custom context length retrier"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    base_model = get_registry_model(args.model)

    await custom_retrier_context(base_model)
    await custom_retrier_callback(base_model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
