# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import logging
from typing import Any, Awaitable, Callable

from pydantic import SecretStr

from model_library.base import LLM, LLMConfig, TextInput
from model_library.exceptions import BackoffRetryException
from model_library.registry_utils import get_raw_model, get_registry_model

from examples.setup import console_log, setup


async def custom_endpoint() -> None:
    """Use a custom endpoint with an OpenAI-compatible model."""
    console_log("\n--- Custom Endpoint (OpenAI) ---\n")

    config = LLMConfig(
        custom_endpoint="https://api.openai.com/v1",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_raw_model("openai/gpt-5.4-nano-2026-03-17", config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


async def custom_endpoint_with_registry() -> None:
    """Use a custom endpoint with a registry model and default config."""
    console_log("\n--- Custom Endpoint (Registry) ---\n")

    config = LLMConfig(
        custom_endpoint="https://api.openai.com/v1",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_registry_model("openai/gpt-5.4-nano-2026-03-17", override_config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


def is_context_length_error(error_str: str) -> bool:
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


def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    logger = logging.getLogger("llm.decorator")

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if is_context_length_error(str(e).lower()):
                logger.warning(f"Context length error detected: {e}")
                raise BackoffRetryException(f"Context length error: {e}")
            raise

    return wrapper


async def custom_retrier_context(model: LLM) -> None:
    """Run a large-input query that should trigger context-length handling."""
    console_log("\n--- Custom Retrier ---\n")

    model.custom_retrier = decorator
    try:
        await model.query(
            [
                TextInput(
                    text="What is QSBS? Explain your thinking in detail and make it concise. "
                    * 1000
                )
                for _ in range(20)
            ],
        )
    except BackoffRetryException:
        console_log("Custom retrier raised BackoffRetryException")
    except Exception as e:
        raise RuntimeError("Custom retrier did not raise BackoffRetryException") from e


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider setup extras")
    parser.add_argument(
        "mode",
        choices=["endpoint", "registry-endpoint", "custom-retrier"],
        help="Provider setup demo to run",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        help="Registry model for custom-retrier mode",
    )
    args = parser.parse_args()

    if args.mode == "endpoint":
        await custom_endpoint()
    elif args.mode == "registry-endpoint":
        await custom_endpoint_with_registry()
    elif args.mode == "custom-retrier":
        await custom_retrier_context(get_registry_model(args.model))


if __name__ == "__main__":
    setup()
    asyncio.run(main())
