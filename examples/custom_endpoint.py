"""
Custom Endpoint Example

Demonstrates how to use custom_endpoint to point any provider
at a custom API endpoint (e.g., a proxy, self-hosted model, or
alternative API-compatible service).
"""

import asyncio

from pydantic import SecretStr

from model_library.base import LLMConfig, TextInput
from model_library.registry_utils import get_raw_model

from .setup import console_log, setup


async def custom_endpoint():
    """Use a custom endpoint with an OpenAI-compatible model."""
    console_log("\n--- Custom Endpoint (OpenAI) ---\n")

    config = LLMConfig(
        custom_endpoint="https://api.openai.com/v1",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_raw_model("openai/gpt-4o-mini", config=config)

    await model.query([TextInput(text="Say hello in one sentence.")])


async def custom_endpoint_with_registry():
    """Use a custom endpoint with a registry model (preserves default config)."""
    console_log("\n--- Custom Endpoint (Registry) ---\n")

    from model_library.registry_utils import get_registry_model

    config = LLMConfig(
        custom_endpoint="https://api.openai.com/v1",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_registry_model("openai/gpt-4o-mini", override_config=config)

    await model.query([TextInput(text="Say hello in one sentence.")])


async def main():
    await custom_endpoint()
    await custom_endpoint_with_registry()


if __name__ == "__main__":
    setup()
    asyncio.run(main())
