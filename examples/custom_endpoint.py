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

    model = get_raw_model("openai/gpt-5.4-nano-2026-03-17", config=config)

    await model.query([TextInput(text="Say hello in one sentence.")])


async def custom_endpoint_with_registry():
    """Use a custom endpoint with a registry model (preserves default config)."""
    console_log("\n--- Custom Endpoint (Registry) ---\n")

    from model_library.registry_utils import get_registry_model

    config = LLMConfig(
        custom_endpoint="https://api.openai.com/v1",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_registry_model("openai/gpt-5.4-nano-2026-03-17", override_config=config)

    await model.query([TextInput(text="Say hello in one sentence.")])


async def google_delegate_to_openai_completions():
    """Route Google through OpenAI chat completions via native=False.

    When native=False, GoogleModel delegates to OpenAIModel(use_completions=True).
    By default it targets Google's OpenAI-compat endpoint
    (https://generativelanguage.googleapis.com/v1beta/openai/) using
    GOOGLE_API_KEY, but both can be overridden via custom_endpoint /
    custom_api_key — useful for proxies or other OpenAI-compatible backends.

    Standard LLMConfig sampling params (max_tokens, temperature, top_p) are
    forwarded in the chat-completions body. top_k is forwarded via extra_body
    but Gemini's OpenAI-compat proxy ignores it; use native mode for top_k
    on Gemini.
    """
    console_log("\n--- Google Delegate -> OpenAI Chat Completions ---\n")

    # custom_endpoint / custom_api_key are optional — defaults point at
    # Google's OpenAI-compat endpoint using GOOGLE_API_KEY from env.
    config = LLMConfig(
        native=False,
        max_tokens=64,
        temperature=0.5,
        top_p=0.9,
        custom_endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
        custom_api_key=SecretStr("your-api-key-here"),
    )

    model = get_raw_model("google/gemini-2.5-flash", config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


async def google_native_with_top_k():
    """top_k is only honored on Google's native genai path."""
    console_log("\n--- Google Native (top_k) ---\n")

    config = LLMConfig(
        max_tokens=64,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
    )
    model = get_raw_model("google/gemini-2.5-flash", config=config)
    await model.query([TextInput(text="Say hello in one sentence.")])


async def main():
    await custom_endpoint()
    await custom_endpoint_with_registry()
    await google_delegate_to_openai_completions()
    await google_native_with_top_k()


if __name__ == "__main__":
    setup()
    asyncio.run(main())
