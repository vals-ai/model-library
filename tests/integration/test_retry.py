"""
Unit tests for retry logic.
"""

import asyncio
import json
import random

import pytest

from model_library.base import LLMConfig
from model_library.exceptions import (
    MaxContextWindowExceededError,
)
from model_library.registry_utils import get_registry_model


@pytest.mark.integration
async def test_context_window_error_caught_real_api_calls():
    """
    Goes through all the model providers and parsers 2 models each,
    ensures that we are correctly identifying context window errors AND that they are not changing depending on the model

    Makes real api calls and should only be ran during important updates to ensure that if an error message changes, we catch it and can update the regex patterns accordingly
    """

    # Parse directly from the all_models.json file, its easier to get what we need from here than using the model registry
    with open("vals_model_proxy/config/all_models.json", "r") as f:
        model_config = json.load(f)

    model_keys = [
        key
        for key in model_config.keys()
        if not model_config[key].get("metadata", {}).get("deprecated", False)
    ]

    # Group all the models together by the provider
    provider_to_model: dict[str, list[str]] = {}
    providers_with_issues = ["vals", "databricks", "zai"]
    for model in model_keys:
        if "/" not in model or any(
            provider in model for provider in providers_with_issues
        ):
            continue

        provider = model.split("/", 1)[0]
        if provider not in provider_to_model:
            provider_to_model[provider] = []

        provider_to_model[provider].append(model)

    # Randomly select 2 models from each provider inside of the mapping
    models_to_test: list[str] = []
    for provider, models in provider_to_model.items():
        models_to_test.extend(random.sample(models, min(2, len(models))))

    async def test_model(model_str: str) -> tuple[bool, str]:
        """
        Simple model query method that creates a long input and catches the exception,
        only returns True if the exception is a MaxContextWindowExceededError, False otherwise
        """
        model = get_registry_model(
            model_str, override_config=LLMConfig(supports_batch=False)
        )

        model.logger.disabled = True

        long_input = "Hello" * 100000

        try:
            await asyncio.wait_for(model.query(long_input), timeout=15.0)
            return False, "No exception raised"
        except asyncio.TimeoutError:
            return False, "Timeout after 15 seconds"
        except MaxContextWindowExceededError:
            return True, "MaxContextWindowExceededError raised"
        except Exception as e:
            return False, str(e)

    message = "Models to test:\n" + "\n".join(
        [
            f"Provider: {model_str.split('/')[0]}, Model: {model_str.split('/')[1]}"
            for model_str in models_to_test
        ]
    )

    print(message)

    tasks = [test_model(model_str) for model_str in models_to_test]
    results = await asyncio.gather(*tasks)

    for model_str, result in zip(models_to_test, results):
        assert result[0], (
            f"Model {model_str} should have raised MaxContextWindowExceededError: {result[1]}"
        )
