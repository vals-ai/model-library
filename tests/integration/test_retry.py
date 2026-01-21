"""
Unit tests for retry logic.
"""

import asyncio

from model_library.base import TextInput
from model_library.exceptions import (
    MaxContextWindowExceededError,
)
from model_library.registry_utils import get_registry_model


# @parametrize_all_models
async def test_context_window_error_caught_real_api_calls():
    model_key = "google/gemini-2.5-flash-lite"
    model = get_registry_model(model_key)

    long_input = "Break" * 1000000
    input = [TextInput(text=long_input)]

    try:
        await asyncio.wait_for(model.query(input), timeout=15.0)
        assert False, "No exception raised"
    except MaxContextWindowExceededError:
        assert True
    except Exception as e:
        assert False, str(e)
