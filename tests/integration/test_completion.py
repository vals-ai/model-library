"""
Integration tests for model completions (real API when available).
"""

from typing import Any

import pytest

from tests.conftest import requires_google_api, requires_mercury_api
from tests.test_helpers import assert_basic_result
from model_library.base import LLMConfig
from model_library.providers.google import GoogleModel
from model_library.registry_utils import get_registry_model


@pytest.mark.integration
@requires_google_api
class TestCompletionGenAI:
    """Test Google GenAI model completions."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, create_test_input: Any):
        model = GoogleModel("gemini-2.5-flash-lite", config=LLMConfig(native=True))
        result = await model.query(create_test_input("Say 'Hello World'"))
        assert_basic_result(result)
        assert "hello" in (result.output_text or "").lower()


@pytest.mark.integration
@requires_mercury_api
class TestCompletionMercury:
    """Test Mercury model completions."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, create_test_input: Any):
        model = get_registry_model("inception/mercury")
        result = await model.query(create_test_input("What is 2 + 2?"))
        assert_basic_result(result)

        assert result.output_text and len(result.output_text.strip()) > 0
        assert "4" in result.output_text
