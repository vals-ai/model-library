"""
Integration tests for Google provider delegate mode (native=False -> OpenAI-compat endpoint).
"""

from model_library.base import LLMConfig
from model_library.providers.google import GoogleModel
from tests.test_helpers import assert_basic_result

_MODEL = "gemini-3.1-pro-preview"


async def test_google_delegate_reasoning_effort():
    model = GoogleModel(
        _MODEL,
        config=LLMConfig(native=False, reasoning=True, reasoning_effort="high"),
    )
    result = await model.query(
        "A train leaves NYC at 3pm going 60mph toward Boston (200 miles away). "
        "Another leaves Boston at 4pm going 40mph toward NYC. When do they meet?"
    )
    assert_basic_result(result)
    assert result.reasoning
