from typing import Literal

import pytest

from model_library.base import LLMConfig
from model_library.providers.openai import OpenAIConfig, OpenAIModel


@pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
async def test_verbosity_added_to_body(verbosity: Literal["low", "medium", "high"]):
    """Test that verbosity is correctly added to request body."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=verbosity)),
    )

    body = await model.build_body([], tools=[])

    assert "text" in body
    assert body["text"]["verbosity"] == verbosity


async def test_verbosity_not_in_body_when_none():
    """Test that text field is not added when verbosity is None."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=None)),
    )

    body = await model.build_body([], tools=[])

    assert "text" not in body
