"""
Unit tests for count_tokens functionality
"""

import pytest

from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models
from tests.test_helpers import get_example_tool_input


@parametrize_all_models
async def test_empty_input_returns_zero(model_key: str):
    """Test that count_tokens returns 0 for empty input"""
    model = get_registry_model(model_key)

    token_count = await model.count_tokens([])

    assert isinstance(token_count, int)
    assert token_count == 0


@parametrize_all_models
async def test_count_tokens_with_text_and_tools(model_key: str):
    """Test that count_tokens returns positive counts for non-empty input"""
    model = get_registry_model(model_key)

    using_provider_api = "count_tokens" in model.__class__.__dict__

    if using_provider_api:
        pytest.skip("Using provider's count_token api")

    input, system_prompt, tools = get_example_tool_input()

    token_count = await model.count_tokens(
        input,
        tools=tools,
        system_prompt=system_prompt,
    )

    assert isinstance(token_count, int)
    assert token_count > 150
