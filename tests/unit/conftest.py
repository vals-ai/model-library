from unittest.mock import AsyncMock

import pytest

from model_library.base import LLM, QueryResult

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def mock_llm():
    """Create a mock LLM subclass."""
    # Create subclass dynamically
    MockLLM = type(
        "MockLLM",
        (LLM,),
        {
            "get_client": None,
            "_query_impl": AsyncMock(return_value=QueryResult()),
            "parse_input": AsyncMock(return_value=None),
            "parse_image": AsyncMock(return_value=None),
            "parse_file": AsyncMock(return_value=None),
            "parse_tools": AsyncMock(return_value=None),
            "upload_file": AsyncMock(return_value=None),
        },
    )

    return MockLLM("mock_model", "mock_provider")
