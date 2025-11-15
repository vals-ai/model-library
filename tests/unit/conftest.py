import importlib
import pkgutil
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_library import providers
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


@pytest.fixture(autouse=True)
def mock_model_library_settings_in_providers():
    """Patch model_library_settings in all provider modules."""

    class FakeSettings:
        def __getattr__(self, name: str) -> MagicMock:
            return MagicMock(name=f"mock_ENV_{name}")

    fake = FakeSettings()

    # walk through all provider submodules
    for _, modname, _ in pkgutil.walk_packages(
        providers.__path__, providers.__name__ + "."
    ):
        mod = importlib.import_module(modname)
        if hasattr(mod, "model_library_settings"):
            setattr(mod, "model_library_settings", fake)

    yield
