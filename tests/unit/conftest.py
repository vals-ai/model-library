import importlib
import pkgutil
from types import ModuleType
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from model_library import providers
from model_library.base import LLM, QueryResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM subclass."""
    # Create subclass dynamically
    MockLLM = type(
        "MockLLM",
        (LLM,),
        {
            "_get_default_api_key": Mock(return_value="mock_api_key"),
            "get_client": Mock(return_value=MagicMock()),
            "build_body": AsyncMock(return_value={}),
            "_query_impl": AsyncMock(return_value=QueryResult()),
            "parse_input": AsyncMock(return_value=None),
            "parse_image": AsyncMock(return_value=None),
            "parse_file": AsyncMock(return_value=None),
            "parse_tools": AsyncMock(return_value=None),
            "upload_file": AsyncMock(return_value=None),
        },
    )

    return MockLLM("mock_model", "mock_provider")


def walk_provider_modules(callback: Callable[[Any], None]):
    """Walk through all provider modules and call callback for each."""
    for _, modname, _ in pkgutil.walk_packages(
        providers.__path__, providers.__name__ + "."
    ):
        mod = importlib.import_module(modname)
        callback(mod)


def mock_llm_subclasses(method_name: str, mock_func: Callable[..., Any]):
    """Mock a specific method on all LLM subclasses."""

    def apply_mock(mod: ModuleType):
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, LLM) and obj is not LLM:
                setattr(obj, method_name, mock_func)

    walk_provider_modules(apply_mock)


class MockSequence:
    def __len__(self):
        return -1


@pytest.fixture(autouse=True)
def mock_all_get_client():
    """Mock get_client for all LLM provider subclasses."""

    mock_client = MagicMock(return_value=AsyncMock())

    # for xai tokenize
    mock_client.return_value.tokenize.tokenize_text = AsyncMock(
        return_value=MockSequence()
    )
    mock_llm_subclasses("get_client", mock_client)

    yield


@pytest.fixture(autouse=True)
def mock_all_mock_model_library_settings():
    """Patch model_library_settings in all provider modules."""

    class FakeSettings:
        def __getattr__(self, name: str) -> str:
            match name:
                case "AWS_DEFAULT_REGION":
                    return "us-east-1"
                case _:
                    return f"mock_ENV_{name}"

        def get(self, name: str, default: str = "") -> str:
            try:
                return getattr(self, name)
            except AttributeError:
                return default

    fake = FakeSettings()

    def patch_settings(mod: ModuleType):
        if hasattr(mod, "model_library_settings"):
            setattr(mod, "model_library_settings", fake)

    walk_provider_modules(patch_settings)
    yield
