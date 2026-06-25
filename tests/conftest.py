"""
Test configuration and fixtures.
"""

import os
from typing import Any, Callable, TypeVar

import pytest

# Prevent MODEL_GATEWAY_URL leaking from .env/direnv into collection-time registry helpers.
os.environ.pop("MODEL_GATEWAY_URL", None)

from examples.setup import setup
from model_library import model_library_settings
from model_library.registry_utils import (
    get_model_names,
    get_registry_config,
)

F = TypeVar("F", bound=Callable[..., Any])


def _local_model_names(provider_name: str | None = None) -> list[str]:
    model_library_settings.unset("MODEL_GATEWAY_URL")
    return get_model_names(provider_name)


def is_cli_model(model: str) -> bool:
    config = get_registry_config(model)
    assert config is not None
    return config.provider_name in {"cursor", "devin"}


def parametrize_all_models(func: F) -> F:
    """Decorator to parametrize test with all available models"""
    all_models = _local_model_names()
    all_models = [
        model
        for model in all_models
        if "research" not in model
        and "dumbmar" not in model
        and not is_cli_model(model)
    ]
    return pytest.mark.parametrize("model_key", all_models)(func)


def parametrize_models_for_provider(provider_name: str) -> Callable[[F], F]:
    """Decorator to parametrize test with all available models from a specific provider"""

    def decorator(func: F) -> F:
        models = _local_model_names(provider_name)
        return pytest.mark.parametrize("model_key", models)(func)

    return decorator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Run tests for a specific model key",
    )


@pytest.fixture(autouse=True)
def filter_model_key(request: pytest.FixtureRequest):
    """Filter model_key based on --model parameter"""
    requested_model = request.config.getoption("--model", None)
    if not requested_model:
        return

    if "model_key" not in request.fixturenames:
        pytest.skip("Skipping test that is not model specific")

    model_key = request.getfixturevalue("model_key")
    if not model_key:
        return

    if requested_model and model_key != requested_model:
        pytest.skip(f"Skipping {model_key}, only running {requested_model}")


pytestmark = [pytest.mark.asyncio]


def pytest_configure(config: Any):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Tests that don't require API access")
    config.addinivalue_line(
        "markers", "integration: Integration tests that require API access"
    )

    if config.option.markexpr == "integration":
        setup()
        # Integration tests exercise provider implementations directly. Local .env
        # files may point the registry at a deployed gateway, which hides provider
        # failures and can make model-specific tests hit the wrong runtime.
        model_library_settings.unset("MODEL_GATEWAY_URL")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Apply markers to tests based on directory."""
    for item in items:
        if "tests/unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
