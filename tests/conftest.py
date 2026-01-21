"""
Test configuration and fixtures.
"""

from typing import Any, Callable, TypeVar

import pytest

from examples.setup import setup
from model_library.registry_utils import (
    get_model_names,
)

F = TypeVar("F", bound=Callable[..., Any])


def parametrize_all_models(func: F) -> F:
    """Decorator to parametrize test with all available models"""
    all_models = get_model_names()
    all_models = [
        model
        for model in all_models
        if "research" not in model and "dumbmar" not in model
    ]
    return pytest.mark.parametrize("model_key", all_models)(func)


def parametrize_models_for_provider(provider_name: str) -> Callable[[F], F]:
    """Decorator to parametrize test with all available models from a specific provider"""

    def decorator(func: F) -> F:
        models = get_model_names(provider_name)
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Apply markers to tests based on directory."""
    for item in items:
        if "tests/unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
