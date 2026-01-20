import pytest

from examples.setup import setup


@pytest.fixture(autouse=True)
def setup_fixture():
    setup()


@pytest.fixture(autouse=True)
def clear_client_registry():
    """Clear the client registry between tests."""
    yield
    from model_library.base.base import client_registry

    client_registry.clear()
