import pytest


@pytest.fixture(autouse=True)
def reset_client_registry():
    from model_library.base._base import client_registry

    yield
    client_registry.clear()
