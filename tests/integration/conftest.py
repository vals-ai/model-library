import pytest

from examples.setup import setup


@pytest.fixture(autouse=True)
def setup_fixture():
    setup()
