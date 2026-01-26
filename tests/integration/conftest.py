import pytest


@pytest.fixture()
def require_model_flag(request: pytest.FixtureRequest):
    """Skip this test unless --model flag is provided"""
    if not request.config.getoption("--model", None):
        pytest.skip("Hard question tests require --model flag")
