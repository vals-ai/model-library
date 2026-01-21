"""
Integration tests for model completions (real API when available).
"""

from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models
from tests.test_helpers import assert_basic_result


@parametrize_all_models
async def test_basic(model_key: str):
    model = get_registry_model(model_key)
    result = await model.query("Ping")

    assert_basic_result(result)
