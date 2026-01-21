"""
Integration tests for file handling (real API when available).
Tests both image and PDF file inputs with GenAI and Vertex AI.
"""

import pytest

from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models
from tests.test_helpers import (
    get_example_file_base64_input,
    get_example_image_base64_input,
)


@parametrize_all_models
async def test_image_base64(model_key: str):
    model = get_registry_model(model_key)
    if not model.supports_images:
        pytest.skip("Model does not support images")

    image_color = "red"
    input = get_example_image_base64_input(image_color=image_color)

    result = await model.query(input)
    assert image_color in (result.output_text or "").lower()


@parametrize_all_models
async def test_file_base64(model_key: str):
    model = get_registry_model(model_key)
    if not model.supports_files:
        pytest.skip("Model does not support files")

    input = get_example_file_base64_input()

    result = await model.query(input)
    assert "pineapple" in (result.output_text or "").lower()
