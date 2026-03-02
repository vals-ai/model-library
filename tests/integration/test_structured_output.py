"""
Integration tests for structured outputs.
"""

import base64
from io import BytesIO

import pytest
from pydantic import BaseModel

from model_library.base import FileWithBase64, TextInput
from model_library.registry_utils import get_registry_config, get_registry_model
from tests.conftest import parametrize_all_models


def get_model_if_supports_output_schema(model_key: str):
    config = get_registry_config(model_key)
    if not config or not config.supports.output_schema:
        pytest.skip("Model does not support output_schema")
    return get_registry_model(model_key)


def create_test_image() -> bytes:
    """Create a simple test image (red square)."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class Color(BaseModel):
    """Pydantic model for color analysis."""

    color: str


class Inner(BaseModel):
    """Inner model for nested structure test."""

    value: str


class Outer(BaseModel):
    """Outer model for nested structure test."""

    inner: Inner


@parametrize_all_models
async def test_structured_output_pydantic(model_key: str):
    model = get_model_if_supports_output_schema(model_key)

    result = await model.query(
        [TextInput(text="Provide a nested structure with inner value 'hello world'")],
        output_schema=Outer,
    )

    assert isinstance(result.output_parsed, Outer)
    assert isinstance(result.output_parsed.inner, Inner)
    assert result.output_parsed.inner.value.lower() == "hello world"


@parametrize_all_models
async def test_structured_output_dict_schema(model_key: str):
    model = get_model_if_supports_output_schema(model_key)

    result = await model.query(
        [TextInput(text="What color is the sky? Respond with just the color name.")],
        output_schema={
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
            "additionalProperties": False,
        },
    )

    assert isinstance(result.output_parsed, dict)
    assert "color" in result.output_parsed


@parametrize_all_models
async def test_structured_output_with_image(model_key: str):
    model = get_model_if_supports_output_schema(model_key)
    if not model.supports_images:
        pytest.skip("Model does not support images")

    image_bytes = create_test_image()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    result = await model.query(
        [
            TextInput(text="What color is the image? Respond with just the color name."),
            FileWithBase64(type="image", name="red_image.png", mime="png", base64=encoded),
        ],
        output_schema=Color,
    )

    assert isinstance(result.output_parsed, Color)
    assert result.output_parsed.color.lower() == "red"
