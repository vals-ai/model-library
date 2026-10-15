"""Inline image mimes may arrive with or without the `image/` prefix."""

import pytest

from model_library.base import LLMConfig
from model_library.base.input import FileWithBase64, FileWithBytes
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google.google import GoogleModel
from model_library.providers.mistral import MistralModel
from model_library.providers.xai import XAIConfig, XAIModel

_MIMES = pytest.mark.parametrize("mime", ["png", "image/png"])


def _image(mime: str) -> FileWithBase64:
    return FileWithBase64(type="image", name="a.png", mime=mime, base64="dGVzdA==")


@_MIMES
async def test_anthropic_media_type(mime: str):
    parsed = await AnthropicModel("claude-test").parse_image(_image(mime))
    assert parsed["source"]["media_type"] == "image/png"


@_MIMES
async def test_bedrock_image_format(mime: str):
    """Convert the stored MIME type to Bedrock's short image format."""
    parsed = await AmazonModel("amazon-test").parse_image(_image(mime))
    assert parsed["image"] == {"format": "png", "source": {"bytes": b"test"}}


@_MIMES
@pytest.mark.parametrize("append_type", ["base64", "bytes"])
async def test_google_mime_type(mime: str, append_type: str):
    """Normalize both image representations without changing their payload."""
    image: FileWithBase64 | FileWithBytes = _image(mime)
    if append_type == "bytes":
        image = FileWithBytes(type="image", name="a.png", mime=mime, data=b"test")
    part = await GoogleModel("gemini-2.5-flash", config=LLMConfig()).parse_image(image)
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == b"test"


@_MIMES
async def test_mistral_data_url(mime: str):
    parsed = await MistralModel("mistral-test").parse_image(_image(mime))
    assert parsed["image_url"] == "data:image/png;base64,dGVzdA=="


@_MIMES
async def test_xai_native_data_url(mime: str):
    model = XAIModel(
        "grok-4.7",
        config=LLMConfig(provider_config=XAIConfig(native=True)),
    )
    parsed = await model.parse_image(_image(mime))
    assert parsed.image_url.image_url == "data:image/png;base64,dGVzdA=="
