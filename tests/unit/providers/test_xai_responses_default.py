"""xAI routes through the OpenAI-compatible Responses API unless `native: true`."""

import io
from unittest.mock import AsyncMock, patch

import pytest

from model_library.agent.tool import NativeWebSearch
from model_library.base import LLMConfig
from model_library.base.input import (
    FileWithBase64,
    FileWithId,
    TextInput,
    ToolBody,
    ToolDefinition,
)
from model_library.exceptions import BadInputError
from model_library.providers.openai import OpenAIModel
from model_library.providers.xai import XAIConfig, XAIModel
from model_library.registry_utils import get_registry_model


def _delegate(model: XAIModel) -> OpenAIModel:
    assert isinstance(model.delegate, OpenAIModel)
    return model.delegate


def test_default_uses_responses_delegate():
    model = XAIModel("grok-4.7")
    assert model.provider_config.native is False
    assert model.native is False
    delegate = _delegate(model)
    assert delegate.use_completions is False
    assert delegate.provider == "xai"
    assert delegate.custom_endpoint == "https://api.x.ai/v1"


def test_grok_3_mini_reasoning_uses_regional_endpoint():
    model = XAIModel("grok-3-mini-reasoning")
    assert _delegate(model).custom_endpoint == "https://us-west-1.api.x.ai/v1"


def test_native_true_skips_delegate():
    model = XAIModel(
        "grok-4.7", config=LLMConfig(provider_config=XAIConfig(native=True))
    )
    assert model.native is True
    assert model.delegate is None


def test_registry_provider_properties_select_native():
    default = get_registry_model("grok/grok-4.7")
    assert isinstance(default, XAIModel)
    assert isinstance(default.delegate, OpenAIModel)
    assert default.delegate.provider == "grok"
    assert default.delegate.search_tool == {"type": "web_search"}

    native = get_registry_model(
        "grok/grok-4.7", LLMConfig(provider_config=XAIConfig(native=True))
    )
    assert isinstance(native, XAIModel)
    assert native.delegate is None


async def test_responses_body_requests_encrypted_reasoning():
    model = XAIModel("grok-4.7", config=LLMConfig(reasoning=True))
    body = await _delegate(model).build_body([TextInput(text="hello")], tools=[])
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["store"] is False
    assert body["reasoning"]["summary"] == "auto"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


async def test_responses_rejects_image_file_id_as_unsupported():
    model = XAIModel("grok-4.7", config=LLMConfig(supports_images=True))
    with pytest.raises(BadInputError, match="does not support image file_id"):
        await _delegate(model).parse_image(
            FileWithId(type="image", name="a.png", mime="image/png", file_id="f1")
        )


async def test_responses_body_encodes_image_file_and_tools():
    model = XAIModel("grok-4.7", config=LLMConfig(supports_images=True))
    tool = ToolDefinition(
        name="echo",
        body=ToolBody(name="echo", description="echo", properties={}, required=[]),
    )
    body = await _delegate(model).build_body(
        [
            TextInput(text="describe"),
            FileWithBase64(type="image", name="a.png", mime="image/png", base64="AA"),
            FileWithId(type="file", name="a.pdf", mime="application/pdf", file_id="f1"),
        ],
        tools=[tool, NativeWebSearch().definition],
    )
    content = body["input"][0]["content"]
    assert content[1] == {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,AA",
    }
    assert content[2] == {"type": "input_file", "file_id": "f1"}
    assert [t["type"] for t in body["tools"]] == ["function", "web_search"]


async def test_upload_file_delegates_to_openai_client():
    model = XAIModel("grok-4.7")
    uploaded = FileWithId(type="file", name="a.pdf", mime="application/pdf", file_id="f1")
    with patch.object(
        OpenAIModel, "upload_file", new=AsyncMock(return_value=uploaded)
    ) as upload:
        result = await model.upload_file("a.pdf", "application/pdf", io.BytesIO(b"x"))
    assert result is uploaded
    upload.assert_awaited_once()
