from typing import Any
from unittest.mock import MagicMock, patch

from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.providers.xai import XAIConfig, XAIModel

_NATIVE = LLMConfig(provider_config=XAIConfig(native=True))


class FakeChat:
    def __init__(self):
        self.messages: list[Any] = []

    def append(self, message: Any):
        self.messages.append(message)


def _client() -> MagicMock:
    client = MagicMock()
    client.chat.create.return_value = FakeChat()
    return client


async def test_reasoning_model_requests_encrypted_content():
    model = XAIModel("grok-4.7", config=_NATIVE)
    model.reasoning = True
    with patch.object(XAIModel, "get_client", return_value=_client()):
        body = await model.build_body([TextInput(text="hello")], tools=[])

    assert body["use_encrypted_content"] is True


async def test_non_reasoning_model_does_not_request_encrypted_content():
    model = XAIModel("grok-3-latest", config=_NATIVE)
    model.reasoning = False
    with patch.object(XAIModel, "get_client", return_value=_client()):
        body = await model.build_body([TextInput(text="hello")], tools=[])

    assert "use_encrypted_content" not in body
