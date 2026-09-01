import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from model_library.base.input import FileWithBase64, TextInput
from model_library.providers.xai import XAIModel

# The unit-test fixture replaces provider get_client methods at runtime.
_XAI_COUNT_TOKENS = XAIModel.count_tokens


class FakeChat:
    """Collects appended messages the way xai_sdk's Chat does."""

    def __init__(self):
        self.messages: list[Any] = []

    def append(self, message: Any):
        self.messages.append(message)


async def test_count_tokens_counts_locally_and_ignores_image_payloads():
    """Token counting must not call the provider's tokenize RPC.

    Images are tokenized by a separate encoder on the provider side, so feeding
    their base64 to a text tokenizer both overcounts wildly and (over gRPC's
    4 MiB decode limit) fails the request outright.
    """
    payload = base64.b64encode(b"\x00" * 5_000_000).decode()
    client = MagicMock()
    client.chat.create.return_value = FakeChat()
    client.tokenize.tokenize_text = AsyncMock(return_value=[1, 2, 3])

    model = XAIModel("grok-3-mini")
    with patch.object(XAIModel, "get_client", return_value=client):
        count = await _XAI_COUNT_TOKENS(
            model,
            [
                TextInput(text="read this tax certificate"),
                FileWithBase64(
                    type="image", name="cert.png", mime="png", base64=payload
                ),
            ],
        )

    client.tokenize.tokenize_text.assert_not_called()
    # the payload alone would tokenize to over a million tokens
    assert 0 < count < 100
