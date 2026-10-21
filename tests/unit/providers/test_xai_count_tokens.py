import base64

from model_library.base.input import FileWithBase64, TextInput
from model_library.providers.xai import XAIModel

async def test_count_tokens_ignores_image_payloads():
    """Images are tokenized by a separate encoder on the provider side, so
    feeding their base64 to a text tokenizer overcounts wildly."""
    payload = base64.b64encode(b"\x00" * 5_000_000).decode()
    model = XAIModel("grok-4.7")
    count = await model.count_tokens(
        [
            TextInput(text="read this tax certificate"),
            FileWithBase64(type="image", name="cert.png", mime="png", base64=payload),
        ],
    )

    # the payload alone would tokenize to over a million tokens
    assert 0 < count < 100
