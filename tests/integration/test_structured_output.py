"""
Integration tests for structured output using query_json.
Tests with OpenAI models for both image and document inputs.
"""

import base64
from io import BytesIO

import pytest
from pydantic import BaseModel

from tests.conftest import requires_openai_api
from model_library.base import FileWithBase64, FileWithId, TextInput
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model


def create_test_pdf() -> bytes:
    """Create a minimal test PDF with secret text."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 24 Tf
100 100 Td
(secret: pineapple) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000061 00000 n
0000000114 00000 n
0000000173 00000 n
trailer
<< /Root 1 0 R /Size 5 >>
startxref
243
%%EOF
"""


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


class Secret(BaseModel):
    """Pydantic model for secret extraction."""

    secret: str


class Inner(BaseModel):
    """Inner model for nested structure test."""

    value: str


class Outer(BaseModel):
    """Outer model for nested structure test."""

    inner: Inner


@pytest.mark.integration
@requires_openai_api
class TestStructuredOutputOpenAI:
    """Test OpenAI structured output with files."""

    @pytest.mark.asyncio
    async def test_structured_output_with_image(self):
        """Test query_json with an image file."""
        model = get_registry_model("openai/gpt-5-mini-2025-08-07")
        assert isinstance(model, OpenAIModel)

        if not model.supports_images:
            pytest.skip("Model does not support images")

        image_bytes = create_test_image()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        color = await model.query_json(
            [
                TextInput(
                    text="What color is the image? Respond with just the color name."
                ),
                FileWithBase64(
                    type="image",
                    name="red_image.png",
                    mime="png",
                    base64=encoded,
                ),
            ],
            pydantic_model=Color,
        )

        assert isinstance(color, Color)
        assert color.color.lower() == "red"

    @pytest.mark.asyncio
    async def test_structured_output_with_document(self):
        """Test query_json with a document file (requires file upload)."""
        model = get_registry_model("openai/gpt-5-mini-2025-08-07")
        assert isinstance(model, OpenAIModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        pdf_bytes = create_test_pdf()

        # Upload the file first to get a file_id (required for structured output)
        uploaded_file: FileWithId = await model.upload_file(
            "test.pdf", "application/pdf", BytesIO(pdf_bytes), type="file"
        )

        assert uploaded_file.file_id is not None
        assert len(uploaded_file.file_id) > 0

        secret = await model.query_json(
            [
                TextInput(text="What is the secret mentioned in this document?"),
                uploaded_file,
            ],
            pydantic_model=Secret,
        )

        assert isinstance(secret, Secret)
        assert secret.secret.lower() == "pineapple"

    @pytest.mark.asyncio
    async def test_nested_structured_output(self):
        """Test query_json with nested Pydantic models."""
        model = get_registry_model("openai/gpt-5-mini-2025-08-07")
        assert isinstance(model, OpenAIModel)

        outer = await model.query_json(
            [
                TextInput(
                    text="Provide a nested structure with inner value 'hello world'"
                )
            ],
            pydantic_model=Outer,
        )

        assert isinstance(outer, Outer)
        assert isinstance(outer.inner, Inner)
        assert outer.inner.value.lower() == "hello world"
