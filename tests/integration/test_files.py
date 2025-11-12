"""
Integration tests for file handling (real API when available).
Tests both image and PDF file inputs with GenAI and Vertex AI.
"""

import base64
from io import BytesIO

import pytest

from tests.conftest import requires_google_api
from tests.test_helpers import assert_basic_result
from model_library.base import FileWithBase64, FileWithId, LLMConfig, TextInput
from model_library.providers.google import GoogleModel
from model_library.providers.google.google import GoogleConfig
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
    try:
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        # Fallback: minimal valid PNG if PIL not available
        # 1x1 red pixel PNG
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )


@pytest.mark.integration
@requires_google_api
class TestFilesGenAI:
    """Test Google GenAI file handling."""

    @pytest.mark.asyncio
    async def test_image_base64(self):
        """Test GenAI with base64-encoded image."""
        model = get_registry_model(
            "google/gemini-2.5-flash-lite",
        )
        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        image_bytes = create_test_image()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        result = await model.query(
            [
                TextInput(text="What color is this image? Reply with just the color."),
                FileWithBase64(
                    type="image",
                    name="test.png",
                    mime="png",
                    base64=encoded,
                ),
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        # The image is red, so the model should mention red
        assert "red" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_pdf_base64(self):
        """Test GenAI with base64-encoded PDF."""
        model = get_registry_model("google/gemini-2.5-flash-lite")
        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        pdf_bytes = create_test_pdf()
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")

        result = await model.query(
            [
                TextInput(text="What is the secret mentioned in this PDF?"),
                FileWithBase64(
                    type="file",
                    name="test.pdf",
                    mime="application/pdf",
                    base64=encoded,
                ),
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        # The PDF contains "secret: pineapple"
        assert "pineapple" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_image_file_api(self):
        """Test GenAI with File API upload."""
        model = get_registry_model("google/gemini-2.5-flash-lite")
        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        image_bytes = create_test_image()

        # Upload file using GenAI File API
        uploaded_file: FileWithId = await model.upload_file(
            "test_image.png", "image/png", BytesIO(image_bytes)
        )

        assert uploaded_file.file_id is not None
        assert len(uploaded_file.file_id) > 0

        result = await model.query(
            [
                TextInput(text="What color is this image? Reply with just the color."),
                uploaded_file,
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        assert "red" in result.output_text.lower()


@pytest.mark.integration
@requires_google_api
class TestFilesVertex:
    """Test Google Vertex AI file handling."""

    @pytest.mark.asyncio
    async def test_image_base64(self):
        """Test Vertex AI with base64-encoded image."""
        model = get_registry_model(
            "google/gemini-2.5-flash-lite",
            override_config=LLMConfig(provider_config=GoogleConfig(use_vertex=True)),
        )
        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        image_bytes = create_test_image()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        result = await model.query(
            [
                TextInput(text="What color is this image? Reply with just the color."),
                FileWithBase64(
                    type="image",
                    name="test.png",
                    mime="png",
                    base64=encoded,
                ),
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        assert "red" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_pdf_base64(self):
        """Test Vertex AI with base64-encoded PDF."""
        model = get_registry_model(
            "google/gemini-2.5-flash-lite",
            override_config=LLMConfig(provider_config=GoogleConfig(use_vertex=True)),
        )

        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        pdf_bytes = create_test_pdf()
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")

        result = await model.query(
            [
                TextInput(text="What is the secret mentioned in this PDF?"),
                FileWithBase64(
                    type="file",
                    name="test.pdf",
                    mime="application/pdf",
                    base64=encoded,
                ),
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        assert "pineapple" in result.output_text.lower()

    @pytest.mark.asyncio
    async def test_multiple_files(self):
        """Test Vertex AI with multiple files in one query."""
        model = get_registry_model(
            "google/gemini-2.5-flash-lite",
            override_config=LLMConfig(provider_config=GoogleConfig(use_vertex=True)),
        )
        assert isinstance(model, GoogleModel)

        if not model.supports_files:
            pytest.skip("Model does not support files")

        image_bytes = create_test_image()
        pdf_bytes = create_test_pdf()

        image_encoded = base64.b64encode(image_bytes).decode("utf-8")
        pdf_encoded = base64.b64encode(pdf_bytes).decode("utf-8")

        result = await model.query(
            [
                TextInput(
                    text="What color is the image and what is the secret in the PDF? Answer both questions."
                ),
                FileWithBase64(
                    type="image",
                    name="test.png",
                    mime="png",
                    base64=image_encoded,
                ),
                FileWithBase64(
                    type="file",
                    name="test.pdf",
                    mime="application/pdf",
                    base64=pdf_encoded,
                ),
            ]
        )

        assert_basic_result(result)
        assert result.output_text is not None
        # Should mention both the color and the secret
        assert "red" in result.output_text.lower()
        assert "pineapple" in result.output_text.lower()
