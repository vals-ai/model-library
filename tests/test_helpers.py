"""
Shared test utilities and helpers for Google provider tests.
"""

import base64
from io import BytesIO
from typing import Any, Dict, cast

from PIL import Image

from model_library.base import InputItem, TextInput, ToolBody, ToolDefinition
from model_library.base.input import FileWithBase64


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


def create_test_image(color: str) -> bytes:
    """Create a simple test image (red square)."""

    img = Image.new("RGB", (100, 100), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_example_image_base64_input(image_color: str = "red") -> list[InputItem]:
    image_bytes = create_test_image(image_color)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return [
        TextInput(text="What color is this image?"),
        FileWithBase64(
            type="image",
            name="test.png",
            mime="png",
            base64=encoded,
        ),
    ]


def get_example_file_base64_input() -> list[InputItem]:
    pdf_bytes = create_test_pdf()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    return [
        TextInput(text="What is the secret mentioned in this PDF?"),
        FileWithBase64(
            type="file",
            name="test.pdf",
            mime="application/pdf",
            base64=encoded,
        ),
    ]


def get_example_tool_input() -> tuple[list[InputItem], str, list[ToolDefinition]]:
    text_input = "What is the weather in San Francisco right now?"
    system_prompt = "You are a weather expert which makes use of the tools provided."
    tools = [
        ToolDefinition(
            name="get_weather",
            body=ToolBody(
                name="get_weather",
                description="Get current temperature in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. San Francisco, USA",
                    },
                },
                required=["location"],
            ),
        ),
        ToolDefinition(
            name="get_danger",
            body=ToolBody(
                name="get_danger",
                description="Get current danger in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. San Francisco, USA",
                    },
                },
                required=["location"],
            ),
        ),
    ]
    return [TextInput(text=text_input)], system_prompt, tools


def assert_has_thinking_config(config: Dict[str, Any]) -> None:
    """Assert that config has thinking configuration enabled."""
    assert "thinking_config" in config
    tc = cast(Dict[str, Any], config["thinking_config"])  # normalized dict
    assert tc.get("include_thoughts") is True


def assert_no_thinking_config(config: Dict[str, Any]) -> None:
    """Assert that config has no thinking configuration."""
    assert "thinking_config" not in config


def assert_basic_result(result: Any) -> None:
    """Assert basic requirements for a query result."""
    assert result.output_text
    assert result.metadata.in_tokens > 0
    assert result.metadata.out_tokens > 0


def get_api_call_config(mock_client: Any) -> Dict[str, Any]:
    """Extract the config from the last API call, normalized to a dict.

    The Google SDK returns a pydantic model (GenerateContentConfig). Tests expect
    a dict-like object. This shim converts it to a thin dict with the keys
    we assert on in tests.
    """
    call_args = mock_client.return_value.aio.models.generate_content.call_args
    print(call_args)
    cfg = call_args[1]["config"]

    # Map core fields
    out: Dict[str, Any] = {}
    if hasattr(cfg, "temperature"):
        out["temperature"] = cfg.temperature
    if hasattr(cfg, "max_output_tokens"):
        out["max_output_tokens"] = cfg.max_output_tokens
    if hasattr(cfg, "top_p"):
        out["top_p"] = cfg.top_p
    # thinking_config is optional
    if hasattr(cfg, "thinking_config") and cfg.thinking_config is not None:
        tc = cfg.thinking_config
        out["thinking_config"] = {
            "thinking_budget": getattr(tc, "thinking_budget", None),
            "include_thoughts": getattr(tc, "include_thoughts", None),
        }

    if hasattr(cfg, "tool_config") and cfg.tool_config is not None:
        tcfg = cfg.tool_config
        fcfg = getattr(tcfg, "function_calling_config", None)
        mode_val = None
        if fcfg is not None:
            mode = getattr(fcfg, "mode", None)
            mode_val = getattr(mode, "name", None) or str(mode)
        out["tool_config"] = {
            "function_calling_config": {
                "mode": mode_val,
            }
        }
    return out
