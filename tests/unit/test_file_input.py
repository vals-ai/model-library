"""File MIME normalization at construction and deserialization."""

from typing import Literal

import pytest
from pydantic import TypeAdapter

from model_library.base.input import (
    FileBase,
    FileWithBase64,
    FileWithBytes,
    FileWithId,
    FileWithUrl,
    InputItem,
)


@pytest.mark.parametrize(
    ("file_class", "payload"),
    [
        (FileWithBase64, {"base64": "dGVzdA=="}),
        (FileWithBytes, {"data": b"test"}),
        (FileWithId, {"file_id": "file-123"}),
        (FileWithUrl, {"url": "https://example.com/image.png"}),
    ],
)
@pytest.mark.parametrize(
    ("file_type", "mime", "expected"),
    [
        ("image", "png", "image/png"),
        ("image", "image/png", "image/png"),
        ("file", "application/pdf", "application/pdf"),
        ("file", "audio/wav", "audio/wav"),
        ("file", "png", "png"),
    ],
)
def test_file_mime_normalization(
    file_class: type[FileBase],
    payload: dict[str, str | bytes],
    file_type: Literal["image", "file"],
    mime: str,
    expected: str,
):
    """Normalize image MIME types across file variants and wire round trips."""
    file = file_class(type=file_type, name="test-file", mime=mime, **payload)
    assert file.mime == expected
    assert file.model_dump()["mime"] == expected

    adapter = TypeAdapter(InputItem)
    wire_input = {**file.model_dump(mode="json"), "mime": mime}
    assert adapter.validate_python(wire_input) == file
    assert adapter.validate_json(file.model_dump_json()) == file
