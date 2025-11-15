from pprint import pformat
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import override

from model_library.utils import truncate_str

"""
--- FILES ---
"""


class FileBase(BaseModel):
    type: Literal["image", "file"]
    name: str
    mime: str

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        if "base64" in attrs:
            attrs["base64"] = truncate_str(attrs["base64"])
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2)}\n)"


class FileWithBase64(FileBase):
    append_type: Literal["base64"] = "base64"
    base64: str


class FileWithUrl(FileBase):
    append_type: Literal["url"] = "url"
    url: str


class FileWithId(FileBase):
    append_type: Literal["file_id"] = "file_id"
    file_id: str


FileInput = Annotated[
    FileWithBase64 | FileWithUrl | FileWithId,
    Field(discriminator="append_type"),
]


"""
--- TOOLS ---
"""


class ToolBody(BaseModel):
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    kwargs: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    name: str  # acts as a key
    body: ToolBody | Any


class ToolCall(BaseModel):
    id: str
    call_id: str | None = None
    name: str
    args: dict[str, Any] | str


"""
--- INPUT ---
"""

RawResponse = Any


class ToolInput(BaseModel):
    tools: list[ToolDefinition] = []


class ToolResult(BaseModel):
    tool_call: ToolCall
    result: Any


class TextInput(BaseModel):
    text: str


RawInputItem = dict[
    str, Any
]  # to pass in, for example, a mock convertsation with {"role": "user", "content": "Hello"}


InputItem = (
    TextInput | FileInput | ToolResult | RawInputItem | RawResponse
)  # input item can either be a prompt, a file (image or file), a tool call result, raw input, or a previous response
