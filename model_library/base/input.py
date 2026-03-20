import json
from pprint import pformat
from typing import Annotated, Any, Literal, cast

from pydantic import Field, computed_field
from typing_extensions import override

from model_library.utils import PrettyModel, truncate_str

"""
--- FILES ---
"""


class FileBase(PrettyModel):
    type: Literal["image", "file"]
    name: str
    mime: str

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        if "base64" in attrs:
            attrs["base64"] = truncate_str(attrs["base64"])
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2, sort_dicts=False)}\n)"


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


class ToolBody(PrettyModel):
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    kwargs: dict[str, Any] = {}


class ToolDefinition(PrettyModel):
    name: str  # acts as a key
    body: ToolBody | Any


class ToolCall(PrettyModel):
    id: str
    call_id: str | None = None
    name: str
    args: dict[str, Any] | str

    @computed_field
    @property
    def parsed_args(self) -> dict[str, Any] | None:
        """Parsed args as a dict, or None if args is an unparseable string"""
        if isinstance(self.args, dict):
            return self.args
        try:
            loaded = json.loads(self.args)
            if isinstance(loaded, dict):
                return cast(dict[str, Any], loaded)
            return None
        except (json.JSONDecodeError, TypeError):
            return None


"""
--- INPUT ---
"""


class ToolInput(PrettyModel):
    tools: list[ToolDefinition] = []


class ToolResult(PrettyModel):
    tool_call: ToolCall
    result: Any


class TextInput(PrettyModel):
    text: str


class RawResponse(PrettyModel):
    # used to store a received response
    response: Any


class RawInput(PrettyModel):
    # used to pass in anything provider specific (e.g. a mock conversation)
    input: Any


InputItem = (
    TextInput | FileInput | ToolResult | RawInput | RawResponse
)  # input item can either be a prompt, a file (image or file), a tool call result, a previous response, or raw input
