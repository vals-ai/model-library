import base64
import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

from typing_extensions import TypeAlias, override

from pydantic import Field, computed_field, field_serializer, field_validator

from model_library.utils import ValsModel, content_length, content_preview

"""
--- FILES ---
"""


def _sanitize_content_field(
    field: str,
    value: object,
    *,
    show_content: bool,
) -> dict[str, object]:
    if show_content:
        return {field: content_preview(value)}
    return {f"{field}_length": content_length(value)}


class FileBase(ValsModel):
    kind: Literal["file_base"] = "file_base"
    type: Literal["image", "file"]
    name: str
    mime: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            "type": self.type,
            "name": self.name,
            "mime": self.mime,
        }


class FileWithBase64(FileBase):
    append_type: Literal["base64"] = "base64"
    base64: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            **super().sanitize_content(show_content=show_content),
            "append_type": self.append_type,
            "base64_length": content_length(self.base64),
        }


class FileWithBytes(FileBase):
    append_type: Literal["bytes"] = "bytes"
    data: bytes

    @field_validator("data", mode="before")
    @classmethod
    def deserialize_data(cls, value: object) -> object:
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("data")
    def serialize_data(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            **super().sanitize_content(show_content=show_content),
            "append_type": self.append_type,
            "data_length": len(self.data),
        }


class FileWithUrl(FileBase):
    append_type: Literal["url"] = "url"
    url: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            **super().sanitize_content(show_content=show_content),
            "append_type": self.append_type,
            **_sanitize_content_field("url", self.url, show_content=show_content),
        }


class FileWithId(FileBase):
    append_type: Literal["file_id"] = "file_id"
    file_id: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            **super().sanitize_content(show_content=show_content),
            "append_type": self.append_type,
            "file_id": self.file_id,
        }


FileInput = Annotated[
    FileWithBase64 | FileWithBytes | FileWithUrl | FileWithId,
    Field(discriminator="append_type"),
]


"""
--- TOOLS ---
"""


AllowedToolCaller: TypeAlias = Literal["direct", "code_mode"]


class ToolBody(ValsModel):
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    allowed_callers: list[AllowedToolCaller] | None = None
    kwargs: dict[str, Any] = {}


class ToolDefinition(ValsModel):
    name: str  # acts as a key
    body: ToolBody | Any

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        summary: dict[str, object] = {"name": self.name}
        if show_content:
            summary["body"] = content_preview(self.body)
        else:
            summary["body_length"] = content_length(self.body)
        return summary


class ToolCall(ValsModel):
    id: str
    call_id: str | None = None
    name: str
    args: dict[str, Any] | str
    code_mode_id: str | None = None
    sequence: int | None = None

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        data = cast(
            dict[str, object],
            self.model_dump(mode="json", exclude={"args", "parsed_args"}),
        )
        data.update(
            _sanitize_content_field("args", self.args, show_content=show_content)
        )
        return data

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


class ToolInput(ValsModel):
    tools: list[ToolDefinition] = []

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "tools": [
                tool.sanitize_content(show_content=show_content) for tool in self.tools
            ]
        }


class ToolResult(ValsModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_call: ToolCall
    result: Any

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            "tool_call": self.tool_call.sanitize_content(show_content=show_content),
            **_sanitize_content_field("result", self.result, show_content=show_content),
        }


class SystemInput(ValsModel):
    """System prompt input item. Must be first in the input sequence if present. At most one allowed."""

    kind: Literal["system"] = "system"
    text: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            **_sanitize_content_field("text", self.text, show_content=show_content),
        }


class TextInput(ValsModel):
    kind: Literal["text"] = "text"
    text: str

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            **_sanitize_content_field("text", self.text, show_content=show_content),
        }


class RawResponse(ValsModel):
    kind: Literal["raw_response"] = "raw_response"
    # used to store a received response
    response: Any

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            **_sanitize_content_field("response", self.response, show_content=False),
        }


class RawInput(ValsModel):
    kind: Literal["raw_input"] = "raw_input"
    # used to pass in anything provider specific (e.g. a mock conversation)
    input: Any

    @override
    def sanitize_content(self, *, show_content: bool = False) -> dict[str, object]:
        return {
            "kind": self.kind,
            **_sanitize_content_field("input", self.input, show_content=False),
        }


InputItem = Annotated[
    SystemInput | TextInput | FileInput | ToolResult | RawInput | RawResponse,
    Field(discriminator="kind"),
]


def validate_query_input(input: Sequence[InputItem]) -> None:
    """Validate query-level SystemInput placement."""
    system_inputs = [i for i, item in enumerate(input) if isinstance(item, SystemInput)]
    if len(system_inputs) > 1:
        raise ValueError("At most one SystemInput is allowed per query")
    if system_inputs and system_inputs[0] != 0:
        raise ValueError("SystemInput must be the first item in the input sequence")


def normalize_query_input(
    input: Sequence[InputItem] | str,
    *,
    history: Sequence[InputItem] = (),
    kwargs: dict[str, object] | None = None,
) -> list[InputItem]:
    """Normalize public query input and validate SystemInput placement."""
    if isinstance(input, str):
        input_items: list[InputItem] = [TextInput(text=input)]
    else:
        input_items = list(input)

    # Back-compat: system_prompt kwarg is deprecated; prefer a leading SystemInput.
    system_prompt = kwargs.pop("system_prompt", None) if kwargs is not None else None
    if system_prompt is not None:
        input_items = [SystemInput(text=str(system_prompt)), *input_items]

    all_input = [*list(history), *input_items]
    validate_query_input(all_input)
    return all_input
