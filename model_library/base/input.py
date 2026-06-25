import json
from collections.abc import Generator, Sequence
from typing import Annotated, Any, Literal, cast

from typing_extensions import TypeAlias

from pydantic import Field, computed_field

from model_library.utils import ValsModel

"""
--- FILES ---
"""


class FileBase(ValsModel):
    kind: Literal["file_base"] = "file_base"
    type: Literal["image", "file"]
    name: str
    mime: str


class FileWithBase64(FileBase):
    append_type: Literal["base64"] = "base64"
    base64: str

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        # Elide the raw base64 payload — it can be megabytes and pollutes
        # any rendered transcript (e.g. compaction prompts, debug logs).
        yield "type", self.type
        yield "name", self.name
        yield "mime", self.mime
        yield "append_type", self.append_type
        yield "base64", f"<{len(self.base64)} chars>"


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


class ToolCall(ValsModel):
    id: str
    call_id: str | None = None
    name: str
    args: dict[str, Any] | str
    code_mode_id: str | None = None
    sequence: int | None = None

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


class ToolResult(ValsModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_call: ToolCall
    result: Any


class SystemInput(ValsModel):
    """System prompt input item. Must be first in the input sequence if present. At most one allowed."""

    kind: Literal["system"] = "system"
    text: str


class TextInput(ValsModel):
    kind: Literal["text"] = "text"
    text: str


class RawResponse(ValsModel):
    kind: Literal["raw_response"] = "raw_response"
    # used to store a received response
    response: Any


class RawInput(ValsModel):
    kind: Literal["raw_input"] = "raw_input"
    # used to pass in anything provider specific (e.g. a mock conversation)
    input: Any


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
