import json
import re
from datetime import datetime, timedelta
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from model_library.base.input import (
    FileBase,
    InputItem,
    RawInput,
    RawResponse,
    TextInput,
    ToolResult,
)
from model_library.utils import truncate_str

T = TypeVar("T", bound=BaseModel)


def serialize_for_tokenizing(content: Any) -> str:
    """
    Serialize parsed content into a string for tokenization
    """
    parts: list[str] = []
    if content:
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(json.dumps(content, default=str))
    return "\n".join(parts)


def add_optional(
    a: int | float | T | None, b: int | float | T | None
) -> int | float | T | None:
    """Add two optional objects, returning None if both are None.

    Preserves None to indicate "unknown/not provided" when both inputs are None,
    otherwise returns the non-None value or their sum.
    """
    if a is None and b is None:
        return None

    if a is None or b is None:
        return a or b

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b

    # NOTE: Ensure that the subtypes are the same so we can use the __add__ method just from one
    if type(a) is type(b):
        add_method = getattr(a, "__add__", None)
        if add_method is not None:
            return add_method(b)
    else:
        raise ValueError(
            f"Cannot add {type(a)} and {type(b)} because they are not the same subclass"
        )

    return None


def get_pretty_input_types(input: Sequence["InputItem"], verbose: bool = False) -> str:
    # for logging
    def process_item(item: "InputItem"):
        match item:
            case TextInput():
                item_str = repr(item)
                return item_str if verbose else truncate_str(item_str)
            case FileBase():  # FileInput
                return repr(item)
            case ToolResult():
                return repr(item)
            case RawInput():
                return repr(item)
            case RawResponse():
                return repr(item)

    processed_items = [f"  {process_item(item)}" for item in input]
    return "\n" + "\n".join(processed_items) if processed_items else ""


TIME_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)([a-zA-Z]+)$")
UNIT_TO_SECONDS = {
    "ms": 0.001,
    "s": 1,
    "m": 60,
    "h": 3600,
}


def to_timestamp(input_str: str, server_now: datetime) -> int:
    """Converts a header string into a server-relative Unix timestamp in ms."""
    input_str = input_str.strip()

    # ISO Timestamp (e.g. 2026-01-09T21:58:01Z)
    if "T" in input_str and "-" in input_str:
        try:
            dt = datetime.fromisoformat(input_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Duration (e.g. 10s, 6ms)
    match = TIME_PATTERN.match(input_str)
    if match:
        value, unit = match.groups()
        offset_seconds = float(value) * UNIT_TO_SECONDS.get(unit.lower(), 0)
        # Add duration to the SERVER'S provided date
        dt = server_now + timedelta(seconds=offset_seconds)
        return int(dt.timestamp() * 1000)

    raise ValueError(f"Unsupported time format: {input_str}")
