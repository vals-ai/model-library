from typing import Sequence, TypeVar, cast

from model_library.base.input import (
    FileBase,
    InputItem,
    RawInputItem,
    TextInput,
    ToolResult,
)
from model_library.utils import truncate_str
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
            case dict():
                item = cast(RawInputItem, item)
                return repr(item)
            case _:
                # RawResponse
                return repr(item)

    processed_items = [f"  {process_item(item)}" for item in input]
    return "\n" + "\n".join(processed_items) if processed_items else ""
