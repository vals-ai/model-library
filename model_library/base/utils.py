from typing import Sequence, cast

from model_library.base.input import (
    FileBase,
    InputItem,
    RawInputItem,
    TextInput,
    ToolResult,
)
from model_library.utils import truncate_str


def sum_optional(a: int | None, b: int | None) -> int | None:
    """Sum two optional integers, returning None if both are None.

    Preserves None to indicate "unknown/not provided" when both inputs are None,
    otherwise treats None as 0 for summation.
    """
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


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
