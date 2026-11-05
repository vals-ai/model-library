from collections.abc import Sequence
from typing import Any, cast

from model_library.base.input import ComputerCallOutput, InputItem, ToolDefinition
from model_library.exceptions import BadInputError


def is_computer_tool(tool: ToolDefinition) -> bool:
    if not isinstance(tool.body, dict):
        return False
    body = cast(dict[str, Any], tool.body)
    tool_type = body.get("type")
    return isinstance(tool_type, str) and tool_type.startswith("computer")


def validate_computer_tools(tools: Sequence[ToolDefinition]) -> bool:
    computer_tools = [tool for tool in tools if is_computer_tool(tool)]
    if not computer_tools:
        return False
    if len(computer_tools) != 1:
        raise BadInputError("Computer use requires exactly one computer tool")
    return True


def validate_computer_provider(
    provider: str,
    has_computer_tool: bool,
    input: Sequence[InputItem],
) -> None:
    uses_computer = has_computer_tool or any(
        isinstance(item, ComputerCallOutput) for item in input
    )
    if uses_computer and provider not in {"anthropic", "openai"}:
        raise BadInputError(f"{provider} does not support native computer use")
