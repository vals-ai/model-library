from __future__ import annotations

import logging
from collections.abc import Sequence

from rich.pretty import pretty_repr

from model_library.base.input import InputItem, ToolDefinition, ToolResult
from model_library.base.output import QueryResult
from model_library.base.utils import get_pretty_input_types
from model_library.utils import MAX_LOG_HISTORY, truncate_str


def scoped_query_logger(
    base_logger: logging.Logger,
    *,
    question_id: str,
    query_id: str,
    in_agent: bool,
) -> logging.Logger:
    child_name = (
        f"<query={query_id}>"
        if in_agent
        else f"<question={question_id}><query={query_id}>"
    )
    query_logger = base_logger.getChild(child_name)
    if in_agent:
        query_logger.setLevel(logging.WARNING)
    return query_logger


def log_query_started(
    query_logger: logging.Logger,
    *,
    input: Sequence[InputItem],
    all_input: Sequence[InputItem],
    history: Sequence[InputItem],
    tools: list[ToolDefinition],
    kwargs: dict[str, object],
    info_enabled: bool | None = None,
    debug_enabled: bool | None = None,
) -> None:
    if info_enabled is None:
        info_enabled = query_logger.isEnabledFor(logging.INFO)
    if debug_enabled is None:
        debug_enabled = query_logger.isEnabledFor(logging.DEBUG)

    if not info_enabled:
        return

    item_info = (
        f"--- input ({len(input)}): {get_pretty_input_types(input, debug_enabled)}\n"
    )
    if history:
        logged_history = history if debug_enabled else history[-MAX_LOG_HISTORY:]
        item_info += f"--- history({len(history)}): {get_pretty_input_types(logged_history, debug_enabled)}\n"

    tool_results = [item for item in input if isinstance(item, ToolResult)]
    tool_names = [tool.name for tool in tools or []]
    tool_info = (
        f"--- tools ({len(tools)}): {tool_names}\n"
        + f"--- tool results ({len(tool_results)}): "
        + f"{[{tool.tool_call.name: truncate_str(str(tool.result))} for tool in tool_results]}\n"
        if tools
        else ""
    )
    short_kwargs = {key: truncate_str(repr(value)) for key, value in kwargs.items()}

    query_logger.info(
        "Query started:\n" + item_info + tool_info + f"--- kwargs: {short_kwargs}\n"
    )
    query_logger.debug([repr(item) for item in all_input])


def log_query_completed(
    query_logger: logging.Logger,
    output: QueryResult,
    *,
    info_enabled: bool | None = None,
    debug_enabled: bool | None = None,
) -> None:
    if info_enabled is None:
        info_enabled = query_logger.isEnabledFor(logging.INFO)
    if debug_enabled is None:
        debug_enabled = query_logger.isEnabledFor(logging.DEBUG)

    if info_enabled:
        max_string = None if debug_enabled else 400
        query_logger.info(
            f"Query completed: {pretty_repr(output, max_string=max_string)}"
        )
    if debug_enabled:
        query_logger.debug(repr(output))
