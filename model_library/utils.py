import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic.main import BaseModel

MAX_LLM_LOG_LENGTH = 100


def truncate_str(s: str | None, max_len: int = MAX_LLM_LOG_LENGTH) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    half = (max_len - 1) // 2
    return s[:half] + " […] " + s[-half:]


def get_logger(name: str | None = None):
    if not name:
        caller = inspect.stack()[1]
        module = inspect.getmodule(caller[0])
        name = module.__name__ if module else "__main__"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def deep_model_dump(obj: object) -> object:
    if isinstance(obj, BaseModel):
        return deep_model_dump(obj.model_dump(exclude_unset=True, exclude_none=True))

    if isinstance(obj, Mapping):
        return {k: deep_model_dump(v) for k, v in obj.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [deep_model_dump(v) for v in obj]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    return obj


def default_httpx_client():
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=30 * 60, connect=60),
        limits=httpx.Limits(
            max_connections=1000, max_keepalive_connections=100
        ),  # TODO: increase, but make sure prod enough sockets to not hit file descriptor limit
    )


def create_openai_client_with_defaults(
    api_key: str, base_url: str | None = None
) -> AsyncOpenAI:
    """
    OpenAI defaults:
    DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=default_httpx_client(),
        max_retries=3,
    )


def sum_optional(a: int | None, b: int | None) -> int | None:
    """Sum two optional integers, returning None if both are None.

    Preserves None to indicate "unknown/not provided" when both inputs are None,
    otherwise treats None as 0 for summation.
    """
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def get_context_window_for_model(model_name: str, default: int = 128_000) -> int:
    """
    Get the context window for a model by looking up its configuration from the registry.

    Args:
        model_name: The name of the model in the registry (e.g., "openai/gpt-4o-mini-2024-07-18" or "azure/gpt-4o-mini-2024-07-18")
        default: Default context window to return if model not found or missing context_window

    Returns:
        Context window size in tokens
    """
    # import here to avoid circular imports
    from model_library.register_models import get_model_registry
    from model_library.utils import get_logger

    logger = get_logger(__name__)
    model_config = get_model_registry().get(model_name, None)
    if (
        model_config
        and model_config.properties
        and model_config.properties.context_window
    ):
        return model_config.properties.context_window
    else:
        logger.warning(
            f"Model {model_name} not found in registry or missing context_window, "
            f"using default context length of {default}"
        )
        return default


def normalize_tool_result(result: Any) -> str:
    """Normalize tool result to non-empty string for API compatibility.

    Empty results (None, empty dict/list, whitespace-only strings) are
    converted to a single space to satisfy API requirements.

    Args:
        result: Tool result value (any type)

    Returns:
        Non-empty string representation of the result
    """
    if result is None or (isinstance(result, (dict, list)) and not result):
        return " "
    result_str = str(result)  # pyright: ignore[reportUnknownArgumentType]
    return result_str.strip() or " "


def filter_empty_text_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out empty text blocks from content list.

    Args:
        content: List of content blocks (dicts with 'type' and potentially 'text' keys)

    Returns:
        Filtered list with empty text blocks removed
    """
    return [
        block
        for block in content
        if block.get("type") != "text" or block.get("text", "").strip()
    ]
