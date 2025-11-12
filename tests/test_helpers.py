"""
Shared test utilities and helpers for Google provider tests.
"""

from typing import Any, Callable, Dict, cast

from model_library.base import LLMConfig
from model_library.providers.google.google import GoogleModel


async def create_and_query_model(
    model_name: str = "gemini-2.5-flash",
    reasoning: bool = False,
    native: bool = True,
    query_text: str = "Test",
    create_test_input: Callable[[str], list[Any]] | None = None,
    use_vertex: bool = True,
    **query_kwargs: Any,
) -> GoogleModel:
    model = GoogleModel(
        model_name, config=LLMConfig(native=native, reasoning=reasoning)
    )
    print(model)

    if create_test_input:
        await model.query(create_test_input(query_text), **query_kwargs)

    return model


def assert_has_thinking_config(config: Dict[str, Any]) -> None:
    """Assert that config has thinking configuration enabled."""
    assert "thinking_config" in config
    tc = cast(Dict[str, Any], config["thinking_config"])  # normalized dict
    assert tc.get("include_thoughts") is True


def assert_no_thinking_config(config: Dict[str, Any]) -> None:
    """Assert that config has no thinking configuration."""
    assert "thinking_config" not in config


def assert_basic_result(result: Any) -> None:
    """Assert basic requirements for a query result."""
    assert result.output_text
    assert result.metadata.in_tokens > 0
    assert result.metadata.out_tokens > 0


def get_api_call_config(mock_client: Any) -> Dict[str, Any]:
    """Extract the config from the last API call, normalized to a dict.

    The Google SDK returns a pydantic model (GenerateContentConfig). Tests expect
    a dict-like object. This shim converts it to a thin dict with the keys
    we assert on in tests.
    """
    call_args = mock_client.return_value.aio.models.generate_content.call_args
    print(call_args)
    cfg = call_args[1]["config"]

    # Map core fields
    out: Dict[str, Any] = {}
    if hasattr(cfg, "temperature"):
        out["temperature"] = cfg.temperature
    if hasattr(cfg, "max_output_tokens"):
        out["max_output_tokens"] = cfg.max_output_tokens
    if hasattr(cfg, "top_p"):
        out["top_p"] = cfg.top_p
    # thinking_config is optional
    if hasattr(cfg, "thinking_config") and cfg.thinking_config is not None:
        tc = cfg.thinking_config
        out["thinking_config"] = {
            "thinking_budget": getattr(tc, "thinking_budget", None),
            "include_thoughts": getattr(tc, "include_thoughts", None),
        }

    if hasattr(cfg, "tool_config") and cfg.tool_config is not None:
        tcfg = cfg.tool_config
        fcfg = getattr(tcfg, "function_calling_config", None)
        mode_val = None
        if fcfg is not None:
            mode = getattr(fcfg, "mode", None)
            mode_val = getattr(mode, "name", None) or str(mode)
        out["tool_config"] = {
            "function_calling_config": {
                "mode": mode_val,
            }
        }
    return out
