"""Hold raw provider requests to the run token's provider-side fetch policy.

The pass-through forwards provider-format bodies, so `EgressFilter` runs on the
raw JSON here instead of on `QueryRequest` inputs and tools.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from model_gateway.egress import EgressFilter
from model_gateway.run_tokens import RunTokenClaims

ClientTools = Callable[[Any, EgressFilter], tuple[Any | None, list[str]]]


def _typed_client_tool(tool: Any, egress: EgressFilter) -> tuple[Any | None, list[str]]:
    """OpenAI / Anthropic: one typed tool, kept whole or dropped."""
    if egress.client_tool(tool):
        return tool, []
    return None, [_tool_name(tool)]


def _function_client_tool(
    tool: Any, egress: EgressFilter
) -> tuple[Any | None, list[str]]:
    """Chat Completions through a router: only `function` tools run at the caller."""
    raw = cast(dict[str, Any], tool) if isinstance(tool, dict) else None
    if raw is not None and raw.get("type") == "function":
        return raw, []
    return None, [_tool_name(tool)]


# Gemini tool entries the caller runs; `googleSearch`, `urlContext`, ... run at Google.
_GEMINI_CLIENT_TOOLS = frozenset(
    {"functionDeclarations", "function_declarations", "computerUse", "computer_use"}
)


def _gemini_client_tools(
    tool: Any, egress: EgressFilter
) -> tuple[Any | None, list[str]]:
    """Gemini: one tool object holds several named tools; keep the caller's keys."""
    if not isinstance(tool, dict):
        return None, [_tool_name(tool)]
    raw = cast(dict[str, Any], tool)
    kept = {k: v for k, v in raw.items() if k in _GEMINI_CLIENT_TOOLS}
    return kept or None, [k for k in raw if k not in _GEMINI_CLIENT_TOOLS]


@dataclass(frozen=True)
class RequestFormat:
    """Where a provider's request format keeps the things the provider fetches or runs.

    `content_fields` hold content parts (URL inputs); `tools` entries pass through
    `client_tools`, which returns the part of a tool the caller runs (or `None`) and
    the names of what was dropped; `provider_side_fields` are whole request fields
    the provider acts on itself (MCP connectors, plugins, presets, fallbacks) and are
    removed like tools; `tool_items` are content item types that carry their own
    `tools`, filtered the same way; `tool_choice_fields` are dropped once they point
    at nothing.
    """

    content_fields: tuple[str, ...]
    client_tools: ClientTools = _typed_client_tool
    provider_side_fields: tuple[str, ...] = ()
    tool_items: tuple[str, ...] = ()
    tool_choice_fields: tuple[str, ...] = ("tool_choice",)


REQUEST_FORMATS: dict[str, RequestFormat] = {
    # A stored `prompt` brings whatever tools it was saved with at OpenAI, and
    # `additional_tools` input items add tools partway through the input.
    "openai": RequestFormat(
        content_fields=("input",),
        provider_side_fields=("prompt",),
        tool_items=("additional_tools",),
    ),
    "anthropic": RequestFormat(
        content_fields=("system", "messages"), provider_side_fields=("mcp_servers",)
    ),
    "google": RequestFormat(
        content_fields=("contents", "systemInstruction", "system_instruction"),
        client_tools=_gemini_client_tools,
        tool_choice_fields=("toolConfig", "tool_config"),
    ),
    # Chat Completions, where OpenRouter itself fetches through plugins (web
    # search, PDF parsing, routers), native web search, presets (a server-side
    # model + tools), `openrouter:*` server tools, and `models`/`route` fallbacks
    # (a served fallback would be billed as `model`).
    "openrouter": RequestFormat(
        content_fields=("messages",),
        client_tools=_function_client_tool,
        provider_side_fields=(
            "plugins",
            "web_search_options",
            "preset",
            "models",
            "route",
        ),
    ),
}


@dataclass
class EgressCheck:
    payload: dict[str, Any]
    blocked_urls: list[str] = field(default_factory=list[str])
    removed_tools: list[str] = field(default_factory=list[str])


def _tool_name(tool: Any) -> str:
    raw = cast(dict[str, Any], tool) if isinstance(tool, dict) else {}
    return str(raw.get("name") or raw.get("type") or "unknown")


def _client_tools(
    fmt: RequestFormat, tools: list[Any], egress: EgressFilter
) -> tuple[list[Any], list[str]]:
    kept: list[Any] = []
    removed: list[str] = []
    for tool in tools:
        client_part, dropped = fmt.client_tools(tool, egress)
        removed += dropped
        if client_part is not None:
            kept.append(client_part)
    return kept, removed


def _filter_tools(
    fmt: RequestFormat, payload: dict[str, Any], egress: EgressFilter
) -> list[str]:
    """Drops provider-side tools and fields from `payload` in place; returns their names."""
    removed: list[str] = []
    tools = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"], removed = _client_tools(fmt, cast(list[Any], tools), egress)
    item_tools_left = False
    for key in fmt.content_fields if fmt.tool_items else ():
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        kept_items: list[Any] = []
        for item in cast(list[Any], items):
            raw = cast(dict[str, Any], item) if isinstance(item, dict) else {}
            item_tools = raw.get("tools")
            if raw.get("type") in fmt.tool_items and isinstance(item_tools, list):
                kept, dropped = _client_tools(fmt, cast(list[Any], item_tools), egress)
                removed += dropped
                if not kept:
                    continue
                item = {**raw, "tools": kept}
                item_tools_left = True
            kept_items.append(item)
        payload[key] = kept_items
    for key in fmt.provider_side_fields:
        if key in payload:
            removed.append(key)
            payload.pop(key)
    if not removed:
        return []
    if not payload.get("tools"):
        payload.pop("tools", None)
    for key in fmt.tool_choice_fields:
        choice = payload.get(key)
        # Any choice once no tools are left, or one naming a removed tool,
        # fails at the provider.
        if ("tools" not in payload and not item_tools_left) or (
            isinstance(choice, dict)
            and {cast(dict[str, Any], choice).get(k) for k in ("name", "type")}
            & set(removed)
        ):
            payload.pop(key, None)
    return removed


def check_request(
    provider: str, payload: dict[str, Any], claims: RunTokenClaims
) -> EgressCheck:
    """Removes provider-side tools and finds URL inputs the token can't fetch.

    Mirrors `enforce_run_token_egress`: URL inputs are held to `egress` only with
    `allow_urls`, and tools are kept only with `allow_provider_tools` and an
    unrestricted `egress`.
    """
    check = EgressCheck(payload)
    if claims.egress == "*" and claims.allow_urls and claims.allow_provider_tools:
        return check
    fmt = REQUEST_FORMATS[provider]
    egress = EgressFilter(claims.egress if claims.allow_urls else [])
    for key in fmt.content_fields:
        value = payload.get(key)
        if isinstance(value, dict):
            value = [cast(dict[str, Any], value)]  # Gemini `systemInstruction`
        if isinstance(value, list):
            egress.filter_parts(cast(list[Any], value))
    check.blocked_urls = egress.logged_urls
    if claims.egress != "*" or not claims.allow_provider_tools:
        check.payload = dict(payload)
        check.removed_tools = _filter_tools(fmt, check.payload, egress)
    return check
