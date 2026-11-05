"""Keep provider-side fetches inside a run token's sandbox egress policy.

Providers fetch URL-backed inputs and run server-side tools from their own
network, so a sandbox limited to the gateway could otherwise reach any host.
"""

import ipaddress
from typing import Any, Literal, cast
from urllib.parse import urlparse, urlsplit, urlunsplit

from pydantic import BaseModel

from model_gateway.errors import EgressPolicyError
from model_library.base import (
    ComputerCallOutput,
    FileWithId,
    FileWithUrl,
    InputItem,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolDefinition,
    ToolResult,
)

EgressPolicy = Literal["*"] | list[str]

EGRESS_NOTICE = (
    "Some request content or provider-side tools were removed because they reach "
    "destinations this run is not allowed to reach."
)

# Fields of a typed content part (OpenAI, Anthropic, Mistral, OpenRouter, …) that
# the provider fetches; values are a URL string or `{"url": ...}`.
_PART_URL_FIELDS = (
    "image_url",
    "file_url",
    "document_url",
    "video_url",
    "audio_url",
    "file_data",
)
# Google Files API handles are provider-side uploads, like other providers' file IDs.
_GOOGLE_FILES_PREFIX = "https://generativelanguage.googleapis.com/"


def _url(value: Any) -> str | None:
    if isinstance(value, dict):
        value = cast(dict[str, Any], value).get("url")
    return value if isinstance(value, str) and ":" in value else None


def _nested_dict(value: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        child = value.get(key)
        if not isinstance(child, dict):
            return None
        value = cast(dict[str, Any], child)
    return value


def _part_urls(part: dict[str, Any]) -> list[str]:
    """URLs the provider fetches for one content part; ordinary JSON yields none."""
    urls: list[str | None] = []
    if isinstance(part.get("type"), str):
        urls += [_url(part.get(key)) for key in _PART_URL_FIELDS]
        file = part.get("file")  # Chat Completions file part
        if isinstance(file, dict):
            file = cast(dict[str, Any], file)
            urls += [_url(file.get("file_data")), _url(file.get("file_url"))]
        audio = part.get("input_audio")  # Chat Completions audio part: base64 `data`
        if isinstance(audio, dict):
            urls.append(_url(cast(dict[str, Any], audio).get("data")))
        source = part.get("source")  # Anthropic image/document block
        if (
            isinstance(source, dict)
            and cast(dict[str, Any], source).get("type") == "url"
        ):
            urls.append(_url(source))
    for key in ("file_data", "fileData"):  # Gemini part
        data = part.get(key)
        if isinstance(data, dict):
            data = cast(dict[str, Any], data)
            uri = _url(data.get("file_uri", data.get("fileUri")))
            # Files API uploads are read inside Google, like `FileWithId`.
            if uri is not None and not uri.startswith(_GOOGLE_FILES_PREFIX):
                urls.append(uri)
    for key in ("image", "document", "video", "audio"):  # Bedrock Converse block
        location = _nested_dict(part, key, "source", "s3Location")
        if location is not None:
            urls.append(_url(location.get("uri")))
    return [url for url in urls if url is not None]


# Keys under which providers nest further content parts: message `content`,
# Gemini `parts`, Responses `function_call_output.output`, and tool results.
_NESTED_PART_LISTS = ("content", "parts", "output")
# Anthropic `source` holds a document's own `content` blocks when its type is "content".
_NESTED_PART_HOLDERS = ("source", "toolResult", "function_response", "functionResponse")

# Tool types the caller executes, so the provider opens no connection for them.
_CLIENT_TOOL_TYPES = frozenset(
    {
        "function",
        "custom",
        "namespace",
        "apply_patch",
        "computer",
        "computer_use_preview",
        "local_shell",
    }
)
_CLIENT_TOOL_PREFIXES = ("bash_", "text_editor_", "computer_", "memory_")


def _log_safe(url: str) -> str:
    try:
        parts = urlsplit(url)
        netloc = parts.hostname or ""
        if ":" in netloc:
            netloc = f"[{netloc}]"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
    except ValueError:
        return "<unparseable>"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_egress_policy(
    policy: list[str],
) -> tuple[set[str], list[str], list[ipaddress.IPv4Network]]:
    """Hosts, `*.` wildcard suffixes and IPv4 networks; raises ValueError like
    the sandbox's `resolve_allowed_addresses` on blank, malformed or IPv6 entries."""
    hosts: set[str] = set()
    wildcards: list[str] = []
    networks: list[ipaddress.IPv4Network] = []
    for raw in policy:
        entry = raw.strip()
        try:
            network: _Network | None = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            network = None
        if network is None:
            host = urlparse(entry if "://" in entry else f"//{entry}").hostname
            if not host:
                raise ValueError(f"egress entry is not a URL, host or CIDR: {raw!r}")
            try:
                network = ipaddress.ip_network(host)
            except ValueError:
                if host.startswith("*."):
                    wildcards.append(host[1:])
                else:
                    hosts.add(host)
                continue
        if not isinstance(network, ipaddress.IPv4Network):
            raise ValueError(
                f"egress entry is IPv6, which the sandbox rejects: {raw!r}"
            )
        networks.append(network)
    return hosts, wildcards, networks


def validate_egress_policy(policy: EgressPolicy) -> EgressPolicy:
    if policy != "*":
        parse_egress_policy(policy)
    return policy


class EgressFilter:
    """Strips one request's inputs and tools that would fetch outside the policy.

    Policy entries use the sandbox allowlist syntax (see `parse_egress_policy`).
    Scheme and port are ignored, as in the sandbox. `"*"` permits every URL.
    `tool_results_are_content` says whether the adapter sends a list `ToolResult`
    result as native content parts rather than as JSON data or text.
    `removed_inputs` counts request inputs that were removed or trimmed.
    """

    def __init__(self, policy: EgressPolicy, *, tool_results_are_content: bool = True):
        self._tool_results_are_content = tool_results_are_content
        self._open = policy == "*"
        self._hosts, self._wildcards, self._networks = parse_egress_policy(
            [] if policy == "*" else policy
        )
        self.blocked_urls: list[str] = []
        self.removed_inputs = 0
        self.removed_tools: list[str] = []

    @property
    def changed(self) -> bool:
        return bool(self.removed_inputs or self.removed_tools)

    def _allowed(self, url: str) -> bool:
        if self._open or url[:5].lower() == "data:":
            return True
        # Parsers disagree on these (e.g. WHATWG reads `\\` as `/`), so the provider
        # could fetch a different host than the one checked here.
        if any(
            c == "\\" or c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in url
        ):
            return False
        try:
            host = urlparse(url).hostname
        except ValueError:
            host = None
        if host is not None:
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                if host in self._hosts or any(
                    host.endswith(w) for w in self._wildcards
                ):
                    return True
            else:
                if any(address in network for network in self._networks):
                    return True
        return False

    def permits(self, url: str) -> bool:
        if self._allowed(url):
            return True
        self.blocked_urls.append(url)
        return False

    @property
    def logged_urls(self) -> list[str]:
        """Blocked URLs without credentials, query or fragment, for logs."""
        return [_log_safe(url) for url in dict.fromkeys(self.blocked_urls)]

    def filter_parts(self, parts: list[Any]) -> list[Any]:
        """Provider-format content parts without those that fetch outside the policy."""
        return self._clean_parts(parts)

    def _clean(self, value: Any) -> tuple[Any, bool]:
        """Drop fetching parts; report whether `value` itself must go.

        Unchanged values are returned as is.
        """
        if isinstance(value, BaseModel):
            # Restored provider objects: kept whole or dropped whole.
            dumped = value.model_dump(mode="json")
            cleaned, blocked = self._clean(dumped)
            return value, blocked or cleaned is not dumped
        if isinstance(value, list):
            items = cast(list[Any], value)
            cleaned = self._clean_parts(items)
            return cleaned, cleaned is not items and not cleaned
        if isinstance(value, dict):
            return self._clean_part(cast(dict[str, Any], value))
        return value, False

    def _clean_parts(self, parts: list[Any]) -> list[Any]:
        kept: list[Any] = []
        for part in parts:
            cleaned, blocked = self._clean(part)
            if not blocked:
                kept.append(cleaned)
        if len(kept) == len(parts) and all(a is b for a, b in zip(kept, parts)):
            return parts
        return kept

    def _clean_part(self, part: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Walks only where providers read content parts, so tool call arguments
        and tool response data are never mistaken for fetches."""
        if not all([self.permits(url) for url in _part_urls(part)]):
            return part, True
        updates: dict[str, Any] = {}
        for key in _NESTED_PART_LISTS:
            children = part.get(key)
            if isinstance(children, list) and children:
                children = cast(list[Any], children)
                cleaned = self._clean_parts(children)
                if cleaned is not children:
                    if not cleaned and "role" in part:
                        return part, True  # no empty message reaches the provider
                    updates[key] = cleaned
        for key in _NESTED_PART_HOLDERS:
            holder = part.get(key)
            if isinstance(holder, dict):
                holder = cast(dict[str, Any], holder)
                cleaned_holder, blocked = self._clean_part(holder)
                if blocked:
                    return part, True
                if cleaned_holder is not holder:
                    updates[key] = cleaned_holder
        return ({**part, **updates} if updates else part), False

    def _strip(self, item: InputItem) -> InputItem | None:
        match item:
            case FileWithUrl():
                return item if self.permits(item.url) else None
            # Google `Part.from_uri` reads URI file IDs such as `gs://…` remotely.
            case FileWithId() if ":" in item.file_id and not item.file_id.startswith(
                _GOOGLE_FILES_PREFIX
            ):
                return item if self.permits(item.file_id) else None
            case RawInput():
                cleaned, blocked = self._clean(item.input)
                if blocked:
                    return None
                return (
                    item
                    if cleaned is item.input
                    else item.model_copy(update={"input": cleaned})
                )
            case RawResponse():
                cleaned, blocked = self._clean(item.response)
                if blocked:
                    return None
                return (
                    item
                    if cleaned is item.response
                    else item.model_copy(update={"response": cleaned})
                )
            # Only list results can be content parts, and only for adapters that
            # send them as such; Gemini `FunctionResponse.response` is data.
            case ToolResult() if self._tool_results_are_content and isinstance(
                item.result, list
            ):
                result = cast(list[Any], item.result)
                cleaned = self._clean_parts(result)
                if cleaned is result:
                    return item
                return item.model_copy(update={"result": cleaned or EGRESS_NOTICE})
            # A computer reply needs its screenshot, so a blocked URL rejects the request.
            case ComputerCallOutput() if item.screenshot_url is not None:
                if not self.permits(item.screenshot_url):
                    raise EgressPolicyError(
                        "Computer screenshot URL is outside this run's egress policy; "
                        "send the screenshot as a data: URL"
                    )
                return item
            case _:
                return item

    def filter_inputs(self, inputs: list[InputItem]) -> list[InputItem]:
        kept: list[InputItem] = []
        notice_at: int | None = None
        for item in inputs:
            stripped = self._strip(item)
            if stripped is not None:
                kept.append(stripped)
            if stripped is not item:
                self.removed_inputs += 1
                if notice_at is None:
                    notice_at = len(kept)
        if notice_at is None and self.removed_tools:
            notice_at = 1 if kept and isinstance(kept[0], SystemInput) else 0
        if notice_at is None:
            return kept
        # Tool results must directly follow their calls.
        while notice_at < len(kept) and isinstance(kept[notice_at], ToolResult):
            notice_at += 1
        notice = self.notice()
        if not any(isinstance(i, TextInput) and i.text == notice for i in kept):
            kept.insert(notice_at, TextInput(text=notice))
        return kept

    def notice(self) -> str:
        if not self.blocked_urls:
            return EGRESS_NOTICE
        urls = ", ".join(dict.fromkeys(self.blocked_urls))
        return f"{EGRESS_NOTICE}\nBlocked URLs: {urls}"

    def client_tool(self, body: Any) -> bool:
        if isinstance(body, ToolBody):
            return True
        if not isinstance(body, dict):
            return False
        raw = cast(dict[str, Any], body)
        kind = raw.get("type")
        if kind is None:
            # Anthropic custom tools carry no type; Gemini server tools carry none either.
            return "name" in raw and ("input_schema" in raw or "parameters" in raw)
        if not isinstance(kind, str):
            return False
        if kind == "shell":
            environment = raw.get("environment")
            return isinstance(environment, dict) and (
                cast(dict[str, Any], environment).get("type") == "local"
            )
        return kind in _CLIENT_TOOL_TYPES or kind.startswith(_CLIENT_TOOL_PREFIXES)

    def filter_tools(self, tools: list[ToolDefinition]) -> list[ToolDefinition]:
        kept: list[ToolDefinition] = []
        for tool in tools:
            if self.client_tool(tool.body):
                kept.append(tool)
            else:
                self.removed_tools.append(tool.name)
        return kept
