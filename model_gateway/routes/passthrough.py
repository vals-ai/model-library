"""Provider-format pass-through routes."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import httpx
from anthropic.types import MessageDeltaUsage
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from google.genai.types import GenerateContentResponseUsageMetadata
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import ResponseUsage
from pydantic import BaseModel, ValidationError
import sentry_sdk
from starlette.requests import ClientDisconnect

from model_library import model_library_settings
import model_library.telemetry as telemetry
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.base.query_ids import resolve_query_ids
from model_library.register_models import RegistryEntry
from model_library.registry_utils import (
    compute_model_cost,
    get_model_names,
    get_registry_config,
)

from model_gateway.errors import AssignedKeyUnavailableError
from model_gateway.model_helpers import (
    assigned_key_setting,
    current_pool_revision,
    managed_api_key_value,
)
from model_gateway.observability import log_failure, log_gateway_event
from model_gateway.passthrough_egress import check_request
from model_gateway.passthrough_exchanges import (
    ExchangeLog,
    ExchangeStore,
    exchange_prefix,
)
from model_gateway.run_tokens import RunTokenClaims
from model_gateway.request_budget import budget_refusal, usage_event_builder
from model_gateway.types import ProviderError, QueryRequest

# Logs under uvicorn.error like the gateway's other events; a module logger has no handler.
logger = logging.getLogger("uvicorn.error.gateway_passthrough")

UsageExtractor = Callable[[dict[str, Any]], dict[str, int]]
CostExtractor = Callable[[dict[str, Any]], float | None]
ErrorBody = Callable[[int, str], dict[str, Any]]
UsageModel = TypeVar("UsageModel", bound=BaseModel)


@dataclass(frozen=True)
class Upstream:
    base_url: str
    key_header: str
    key_format: str
    key_setting: str
    usage: UsageExtractor
    error_body: ErrorBody
    # Separator of routing variants in model ids (OpenRouter `model:online`); the
    # allowlist names base models, so an id carrying one is refused.
    variant_separator: str | None = None
    # The USD cost the provider reports on a response; prices calls to models that
    # a `provider/x/*` allowlist pattern admits without a registry entry.
    reported_cost: CostExtractor | None = None


class _Reject(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class _Call:
    provider: str
    upstream: Upstream
    upstream_path: str
    registry_key: str
    query: QueryRequest
    query_params: dict[str, object | None]
    provider_key_name: str | None
    pool_revision: int | None
    client_bytes: bytes
    request_bytes: bytes
    removed_tools: tuple[str, ...]
    stream: bool


def _validated_usage(
    provider: str, model: type[UsageModel], payload: Any
) -> UsageModel | None:
    try:
        return model.model_validate(payload, extra="ignore")
    except ValidationError as error:
        logger.warning("Pass-through %s usage did not parse: %s", provider, error)
        return None


def _token_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int | None,
    cache_write_tokens: int | None,
    reasoning_tokens: int | None,
) -> dict[str, int]:
    """Ledger fields from OpenAI-style totals, which include cached and reasoning tokens."""
    cache_read_tokens = cache_read_tokens or 0
    cache_write_tokens = cache_write_tokens or 0
    return {
        "in_tokens": input_tokens - cache_read_tokens - cache_write_tokens,
        "out_tokens": output_tokens - (reasoning_tokens or 0),
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        **(
            {"reasoning_tokens": reasoning_tokens}
            if reasoning_tokens is not None
            else {}
        ),
    }


def _openai_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage_data = payload.get("usage")
    if usage_data is None and isinstance(payload.get("response"), dict):
        usage_data = payload["response"].get("usage")
    if not isinstance(usage_data, dict):
        return {}
    usage = _validated_usage("openai", ResponseUsage, usage_data)
    if usage is None:
        return {}
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    return _token_usage(
        usage.input_tokens,
        usage.output_tokens,
        input_details.cached_tokens if input_details else None,
        input_details.cache_write_tokens if input_details else None,
        output_details.reasoning_tokens if output_details else None,
    )


def _anthropic_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage_data = payload.get("usage")
    if usage_data is None and isinstance(payload.get("message"), dict):
        usage_data = payload["message"].get("usage")
    if not isinstance(usage_data, dict):
        return {}
    usage = _validated_usage("anthropic", MessageDeltaUsage, usage_data)
    if usage is None:
        return {}
    return {
        name: value
        for name, value in {
            "in_tokens": usage.input_tokens,
            "out_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cache_write_tokens": usage.cache_creation_input_tokens,
        }.items()
        if value is not None
    }


def _google_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage_data = payload.get("usageMetadata")
    if not isinstance(usage_data, dict):
        return {}
    usage = _validated_usage("google", GenerateContentResponseUsageMetadata, usage_data)
    if usage is None:
        return {}
    cached = usage.cached_content_token_count or 0
    return {
        name: value
        for name, value in {
            "in_tokens": (
                usage.prompt_token_count - cached
                if usage.prompt_token_count is not None
                else None
            ),
            "out_tokens": usage.candidates_token_count,
            "reasoning_tokens": usage.thoughts_token_count,
            "cache_read_tokens": usage.cached_content_token_count,
        }.items()
        if value is not None
    }


def _openrouter_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage_data = payload.get("usage")
    if not isinstance(usage_data, dict):
        return {}
    usage = _validated_usage("openrouter", CompletionUsage, usage_data)
    if usage is None:
        return {}
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return _token_usage(
        usage.prompt_tokens,
        usage.completion_tokens,
        prompt_details.cached_tokens if prompt_details else None,
        prompt_details.cache_write_tokens if prompt_details else None,
        completion_details.reasoning_tokens if completion_details else None,
    )


def _openrouter_cost(payload: dict[str, Any]) -> float | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = cast(dict[str, Any], usage).get("cost")
    return float(cost) if isinstance(cost, (int, float)) else None


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload = dict(pairs)
    if len(payload) != len(pairs):
        # Parsers disagree on which copy wins, so the provider could read a
        # different model or tools than the ones checked here.
        raise ValueError("duplicate key")
    return payload


def _json_object(raw: bytes | bytearray) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeDecodeError):
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


_PROVIDER_ERROR_TYPES = {
    400: "invalid_request_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    502: "api_error",
    503: "overloaded_error",
}
_GOOGLE_ERROR_STATUSES = {
    400: "INVALID_ARGUMENT",
    402: "RESOURCE_EXHAUSTED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    409: "FAILED_PRECONDITION",
    502: "UNAVAILABLE",
    503: "UNAVAILABLE",
}


def _openai_error(status: int, message: str) -> dict[str, Any]:
    error_type = _PROVIDER_ERROR_TYPES.get(status, "api_error")
    return {"error": {"message": message, "type": error_type, "code": error_type}}


def _anthropic_error(status: int, message: str) -> dict[str, Any]:
    error_type = _PROVIDER_ERROR_TYPES.get(status, "api_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _google_error(status: int, message: str) -> dict[str, Any]:
    status_name = _GOOGLE_ERROR_STATUSES.get(status, "INTERNAL")
    return {"error": {"code": status, "message": message, "status": status_name}}


UPSTREAMS: dict[str, Upstream] = {
    "openai": Upstream(
        base_url="https://api.openai.com",
        key_header="authorization",
        key_format="Bearer {key}",
        key_setting="OPENAI_API_KEY",
        usage=_openai_usage,
        error_body=_openai_error,
    ),
    "anthropic": Upstream(
        base_url="https://api.anthropic.com",
        key_header="x-api-key",
        key_format="{key}",
        key_setting="ANTHROPIC_API_KEY",
        usage=_anthropic_usage,
        error_body=_anthropic_error,
    ),
    "google": Upstream(
        base_url="https://generativelanguage.googleapis.com",
        key_header="x-goog-api-key",
        key_format="{key}",
        key_setting="GOOGLE_API_KEY",
        usage=_google_usage,
        error_body=_google_error,
    ),
    "openrouter": Upstream(
        base_url="https://openrouter.ai",
        key_header="authorization",
        key_format="Bearer {key}",
        key_setting="OPENROUTER_API_KEY",
        usage=_openrouter_usage,
        error_body=_openai_error,
        variant_separator=":",
        reported_cost=_openrouter_cost,
    ),
}

_FORWARDED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "anthropic-beta",
        "anthropic-version",
        "content-type",
        "openai-beta",
    }
)
_UNSAVED_HEADER_PARTS = (
    "auth",
    "key",
    "token",
    "secret",
    "cookie",
    "session",
    "signature",
    "credential",
    "passw",
)
_STRIPPED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "set-cookie",
        "keep-alive",
        "openai-organization",
        "openai-project",
        "anthropic-organization-id",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _usage_payloads(body: bytes, event_stream: bool) -> Iterator[dict[str, Any]]:
    if event_stream:
        for line in body.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                yield payload
        return

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(payload, dict):
        yield cast(dict[str, Any], payload)
    elif isinstance(payload, list):
        for item in cast(list[Any], payload):
            if isinstance(item, dict):
                yield cast(dict[str, Any], item)


def _usage_fields(
    upstream: Upstream, body: bytes, event_stream: bool
) -> tuple[dict[str, int], float | None]:
    """Ledger token fields and the provider-reported USD cost, when it sends one."""
    fields: dict[str, int] = {}
    cost = None
    for payload in _usage_payloads(body, event_stream):
        fields.update(upstream.usage(payload))
        if upstream.reported_cost is not None:
            cost = upstream.reported_cost(payload) or cost
    return fields, cost


def _usage_result(
    registry_key: str, fields: dict[str, int], reported_cost: float | None
) -> QueryResult:
    """Draft #1545 response hashes use these fields in build_usage_event_details.

    A registry model is priced from the registry; a model admitted by a `provider/x/*`
    pattern has no entry, so the provider's reported cost is the price.
    """
    result = QueryResult(metadata=QueryResultMetadata(**cast(dict[str, Any], fields)))
    if get_registry_config(registry_key) is not None:
        result.metadata.cost = compute_model_cost(registry_key, result.metadata)
    elif reported_cost is not None:
        result.metadata.cost = QueryResultCost(
            input=0, output=0, total_override=reported_cost
        )
    return result


def _allowed_entries(
    provider: str, claims: RunTokenClaims
) -> Iterator[tuple[str, str, RegistryEntry]]:
    """`(registry key, provider model id, config)` for each entry the token may call."""
    allowed = claims.allowed_models
    candidates = get_model_names(provider) if allowed is None else allowed
    separator = UPSTREAMS[provider].variant_separator
    for key in candidates:
        if not key.startswith(f"{provider}/"):
            continue
        config = get_registry_config(key)
        if config is None:
            continue
        if config.metadata.router:
            # Router entries (`openrouter/auto`, ...) are constrained by their
            # `provider_properties` in `/query`; pass-through forwards the client's
            # bytes and the entry is `ignored_for_cost`, so the served model would
            # go unconstrained and unbilled.
            continue
        if config.metadata.approved_identities_only:
            # Pass-through sends a pooled static key; these models authenticate only
            # through `/query` (workload identity for Anthropic `-cyber` ids).
            continue
        model = config.provider_endpoint.removeprefix("models/")
        if separator is not None and separator in model:
            continue
        yield key, model, config


def _allowed_prefixes(provider: str, claims: RunTokenClaims) -> list[str]:
    """`provider/x/*` allowlist entries admit every provider id under `x/`."""
    return [
        key[:-1]
        for key in claims.allowed_models or ()
        if key.startswith(f"{provider}/") and key.endswith("/*")
    ]


def _registry_key(provider: str, model: str, claims: RunTokenClaims) -> str | None:
    """The ledger key for `model`: its registry key, or `provider/model` under a pattern."""
    for key, endpoint, _ in _allowed_entries(provider, claims):
        if key == f"{provider}/{model}" or endpoint == model:
            return key
    key = f"{provider}/{model}"
    if any(key.startswith(prefix) for prefix in _allowed_prefixes(provider, claims)):
        config = get_registry_config(key)
        if config is None or not (
            config.metadata.router or config.metadata.approved_identities_only
        ):
            return key
    return None


def _error_message(body: bytes, event_stream: bool, status: int) -> str:
    for payload in _usage_payloads(body, event_stream):
        error = payload.get("error")
        message = (
            cast(dict[str, Any], error).get("message")
            if isinstance(error, dict)
            else None
        )
        if isinstance(message, str) and message:
            return message
    return f"Upstream returned HTTP {status}"


def _stream_error(body: bytes) -> str | None:
    """The message of an error event inside a stream that started with HTTP 200."""
    for payload in _usage_payloads(body, event_stream=True):
        event_type = payload.get("type")
        if event_type == "response.failed":
            response = payload.get("response")
            payload = (
                cast(dict[str, Any], response) if isinstance(response, dict) else {}
            )
        elif event_type != "error" and "error" not in payload:
            continue
        error = payload.get("error")
        message = (
            cast(dict[str, Any], error).get("message")
            if isinstance(error, dict)
            else payload.get("message")
        )
        return message if isinstance(message, str) and message else "Stream failed"
    return None


async def _authorize(request: Request, provider: str, model: str) -> str:
    claims = cast(RunTokenClaims, request.state.run_token_claims)
    separator = UPSTREAMS[provider].variant_separator
    if separator is not None and separator in model:
        # `:online` makes OpenRouter fetch web results for the sandbox; `:nitro`,
        # `:free`, ... pick endpoints the allowlisted entry isn't priced for.
        raise _Reject(403, "Model variant suffixes (such as ':online') aren't allowed")
    registry_key = _registry_key(provider, model, claims)
    if registry_key is None:
        raise _Reject(403, "Model is not authorized")
    refusal = await budget_refusal(request)
    if refusal is not None:
        raise _Reject(refusal.status, refusal.message)
    return registry_key


def _checked_body(
    provider: str,
    payload: dict[str, Any],
    client_bytes: bytes,
    claims: RunTokenClaims,
    registry_key: str,
) -> tuple[bytes, list[str]]:
    """The body to forward, and the provider-side tools removed from it."""
    check = check_request(provider, payload, claims)
    if check.blocked_urls or check.removed_tools:
        log_gateway_event(
            "gateway.run_token.egress_blocked",
            route="passthrough",
            run_id=claims.run_id,
            task_id=claims.task_id,
            model=registry_key,
            removed_tools=",".join(check.removed_tools) or None,
            blocked_urls=",".join(check.blocked_urls) or None,
        )
    if check.blocked_urls:
        raise _Reject(
            400,
            "This run doesn't allow the provider to fetch URL inputs: "
            + ", ".join(check.blocked_urls),
        )
    if not check.removed_tools:
        return client_bytes, []
    return json.dumps(check.payload, ensure_ascii=False).encode(), check.removed_tools


async def _provider_key(
    request: Request, body: QueryRequest, upstream: Upstream
) -> tuple[str, str | None, int | None]:
    provider_key_name = None
    pool_revision = None
    key_assignments = request.app.state.key_assignments
    if key_assignments is not None:
        try:
            provider_key_name = await assigned_key_setting(body, key_assignments)
        except AssignedKeyUnavailableError as error:
            raise _Reject(409, str(error)) from error
        if provider_key_name is not None:
            pool_revision = current_pool_revision()
    key = managed_api_key_value(body, key_setting=provider_key_name)
    if key is None and provider_key_name is None:
        key = model_library_settings.get(upstream.key_setting)
    if not isinstance(key, str) or not key:
        raise _Reject(502, "Provider API key is unavailable")
    return key, provider_key_name, pool_revision


def _upstream_request(
    request: Request,
    call: _Call,
    key: str,
) -> httpx.Request:
    headers = {
        name: value
        for name in _FORWARDED_REQUEST_HEADERS
        if (value := request.headers.get(name)) is not None
    }
    headers["accept-encoding"] = "identity"
    headers[call.upstream.key_header] = call.upstream.key_format.format(key=key)
    params = [
        (name, value)
        for name, value in request.query_params.multi_items()
        if name.lower() != "key"
    ]
    return request.app.state.passthrough_client.build_request(
        "POST",
        f"{call.upstream.base_url}{call.upstream_path}",
        content=call.request_bytes,
        headers=headers,
        params=params,
    )


def _saved_headers(
    headers: Mapping[str, str], unsaved_parts: tuple[str, ...]
) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in headers.items()
        if not any(part in name.lower() for part in unsaved_parts)
    ]


async def _record_usage(
    request: Request,
    call: _Call,
    status_code: int,
    body: bytes,
    event_stream: bool,
    exchange: dict[str, str] | None = None,
    failure: str | None = None,
) -> None:
    """Records one call, failed or not, so the run's ledger has every call it made.

    `failure` is why the call broke after (or instead of) the upstream status, such as a
    connection error or a stream that stopped partway.
    """
    ledger = request.app.state.usage_ledger
    if not ledger.enabled:
        return
    with log_failure(logger, "Pass-through usage ledger write failed"):
        if failure is None and status_code >= 400:
            failure = _error_message(body, event_stream, status_code)
        if failure is None and event_stream:
            failure = _stream_error(body)
        error = (
            ProviderError(
                message=failure,
                provider=call.provider,
                code=str(status_code),
                status_code=status_code,
            )
            if failure is not None
            else None
        )
        fields, reported_cost = _usage_fields(call.upstream, body, event_stream)
        result = _usage_result(call.registry_key, fields, reported_cost)
        event = usage_event_builder(request)(
            body=call.query,
            config={},
            query_params=call.query_params,
            dimensions={},
            result=result,
            error=error,
            request={
                "passthrough": {
                    "provider": call.provider,
                    "path": call.upstream_path,
                    "stream": call.stream,
                    "status_code": status_code,
                    "request_bytes": len(call.client_bytes),
                    **(
                        {"removed_tools": list(call.removed_tools)}
                        if call.removed_tools
                        else {}
                    ),
                    **({"exchange": exchange} if exchange is not None else {}),
                }
            },
            api_key_name=request.state.gateway_api_key_name,
            provider_key_name=call.provider_key_name,
            pool_revision=call.pool_revision,
        )
        await ledger.write_event(event)


def _log_body(request: Request, call: _Call, body: bytes, **fields: object) -> None:
    """Queues one side of the exchange for the CloudWatch log, when it is configured."""
    exchange_log: ExchangeLog | None = request.app.state.passthrough_exchange_log
    if exchange_log is None:
        return
    query = call.query
    exchange_log.send(
        {
            "run_id": query.run_id,
            "attempt_id": query.attempt_id,
            "question_id": query.question_id,
            "query_id": query.query_id,
            "provider": call.provider,
            "model": call.registry_key,
            "path": call.upstream_path,
            "stream": call.stream,
            "exchange_prefix": exchange_prefix(
                query.run_id, query.attempt_id, query.query_id
            ),
            **fields,
        },
        body,
    )


async def _proxy(
    request: Request,
    *,
    provider: str,
    model: str,
    upstream_path: str,
    payload: dict[str, Any],
    client_bytes: bytes,
    stream: bool,
) -> Response:
    upstream = UPSTREAMS[provider]
    try:
        registry_key = await _authorize(request, provider, model)
        claims = cast(RunTokenClaims, request.state.run_token_claims)
        request_bytes, removed_tools = _checked_body(
            provider, payload, client_bytes, claims, registry_key
        )
        query = QueryRequest(
            model=registry_key,
            inputs=[],
            run_id=claims.run_id,
            question_id=claims.task_id,
            attempt_id=claims.attempt_id or None,
            identity=claims.identity,
        )
        ids = resolve_query_ids(
            run_id=query.run_id,
            question_id=query.question_id,
            query_id=query.query_id,
            attempt_id=query.attempt_id,
        )
        query = query.model_copy(update=ids.as_attributes())
        query_params = {**ids.as_attributes(), "identity": query.identity}
        key, provider_key_name, pool_revision = await _provider_key(
            request, query, upstream
        )
        call = _Call(
            provider=provider,
            upstream=upstream,
            upstream_path=upstream_path,
            registry_key=registry_key,
            query=query,
            query_params=query_params,
            provider_key_name=provider_key_name,
            pool_revision=pool_revision,
            client_bytes=client_bytes,
            request_bytes=request_bytes,
            removed_tools=tuple(removed_tools),
            stream=stream,
        )
        _log_body(request, call, client_bytes, part="request")
        upstream_request = _upstream_request(request, call, key)
        try:
            upstream_response = await request.app.state.passthrough_client.send(
                upstream_request, stream=True
            )
        except httpx.HTTPError as error:
            logger.exception("Pass-through upstream request failed")
            await _record_usage(
                request, call, 502, b"", event_stream=False, failure=repr(error)
            )
            raise _Reject(502, "Upstream request failed") from error
    except _Reject as reject:
        return JSONResponse(
            status_code=reject.status,
            content=UPSTREAMS[provider].error_body(reject.status, reject.message),
        )

    event_stream = (
        "text/event-stream" in upstream_response.headers.get("content-type", "").lower()
    )
    response_headers = {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() not in _STRIPPED_RESPONSE_HEADERS
    }

    async def response_body() -> AsyncIterator[bytes]:
        body = bytearray()
        failure = None
        try:
            async for chunk in upstream_response.aiter_bytes():
                body.extend(chunk)
                yield chunk
        except httpx.HTTPError as error:
            if telemetry.is_enabled():
                sentry_sdk.get_isolation_scope().set_tags(
                    telemetry.sanitize_attributes(
                        {
                            **telemetry.model_attributes(
                                operation="passthrough", model=call.registry_key
                            ),
                            "gateway.error.provider": call.provider,
                            "gateway.error.phase": "passthrough_response_body",
                            "run_id": call.query.run_id,
                            "question_id": call.query.question_id,
                            "query_id": call.query.query_id,
                            "attempt_id": call.query.attempt_id,
                        }
                    )
                )
            failure = f"Upstream stream failed: {error!r}"
            raise
        finally:
            with log_failure(logger, "Pass-through upstream response close failed"):
                await upstream_response.aclose()
            _log_body(
                request,
                call,
                bytes(body),
                part="response",
                status_code=upstream_response.status_code,
            )
            exchanges: ExchangeStore | None = request.app.state.passthrough_exchanges
            exchange = None
            if exchanges is not None:
                query = call.query
                exchange = exchanges.save(
                    query.run_id,
                    query.attempt_id,
                    query.query_id,
                    {
                        "request": call.client_bytes,
                        # What went upstream, when provider-side tools were removed.
                        **(
                            {"forwarded": call.request_bytes}
                            if call.removed_tools
                            else {}
                        ),
                        "response": bytes(body),
                        "headers": json.dumps(
                            {
                                "request": _saved_headers(
                                    request.headers, _UNSAVED_HEADER_PARTS
                                ),
                                # Provider rate-limit headers contain "token".
                                "response": _saved_headers(
                                    upstream_response.headers, ("cookie",)
                                ),
                            }
                        ).encode(),
                    },
                )
            await _record_usage(
                request,
                call,
                upstream_response.status_code,
                bytes(body),
                event_stream=event_stream,
                exchange=exchange,
                failure=failure,
            )

    return StreamingResponse(
        response_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


_ROUTES: tuple[tuple[str, str, bool | None], ...] = (
    ("openai", "/passthrough/openai/v1/responses", None),
    ("anthropic", "/passthrough/anthropic/v1/messages", None),
    ("google", "/passthrough/google/v1beta/models/{model}:generateContent", False),
    ("google", "/passthrough/google/v1beta/models/{model}:streamGenerateContent", True),
    # Chat Completions only: `/api/v1/responses` and `/api/v1/completions` stay 404.
    ("openrouter", "/passthrough/openrouter/api/v1/chat/completions", None),
)


# The provider's own `GET /models`, filtered to the ids the token may call, so
# every provider field (`description`, `created`, `architecture`, ...) comes through.
_MODEL_LIST_ROUTES: tuple[tuple[str, str], ...] = (
    ("openrouter", "/passthrough/openrouter/api/v1/models"),
)


def _model_list_handler(
    provider: str,
) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    upstream = UPSTREAMS[provider]

    async def route(request: Request) -> Response:
        claims = cast(RunTokenClaims, request.state.run_token_claims)
        client = cast(httpx.AsyncClient, request.app.state.passthrough_client)
        try:
            upstream_response = await client.get(
                f"{upstream.base_url}{request.url.path.removeprefix(f'/passthrough/{provider}')}"
            )
            upstream_response.raise_for_status()
            listed = cast(list[dict[str, Any]], upstream_response.json()["data"])
        except (httpx.HTTPError, ValueError, KeyError):
            logger.exception("Pass-through model list failed")
            return JSONResponse(
                status_code=502,
                content=upstream.error_body(502, "Upstream request failed"),
            )
        separator = upstream.variant_separator
        return JSONResponse(
            {
                "data": [
                    entry
                    for entry in listed
                    if isinstance(model := entry.get("id"), str)
                    and (separator is None or separator not in model)
                    and _registry_key(provider, model, claims) is not None
                ]
            }
        )

    return route


def _route_handler(
    provider: str, fixed_stream: bool | None
) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    async def route(request: Request) -> Response:
        try:
            request_bytes = await request.body()
        except ClientDisconnect:
            return Response(status_code=499)
        payload = _json_object(request_bytes)
        if payload is None:
            claims = cast(RunTokenClaims, request.state.run_token_claims)
            log_gateway_event(
                "gateway.run_token.egress_blocked",
                route="passthrough",
                run_id=claims.run_id,
                task_id=claims.task_id,
                reason="malformed_body",
            )
            return JSONResponse(
                status_code=400,
                content=UPSTREAMS[provider].error_body(
                    400, "Request body must be a JSON object without repeated keys"
                ),
            )
        path_model = request.path_params.get("model")
        model = str(path_model or payload.get("model") or "")
        stream = (
            fixed_stream if fixed_stream is not None else payload.get("stream") is True
        )
        return await _proxy(
            request,
            provider=provider,
            model=model,
            upstream_path=request.url.path.removeprefix(f"/passthrough/{provider}"),
            payload=payload,
            client_bytes=request_bytes,
            stream=stream,
        )

    return route


def _websocket_refusal() -> JSONResponse:
    return JSONResponse(
        _openai_error(426, "WebSockets aren't supported here; use HTTPS."),
        status_code=426,
    )


async def _refuse_websocket_get(_: Request) -> Response:
    return _websocket_refusal()


async def _refuse_websocket_upgrade(websocket: WebSocket) -> None:
    """Answers a Responses API WebSocket handshake with 426 so the client falls back to HTTPS.

    Codex tries `wss://.../responses` first and retries five times on most errors, about
    6.5 s per process; on 426 it switches to HTTPS straight away. Without this route the
    server would refuse the handshake with 403.
    """
    await websocket.send_denial_response(_websocket_refusal())


def register_passthrough_routes(app: FastAPI) -> None:
    # Caps upstream connections at 2,000 per worker. A stream holds its connection
    # until it ends, CPU autoscaling adds tasks well before a worker reaches the cap,
    # and a call that waits 10 s for a free connection fails with 502.
    app.state.passthrough_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=10, pool=10),
        limits=httpx.Limits(max_connections=2000, max_keepalive_connections=100),
    )
    bucket = os.environ.get("GATEWAY_PASSTHROUGH_EXCHANGE_BUCKET")
    app.state.passthrough_exchanges = ExchangeStore(bucket) if bucket else None
    log_group = os.environ.get("GATEWAY_PASSTHROUGH_EXCHANGE_LOG_GROUP")
    app.state.passthrough_exchange_log = ExchangeLog(log_group) if log_group else None
    for provider, path, stream in _ROUTES:
        name = f"{provider}_{path.rsplit('/', 1)[-1].replace('{model}', 'model').replace(':', '_')}"
        app.add_api_route(
            path, _route_handler(provider, stream), methods=["POST"], name=name
        )
    for provider, path in _MODEL_LIST_ROUTES:
        app.add_api_route(
            path,
            _model_list_handler(provider),
            methods=["GET"],
            name=f"{provider}_models",
        )
    app.add_api_route(
        "/passthrough/openai/v1/responses",
        _refuse_websocket_get,
        methods=["GET"],
        name="openai_responses_websocket",
    )
    app.add_api_websocket_route(
        "/passthrough/openai/v1/responses",
        _refuse_websocket_upgrade,
        name="openai_responses_websocket_upgrade",
    )
