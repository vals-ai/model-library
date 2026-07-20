"""Optional OpenTelemetry helpers shared by gateway and model-library code.

Telemetry is disabled by default and must never affect request correctness. This
module keeps tracing imports optional so the base library can run without server
extras installed; when disabled or unavailable, every helper is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import Any, cast

logger = logging.getLogger(__name__)

MAX_ATTRIBUTE_LENGTH = 4096
MAX_CONFIG_JSON_LENGTH = 8192
MAX_IDENTITY_JSON_LENGTH = 4096
MAX_IDENTITY_DEPTH = 8
DEFAULT_SERVICE_NAME = "model-proxy-gateway"
TRACER_NAME = "model_library.gateway"
HTTP_TRACE_EXCLUDED_ROUTES = frozenset({"/health/live", "/health/ready"})
HTTP_TRACE_ALLOWED_ROUTES = frozenset(
    {
        "/benchmark-runs/acquire",
        "/benchmark-runs/release",
        "/benchmark-runs/renew",
        "/benchmark-runs/wait",
        "/embeddings",
        "/files/upload",
        "/models",
        "/moderation",
        "/query",
        "/rate-limit",
        "/registry",
        "/token-retry/status",
        "/tokens/count",
    }
)
SENSITIVE_ATTRIBUTE_KEYS = frozenset({"authorization"})
SENSITIVE_ATTRIBUTE_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "content_base64",
        "credential",
        "hmac",
        "password",
        "secret",
    }
)
CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS = frozenset(
    {"identity", "in_agent", "query_id", "question_id", "run_id"}
)
CONTENT_SAFE_ATTRIBUTE_KEYS = frozenset(
    {
        "prompt_cache_retention",
        "reasoning_context",
        "request_limit",
        "request_timeout",
        "request_timeout_seconds",
        "response_format",
        "response_json_schema",
        "response_mime_type",
    }
)
CONTENT_SAFE_SUFFIXES = (
    "_cache_retention",
    "_format",
    "_json_schema",
    "_limit",
    "_mime_type",
    "_timeout",
    "_timeout_seconds",
)
CONTENT_BEARING_CONFIG_KEYS = frozenset(
    {
        "messages",
        "output",
        "outputs",
        "prompt",
        "request",
        "request_json",
        "response",
        "response_json",
        "response_text",
        "system",
        "system_prompt",
        "text",
    }
)
SENTRY_TAG_ATTRIBUTE_KEYS = frozenset(
    {
        "run_id",
        "question_id",
        "query_id",
        "in_agent",
        "gateway.route",
        "gateway.operation",
        "gateway.status_code",
        "gateway.usage_event_id",
        "gateway.error.code",
        "gateway.error.phase",
        "gateway.error.provider",
        "gateway.error_code",
        "gateway.error_phase",
        "gateway.error_provider",
        "http.status_code",
        "http.response.status_code",
        "model.provider",
        "model.name",
        "model.registry_key",
        "model.provider_endpoint",
        "model.param_group",
        "gen_ai.system",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gateway.retry_queue.mode",
        "gateway.output_schema.mode",
        "retry_queue.mode",
        "retry_queue.question_ref",
        "retry_queue.dynamic_estimate.mode",
        "retry_queue.benchmark_queue.mode",
        "retry_queue.priority",
        "retry_queue.attempt",
        "retry_queue.max_tries",
        "llm.in_agent.mode",
        "llm.output_schema.mode",
        "llm.config.max_tokens",
        "llm.config.temperature",
        "llm.config.top_p",
        "llm.config.top_k",
        "llm.config.reasoning.mode",
        "llm.config.reasoning_effort",
        "llm.config.compute_effort",
        "retry.strategy",
        "retry.attempt",
        "retry.attempts",
        "retry.immediate_attempts",
        "retry.max_tries",
    }
)
SENTRY_PROVIDER_CONFIG_ATTRIBUTE_PREFIX = "llm.config.provider_config."

AttributeValue = str | bool | int | float
JsonValue = None | str | bool | int | float | list["JsonValue"] | dict[str, "JsonValue"]


class IdentityValidationError(ValueError):
    """Raised when identity is not a bounded JSON object."""


_enabled = False
_configured = False
_httpx_instrumented = False
_provider: Any | None = None
_trace_api: Any | None = None
_span_kind: Any | None = None
_status: Any | None = None
_status_code: Any | None = None
_sentry_context: ContextVar[dict[str, AttributeValue]] = ContextVar(
    "gateway_sentry_context",
    default={},
)
CONFIG_SEEN_CACHE_MAX_SIZE = 4096
CONFIG_SEEN_CACHE_TTL_SECONDS = 300.0
_seen_config_hashes: dict[str, float] = {}


def _env_enabled() -> bool:
    return os.environ.get("GATEWAY_OTEL_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_enabled() -> bool:
    """Return whether OpenTelemetry helpers should emit spans/events."""
    return _enabled


def _load_trace_api() -> bool:
    global _span_kind, _status, _status_code, _trace_api
    if _trace_api is not None:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        return False
    _trace_api = trace
    _span_kind = SpanKind
    _status = Status
    _status_code = StatusCode
    return True


def configure_telemetry(app: object | None = None) -> bool:
    """Configure OpenTelemetry tracing when explicitly enabled.

    Standard OpenTelemetry environment variables configure exporter endpoint,
    headers, sampling, and resource attributes. The gateway-specific switch is
    ``GATEWAY_OTEL_ENABLED``; disabled or missing dependencies leave helpers as
    no-ops.
    """
    global _configured, _enabled, _httpx_instrumented, _provider

    if not _env_enabled():
        _enabled = False
        return False
    if _configured:
        _enabled = True
        return True

    try:
        httpx_module = import_module("opentelemetry.instrumentation.httpx")
        resources_module = import_module("opentelemetry.sdk.resources")
        sdk_trace_module = import_module("opentelemetry.sdk.trace")
        sentry_module = import_module("sentry_sdk")
        sentry_consts_module = import_module("sentry_sdk.consts")
        sentry_logging_module = import_module("sentry_sdk.integrations.logging")
        sentry_otlp_module = import_module("sentry_sdk.integrations.otlp")
    except ImportError as exc:
        logger.warning("Sentry OpenTelemetry requested but unavailable: %s", exc)
        _enabled = False
        return False

    HTTPXClientInstrumentor = httpx_module.HTTPXClientInstrumentor
    Resource = resources_module.Resource
    TracerProvider = sdk_trace_module.TracerProvider
    INSTRUMENTER = sentry_consts_module.INSTRUMENTER
    LoggingIntegration = sentry_logging_module.LoggingIntegration
    OTLPIntegration = sentry_otlp_module.OTLPIntegration

    if not _load_trace_api():
        logger.warning("OpenTelemetry requested but trace API is unavailable")
        _enabled = False
        return False
    trace_api = _trace_api
    assert trace_api is not None

    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        logger.warning("Sentry tracing requested but SENTRY_DSN is not set")
        _enabled = False
        return False

    service_name = os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    environment = os.environ.get("GATEWAY_STAGE") or os.environ.get(
        "SENTRY_ENVIRONMENT"
    )
    try:
        resource = Resource.create(_resource_attributes(service_name))
        provider = TracerProvider(
            resource=resource,
            sampler=_otel_sampler(sdk_trace_module.sampling),
        )
        trace_api.set_tracer_provider(provider)
        _provider = provider
        sentry_module.init(
            dsn=dsn,
            environment=environment,
            release=os.environ.get("SENTRY_RELEASE", ""),
            server_name=service_name,
            instrumenter=INSTRUMENTER.OTEL,
            enable_logs=True,
            send_default_pii=True,
            before_send=_before_send,
            before_send_transaction=_before_send_transaction,
            before_send_log=_before_send_log,
            integrations=[
                LoggingIntegration(
                    level=None,
                    event_level=None,
                    sentry_logs_level=logging.WARNING,
                ),
                OTLPIntegration(
                    setup_otlp_traces_exporter=True,
                    collector_url=os.environ.get("SENTRY_OTLP_COLLECTOR_URL") or None,
                    setup_propagator=True,
                    capture_exceptions=False,
                ),
            ],
        )

        if not _httpx_instrumented:
            HTTPXClientInstrumentor().instrument(
                request_hook=_httpx_request_hook,
                async_request_hook=_httpx_async_request_hook,
            )
            _httpx_instrumented = True
    except Exception as exc:
        logger.warning(
            "Sentry OpenTelemetry requested but failed to initialize: %s", exc
        )
        if _provider is not None:
            _provider.shutdown()
        _provider = None
        _enabled = False
        return False

    _configured = True
    _enabled = True
    logger.info("Sentry OpenTelemetry tracing enabled for service %s", service_name)
    return True


def _resource_attributes(service_name: str) -> dict[str, str]:
    attributes = {"service.name": service_name}
    raw_attributes = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    for raw_pair in raw_attributes.split(","):
        key, separator, value = raw_pair.partition("=")
        if separator and key.strip() and value.strip():
            attributes[key.strip()] = value.strip()
    attributes["service.name"] = service_name
    return attributes


def _otel_sampler(sampling_module: Any) -> Any:
    sampler_name = os.environ.get("OTEL_TRACES_SAMPLER", "parentbased_always_on")
    sampler_arg = os.environ.get("OTEL_TRACES_SAMPLER_ARG")
    known_samplers = sampling_module._KNOWN_SAMPLERS
    if sampler_name not in known_samplers:
        raise ValueError(f"Unsupported OTEL_TRACES_SAMPLER={sampler_name!r}")

    sampler = known_samplers[sampler_name]
    if sampler_name in {"traceidratio", "parentbased_traceidratio"}:
        if sampler_arg is None:
            raise ValueError(f"OTEL_TRACES_SAMPLER_ARG is required for {sampler_name}")
        try:
            ratio = float(sampler_arg)
        except ValueError as exc:
            raise ValueError("OTEL_TRACES_SAMPLER_ARG must be a number") from exc
        if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
            raise ValueError("OTEL_TRACES_SAMPLER_ARG must be between 0 and 1")
        return sampler(ratio)

    return sampler


def _httpx_method_and_path(request_info: object) -> tuple[str, str] | None:
    method = getattr(request_info, "method", b"")
    url = getattr(request_info, "url", None)
    if isinstance(method, bytes):
        method_text = method.decode(errors="ignore")
    else:
        method_text = str(method)
    path = getattr(url, "path", None)
    if not method_text or not isinstance(path, str):
        return None
    return method_text.upper(), path or "/"


def _rename_httpx_span(span: object, request_info: object) -> None:
    method_and_path = _httpx_method_and_path(request_info)
    if method_and_path is None:
        return
    method, path = method_and_path
    update_name = getattr(span, "update_name", None)
    if callable(update_name):
        update_name(f"{method} {path}")
    set_attributes_method = getattr(span, "set_attributes", None)
    if callable(set_attributes_method):
        set_attributes_method(
            {
                "sentry.op": "http.client",
                "sentry.origin": "auto.httpx.gateway",
                "http.method": method,
                "http.request.method": method,
                "http.route": path,
                "url.path": path,
            }
        )


def _httpx_request_hook(span: object, request_info: object) -> None:
    _rename_httpx_span(span, request_info)


async def _httpx_async_request_hook(span: object, request_info: object) -> None:
    _rename_httpx_span(span, request_info)


def _attach_sentry_search_tags(event: dict[str, Any]) -> dict[str, object] | None:
    raw_tags = event.setdefault("tags", {})
    if not isinstance(raw_tags, dict):
        return None
    tags = cast(dict[str, object], raw_tags)
    for key, value in _sentry_search_context().items():
        tags[key] = str(value)[:200]
    return tags


def _before_send(event: dict[str, Any], hint: object) -> dict[str, Any] | None:
    """Attach current gateway debug fields as Sentry event tags."""
    tags = _attach_sentry_search_tags(event)
    if tags is not None:
        fingerprint = _sentry_fingerprint(event, hint, tags)
        if fingerprint:
            event["fingerprint"] = fingerprint
    return event


def _sentry_fingerprint(
    event: Mapping[str, Any], hint: object, tags: Mapping[str, object]
) -> list[str] | None:
    error_code = tags.get("gateway.error.code")
    if not error_code:
        return None
    error_provider = tags.get("gateway.error.provider", "unknown_provider")
    error_phase = tags.get("gateway.error.phase", "unknown_phase")
    exception_type = "unknown_exception"
    if isinstance(hint, Mapping):
        hint_mapping = cast(Mapping[str, object], hint)
        exc_info = hint_mapping.get("exc_info")
        if isinstance(exc_info, tuple) and exc_info:
            exc_type = cast(object, exc_info[0])
            exception_type = getattr(exc_type, "__name__", str(exc_type))
    if exception_type == "unknown_exception":
        exception = event.get("exception", {})
        values: object = None
        if isinstance(exception, Mapping):
            exception_mapping = cast(Mapping[str, object], exception)
            values = exception_mapping.get("values", [])
        if isinstance(values, list) and values:
            first = cast(object, values[0])
            if isinstance(first, Mapping):
                first_mapping = cast(Mapping[str, object], first)
                exception_type = str(first_mapping.get("type", exception_type))
    return [
        "gateway",
        str(error_code),
        str(error_provider),
        str(error_phase),
        exception_type,
    ]


def _before_send_transaction(
    event: dict[str, Any], hint: object
) -> dict[str, Any] | None:
    """Attach active gateway debug fields to Sentry transaction tags.

    Route filtering happens before manual spans are created in the gateway HTTP
    and auth middleware. This callback intentionally avoids parsing Sentry's
    transaction event shape; it is only a final tag-scrubbing hook.
    """
    _attach_sentry_search_tags(event)
    return event


def _before_send_log(log: dict[str, Any], _hint: object) -> dict[str, Any] | None:
    """Attach active OTel trace identifiers and gateway fields to Sentry logs."""
    log.update(_sentry_search_context())
    if not _load_trace_api():
        return log
    trace_api = _trace_api
    assert trace_api is not None
    span_context = trace_api.get_current_span().get_span_context()
    if span_context.is_valid:
        log["trace_id"] = f"{span_context.trace_id:032x}"
        log["span_id"] = f"{span_context.span_id:016x}"
    return log


def _is_sentry_tag_attribute(key: str) -> bool:
    if key in SENTRY_TAG_ATTRIBUTE_KEYS:
        return True
    if not key.startswith(SENTRY_PROVIDER_CONFIG_ATTRIBUTE_PREFIX):
        return False
    provider_config_key = key.removeprefix(SENTRY_PROVIDER_CONFIG_ATTRIBUTE_PREFIX)
    provider_config_key = provider_config_key.removesuffix(".mode")
    return is_safe_config_attribute_key(provider_config_key)


def _sentry_search_context() -> dict[str, AttributeValue]:
    return {
        key: value
        for key, value in _sentry_context.get().items()
        if _is_sentry_tag_attribute(key)
    }


def _sentry_search_attributes(
    attributes: Mapping[str, AttributeValue],
) -> dict[str, AttributeValue]:
    return {
        key: value for key, value in attributes.items() if _is_sentry_tag_attribute(key)
    }


def _merge_sentry_context(attributes: Mapping[str, AttributeValue]) -> None:
    searchable = _sentry_search_attributes(attributes)
    if not searchable:
        return
    _sentry_context.set({**_sentry_context.get(), **searchable})


def shutdown_telemetry() -> None:
    """Flush and shut down the configured tracer provider, if any."""
    global _configured, _enabled, _provider
    try:
        sentry_module = import_module("sentry_sdk")
        sentry_module.flush(timeout=0)
        sentry_module.get_client().close(timeout=0)
    except Exception:
        pass
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _configured = False
    _enabled = False
    _sentry_context.set({})
    _seen_config_hashes.clear()


def config_fingerprint(
    model: str,
    config: Mapping[str, object | None],
    params: Mapping[str, object | None] | None = None,
    token_retry_params: object | None = None,
) -> tuple[str, dict[str, object]]:
    """Return a stable hash plus redacted config mapping for lookup events."""
    fingerprint_params = {
        key: value
        for key, value in (params or {}).items()
        if key not in CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS
    }
    redacted = {
        "model": model,
        "config": _redact_config_mapping(config),
        "params": _redact_config_mapping(fingerprint_params),
        "token_retry_params": _redact_object(
            token_retry_params, redact_content_config_keys=True
        ),
    }
    payload = json.dumps(redacted, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16], redacted


def record_config_seen(
    config_hash: str,
    redacted_config: Mapping[str, object],
    *,
    force: bool = False,
) -> None:
    """Emit a deduplicated trace event mapping a config hash to redacted config."""
    span = _current_recording_span()
    if span is None:
        return
    now = time.monotonic()
    _prune_seen_config_hashes(now)
    if not force and config_hash in _seen_config_hashes:
        return

    redacted_json, truncated = _config_json_for_attribute(redacted_config)
    span.add_event(
        "model.config_seen",
        {
            "model.config_hash": config_hash,
            "model.config_redacted_json": redacted_json,
            "model.config_redacted_json_truncated": truncated,
        },
    )
    _seen_config_hashes[config_hash] = now


def json_attribute(value: object, *, max_length: int = MAX_ATTRIBUTE_LENGTH) -> str:
    """Return bounded JSON for a telemetry attribute, redacting credential keys."""
    try:
        payload = json.dumps(
            _redact_object(value), sort_keys=True, separators=(",", ":"), default=str
        )
    except Exception:
        payload = str(value)
    return payload[:max_length]


def normalize_identity(identity: object) -> dict[str, JsonValue]:
    """Return identity if it is a bounded JSON object."""
    normalized = _normalize_identity_value(identity, depth=0)
    if not isinstance(normalized, dict):
        raise IdentityValidationError("identity must be a JSON object")
    _check_identity_json_size(identity)
    _check_identity_json_size(normalized)
    return normalized


def _check_identity_json_size(identity: object) -> None:
    try:
        payload = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise IdentityValidationError("identity must be JSON-safe") from exc
    if len(payload) > MAX_IDENTITY_JSON_LENGTH:
        raise IdentityValidationError(
            f"identity JSON must be at most {MAX_IDENTITY_JSON_LENGTH} bytes"
        )


def _normalize_identity_value(
    value: object,
    *,
    depth: int,
) -> JsonValue:
    if depth > MAX_IDENTITY_DEPTH:
        raise IdentityValidationError(
            f"identity JSON depth must be at most {MAX_IDENTITY_DEPTH}"
        )
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityValidationError("identity numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        mapping = cast(Mapping[object, object], value)
        for item_key, item_value in mapping.items():
            if not isinstance(item_key, str):
                raise IdentityValidationError("identity object keys must be strings")
            normalized[item_key] = _normalize_identity_value(
                item_value, depth=depth + 1
            )
        return normalized
    if isinstance(value, list | tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return [_normalize_identity_value(item, depth=depth + 1) for item in items]
    raise IdentityValidationError("identity values must be JSON-safe")


def sanitize_attributes(
    attributes: Mapping[str, object | None] | None,
) -> dict[str, AttributeValue]:
    """Return safe scalar OpenTelemetry attributes.

    The gateway records high-signal request/debug fields as bounded scalar
    attributes. Secrets and credentials stay excluded. Callers should avoid
    passing prompts, responses, raw inputs, or payload-like values as attributes;
    this helper only bounds/redacts values that are explicitly supplied.
    """
    if not attributes:
        return {}

    sanitized: dict[str, AttributeValue] = {}
    for key, raw_value in attributes.items():
        if raw_value is None or _is_sensitive_key(key):
            continue
        value = _sanitize_value(raw_value)
        if value is not None:
            sanitized[key] = value
    for source, alias in (
        ("gateway.error.code", "gateway.error_code"),
        ("gateway.error.phase", "gateway.error_phase"),
        ("gateway.error.provider", "gateway.error_provider"),
    ):
        if source in sanitized and alias not in sanitized:
            sanitized[alias] = sanitized[source]
    return sanitized


def _should_redact_key(key: str, *, redact_content_config_keys: bool) -> bool:
    return _is_sensitive_key(key) or (
        redact_content_config_keys and _is_content_bearing_config_key(key)
    )


def _redact_mapping(
    mapping: Mapping[str, object | None],
    *,
    redact_content_config_keys: bool = False,
) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in sorted(mapping.items()):
        if key == "custom_endpoint":
            redacted[key] = "custom" if value else "default"
        elif _should_redact_key(
            key, redact_content_config_keys=redact_content_config_keys
        ):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = _redact_object(
                value, redact_content_config_keys=redact_content_config_keys
            )
    return redacted


def _redact_object(
    value: object | None,
    *,
    redact_content_config_keys: bool = False,
) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _redact_mapping(
            cast(Mapping[str, object | None], value),
            redact_content_config_keys=redact_content_config_keys,
        )
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return _redact_mapping(
                cast(Mapping[str, object | None], dumped),
                redact_content_config_keys=redact_content_config_keys,
            )
        return dumped
    if isinstance(value, list | tuple):
        return [
            _redact_object(item, redact_content_config_keys=redact_content_config_keys)
            for item in cast(list[object], value)
        ]
    return str(value)


def _redact_config_mapping(mapping: Mapping[str, object | None]) -> dict[str, object]:
    return _redact_mapping(mapping, redact_content_config_keys=True)


def _config_json_for_attribute(
    redacted_config: Mapping[str, object],
) -> tuple[str, bool]:
    config_json = json.dumps(redacted_config, sort_keys=True, separators=(",", ":"))
    if len(config_json) <= MAX_CONFIG_JSON_LENGTH:
        return config_json, False

    summary = {
        "truncated": True,
        "top_level_keys": sorted(redacted_config.keys()),
        "config_keys": _mapping_keys(redacted_config.get("config")),
        "param_keys": _mapping_keys(redacted_config.get("params")),
        "token_retry_param_keys": _mapping_keys(
            redacted_config.get("token_retry_params")
        ),
    }
    return json.dumps(summary, sort_keys=True, separators=(",", ":")), True


def _mapping_keys(value: object | None) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    mapping = cast(Mapping[str, object | None], value)
    return sorted(mapping.keys())


def _prune_seen_config_hashes(now: float) -> None:
    expired = [
        config_hash
        for config_hash, seen_at in _seen_config_hashes.items()
        if now - seen_at > CONFIG_SEEN_CACHE_TTL_SECONDS
    ]
    for config_hash in expired:
        _seen_config_hashes.pop(config_hash, None)

    overflow = len(_seen_config_hashes) - CONFIG_SEEN_CACHE_MAX_SIZE
    if overflow <= 0:
        return
    oldest = sorted(_seen_config_hashes.items(), key=lambda item: item[1])[:overflow]
    for config_hash, _seen_at in oldest:
        _seen_config_hashes.pop(config_hash, None)


def is_sensitive_key(key: str) -> bool:
    """Return whether an attribute/config key can contain sensitive content."""
    return _is_sensitive_key(key)


def is_safe_config_attribute_key(key: str) -> bool:
    """Return whether a config key is safe to expose as a scalar attribute."""
    return not _is_sensitive_key(key) and not _is_content_bearing_config_key(key)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in CONTENT_SAFE_ATTRIBUTE_KEYS or lowered.endswith(
        CONTENT_SAFE_SUFFIXES
    ):
        return False
    if lowered in SENSITIVE_ATTRIBUTE_KEYS:
        return True
    if any(part in lowered for part in SENSITIVE_ATTRIBUTE_PARTS):
        return True

    return False


def _is_content_bearing_config_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in CONTENT_SAFE_ATTRIBUTE_KEYS or lowered.endswith(
        CONTENT_SAFE_SUFFIXES
    ):
        return False
    return lowered in CONTENT_BEARING_CONFIG_KEYS


def _sanitize_value(value: object) -> AttributeValue | None:
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:MAX_ATTRIBUTE_LENGTH]
    return str(value)[:MAX_ATTRIBUTE_LENGTH]


def should_trace_http_route(route: str) -> bool:
    """Return whether an HTTP route is useful enough to export as a span."""
    return route in HTTP_TRACE_ALLOWED_ROUTES


def http_server_span_name(method: str, route: str) -> str:
    """Return the finite Sentry-visible name for a gateway HTTP request span."""
    return f"{method.upper()} {route or '/'}"


def _kind(kind: str) -> Any:
    if _span_kind is None:
        return None
    return {
        "client": _span_kind.CLIENT,
        "consumer": _span_kind.CONSUMER,
        "internal": _span_kind.INTERNAL,
        "producer": _span_kind.PRODUCER,
        "server": _span_kind.SERVER,
    }.get(kind, _span_kind.INTERNAL)


def _sentry_op_for_span(name: str, kind: str) -> str:
    if kind == "server":
        return "http.server"
    if kind == "client" or name.endswith(".provider_call"):
        return "gen_ai.request"
    if name.startswith("model_library.retry_"):
        return "retry"
    return "function"


def _sentry_origin_for_span(name: str) -> str:
    if name.startswith("gateway.") or name.startswith(("GET ", "POST ")):
        return "manual.gateway"
    return "manual.model_library"


def _span_attributes(
    name: str,
    attributes: Mapping[str, object | None] | None,
    kind: str,
) -> dict[str, AttributeValue]:
    raw_attributes: dict[str, object | None] = {**_sentry_search_context()}
    raw_attributes.update(dict(attributes or {}))
    raw_attributes.setdefault("sentry.op", _sentry_op_for_span(name, kind))
    raw_attributes.setdefault("sentry.origin", _sentry_origin_for_span(name))
    return sanitize_attributes(raw_attributes)


@contextmanager
def start_span(
    name: str,
    attributes: Mapping[str, object | None] | None = None,
    *,
    kind: str = "internal",
) -> Generator[Any | None, None, None]:
    """Start a span if telemetry is enabled, otherwise yield ``None``."""
    if not is_enabled() or not _load_trace_api():
        yield None
        return

    trace_api = _trace_api
    assert trace_api is not None
    tracer = trace_api.get_tracer(TRACER_NAME)
    sanitized_attributes = _span_attributes(name, attributes, kind)
    parent_context = _sentry_context.get()
    context_token = _sentry_context.set(
        {**parent_context, **_sentry_search_attributes(sanitized_attributes)}
    )
    try:
        with tracer.start_as_current_span(
            name,
            kind=_kind(kind),
            attributes=sanitized_attributes,
        ) as span:
            yield span
    finally:
        child_context = _sentry_context.get()
        _sentry_context.reset(context_token)
        if parent_context:
            _sentry_context.set(
                {**parent_context, **_sentry_search_attributes(child_context)}
            )


def set_attributes(attributes: Mapping[str, object | None]) -> None:
    """Set attributes on the current span when recording."""
    span = _current_recording_span()
    if span is None:
        return
    sanitized_attributes = sanitize_attributes(attributes)
    span.set_attributes(sanitized_attributes)
    _merge_sentry_context(sanitized_attributes)


def add_event(
    name: str,
    attributes: Mapping[str, object | None] | None = None,
) -> None:
    """Add an event to the current span when recording."""
    span = _current_recording_span()
    if span is None:
        return
    sanitized_attributes = sanitize_attributes(attributes)
    span.add_event(name, sanitized_attributes)
    _merge_sentry_context(sanitized_attributes)


def record_exception(
    exc: BaseException,
    attributes: Mapping[str, object | None] | None = None,
) -> None:
    """Record a sanitized exception event on the current span.

    OpenTelemetry's standard ``record_exception`` includes exception messages and
    stack traces by default. Provider errors can include response/request text, so
    the gateway records only the exception type plus caller-supplied safe fields.
    """
    sanitized_attributes = sanitize_attributes(
        {"exception.type": type(exc).__name__, **dict(attributes or {})}
    )
    _merge_sentry_context(sanitized_attributes)
    span = _current_recording_span()
    if span is not None:
        span.add_event("exception", sanitized_attributes)
    if is_enabled():
        try:
            sentry_module = import_module("sentry_sdk")
            sentry_module.capture_exception(exc)
        except Exception as sentry_exc:
            logger.debug("Sentry capture_exception failed: %s", sentry_exc)


def set_status_ok() -> None:
    """Mark the current span as successful."""
    span = _current_recording_span()
    if span is None or _status is None or _status_code is None:
        return
    span.set_status(_status(_status_code.OK))


def set_status_error(description: str | None = None) -> None:
    """Mark the current span as failed."""
    span = _current_recording_span()
    if span is None or _status is None or _status_code is None:
        return
    span.set_status(_status(_status_code.ERROR, description=description))


def model_attributes(*, operation: str, model: str) -> dict[str, str]:
    """Return common low-risk GenAI/gateway span attributes."""
    provider, _, model_name = model.partition("/")
    return {
        "gateway.operation": operation,
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "gen_ai.system": provider or "unknown",
        "model.provider": provider or "unknown",
        "model.name": model_name or model,
        "model.registry_key": model,
    }


def mode_attribute(enabled: bool) -> str:
    """Return a string mode for Sentry-searchable boolean state."""
    return "enabled" if enabled else "disabled"


def run_attributes(request_context: Mapping[str, object]) -> dict[str, object | None]:
    """Return run/question identifiers from gateway or LLM request context."""
    in_agent = request_context.get("in_agent")
    identity = request_context.get("identity")
    identity_attribute: str | None = None
    if isinstance(identity, Mapping):
        try:
            identity_mapping = cast(Mapping[object, object], identity)
            identity_attribute = json.dumps(
                normalize_identity(identity_mapping),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except IdentityValidationError:
            identity_attribute = None
    return {
        "run_id": request_context.get("run_id"),
        "question_id": request_context.get("question_id"),
        "query_id": request_context.get("query_id"),
        "identity": identity_attribute,
        "in_agent": in_agent,
        "llm.in_agent.mode": mode_attribute(bool(in_agent)),
    }


def is_recording() -> bool:
    """Return whether the current span is recording telemetry."""
    return _current_recording_span() is not None


def _current_recording_span() -> Any | None:
    if not is_enabled() or not _load_trace_api():
        return None
    trace_api = _trace_api
    assert trace_api is not None
    span = trace_api.get_current_span()
    if not span.is_recording():
        return None
    return span
