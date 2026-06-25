"""Durable successful-query usage ledger for the gateway."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, Protocol, cast

from aiobotocore.config import AioConfig
from aiobotocore.session import get_session  # pyright: ignore[reportUnknownVariableType]
from botocore.exceptions import BotoCoreError, ClientError

import model_library.telemetry as telemetry
import model_gateway.usage_ledger.schema as ledger_schema
from model_gateway.metrics import param_group
from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.dynamodb_writer import (
    AsyncDynamoDbClient,
    put_usage_event_async,
)
from model_gateway.usage_ledger.message import serialize_usage_event_message
from model_library.base.output import QueryResult, QueryResultMetadata

logger = logging.getLogger("model_proxy_server.usage_ledger")

UsageLedgerMode = Literal["disabled", "shadow", "enforced"]

DEFAULT_SCHEMA_VERSION = 1
DEFAULT_NORMALIZATION_VERSION = "2026-05-29"
DEFAULT_METADATA_SCHEMA_VERSION = 1
MAX_LEDGER_JSON_LENGTH = 64_000
DEFAULT_SHARD_COUNT = ledger_schema.DEFAULT_SHARD_COUNT
DEFAULT_MAX_POOL_CONNECTIONS = 1000


class UsageLedgerWriteError(RuntimeError):
    """Raised when an enforced usage ledger write fails."""


class UsageLedger(Protocol):
    enabled: bool

    async def start(self) -> None:
        """Open any app-lifetime resources needed by the ledger."""

    async def close(self) -> None:
        """Close any app-lifetime resources opened by the ledger."""

    async def write_success(self, event: Mapping[str, object]) -> None:
        """Persist a successful query usage event."""


class NoopUsageLedger:
    enabled = False

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_success(self, event: Mapping[str, object]) -> None:
        _ = event
        return


class _DynamoDbClientContext(Protocol):
    async def __aenter__(self) -> AsyncDynamoDbClient: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _SqsClient(Protocol):
    async def send_message(self, **kwargs: object) -> Mapping[str, Any]: ...


class _SqsClientContext(Protocol):
    async def __aenter__(self) -> _SqsClient: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class SqsUsageLedger:
    enabled = True

    def __init__(
        self,
        *,
        queue_url: str,
        mode: UsageLedgerMode,
        region_name: str | None = None,
    ) -> None:
        self._queue_url = queue_url
        self._mode = mode
        self._region_name = region_name
        self._client_context: _SqsClientContext | None = None
        self._client: _SqsClient | None = None

    async def start(self) -> None:
        session = cast(Any, get_session())
        client_context = cast(
            _SqsClientContext,
            session.create_client(
                "sqs",
                region_name=self._region_name or None,
                config=AioConfig(
                    max_pool_connections=_max_pool_connections(),
                    connect_timeout=2,
                    read_timeout=3,
                    retries={"mode": "standard", "max_attempts": 2},
                ),
            ),
        )
        self._client_context = client_context
        try:
            self._client = await client_context.__aenter__()
        except Exception:
            self._client_context = None
            raise

    async def close(self) -> None:
        if self._client_context is None:
            return
        try:
            await self._client_context.__aexit__(None, None, None)
        finally:
            self._client_context = None
            self._client = None

    async def write_success(self, event: Mapping[str, object]) -> None:
        try:
            await self._write_success(event)
        except (BotoCoreError, ClientError, ValueError, TypeError) as exc:
            if self._mode == "shadow":
                logger.warning(
                    "Gateway usage ledger SQS send failed in shadow mode: %s", exc
                )
                return
            raise UsageLedgerWriteError("Gateway usage ledger write failed") from exc

    async def _write_success(self, event: Mapping[str, object]) -> None:
        client = self._client
        if client is None:
            raise ValueError("Gateway usage ledger SQS client is not started")
        await client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=serialize_usage_event_message(event),
        )


class DynamoDbUsageLedger:
    enabled = True

    def __init__(
        self,
        *,
        table_name: str,
        mode: UsageLedgerMode,
        region_name: str | None = None,
    ) -> None:
        self._table_name = table_name
        self._mode = mode
        self._region_name = region_name
        self._client_context: _DynamoDbClientContext | None = None
        self._client: AsyncDynamoDbClient | None = None

    async def start(self) -> None:
        session = cast(Any, get_session())
        client_context = cast(
            _DynamoDbClientContext,
            session.create_client(
                "dynamodb",
                region_name=self._region_name or None,
                config=AioConfig(
                    connector_args={
                        "keepalive_timeout": 60,
                        "ttl_dns_cache": 300,
                        "force_close": False,
                    },
                    max_pool_connections=_max_pool_connections(),
                    connect_timeout=5,
                    read_timeout=30,
                    retries={"mode": "standard", "max_attempts": 3},
                ),
            ),
        )
        self._client_context = client_context
        try:
            self._client = await client_context.__aenter__()
        except Exception:
            self._client_context = None
            raise

    async def close(self) -> None:
        if self._client_context is None:
            return
        try:
            await self._client_context.__aexit__(None, None, None)
        finally:
            self._client_context = None
            self._client = None

    async def write_success(self, event: Mapping[str, object]) -> None:
        try:
            await self._write_success(event)
        except (BotoCoreError, ClientError, ValueError, TypeError) as exc:
            if self._mode == "shadow":
                logger.warning(
                    "Gateway usage ledger write failed in shadow mode: %s", exc
                )
                return
            raise UsageLedgerWriteError("Gateway usage ledger write failed") from exc

    async def _write_success(self, event: Mapping[str, object]) -> None:
        client = self._client
        if client is None:
            raise ValueError("Gateway usage ledger DynamoDB client is not started")
        await put_usage_event_async(
            client=client,
            table_name=self._table_name,
            event=event,
        )


def create_usage_ledger_from_env() -> UsageLedger:
    mode = _usage_ledger_mode()
    if mode == "disabled":
        return NoopUsageLedger()

    region_name = os.environ.get("GATEWAY_USAGE_LEDGER_REGION", "").strip() or None
    queue_url = os.environ.get("GATEWAY_USAGE_LEDGER_QUEUE_URL", "").strip()
    if queue_url:
        return SqsUsageLedger(
            queue_url=queue_url,
            mode=mode,
            region_name=region_name,
        )

    table_name = os.environ.get("GATEWAY_USAGE_LEDGER_TABLE_NAME", "").strip()
    if not table_name:
        if mode == "shadow":
            logger.warning(
                "GATEWAY_USAGE_LEDGER_MODE=shadow but neither "
                "GATEWAY_USAGE_LEDGER_QUEUE_URL nor GATEWAY_USAGE_LEDGER_TABLE_NAME is set"
            )
            return NoopUsageLedger()
        raise ValueError(
            "GATEWAY_USAGE_LEDGER_QUEUE_URL or GATEWAY_USAGE_LEDGER_TABLE_NAME is "
            "required when usage ledger is enforced"
        )

    return DynamoDbUsageLedger(
        table_name=table_name,
        mode=mode,
        region_name=region_name,
    )


def build_success_usage_event(
    *,
    body: QueryRequest,
    config: Mapping[str, Any],
    query_params: Mapping[str, object | None],
    dimensions: Mapping[str, str],
    result: QueryResult,
    completed_at: datetime | None = None,
    api_key_fingerprint: str | None = None,
) -> dict[str, object]:
    completed = completed_at or datetime.now(UTC)
    completed_at_iso = _isoformat(completed)
    day = completed.strftime("%Y%m%d")
    query_id = str(query_params.get("query_id") or body.query_id or "")
    usage_event_id = _usage_event_id()
    shard = _shard(usage_event_id)
    query_config_params = {
        key: value
        for key, value in query_params.items()
        if value is not None
        and key not in telemetry.CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS
    }
    config_hash, redacted_config = telemetry.config_fingerprint(
        body.model,
        config,
        query_config_params,
        body.token_retry_params,
    )
    config_json, config_truncated = _bounded_json(redacted_config)
    metadata_json, metadata_truncated = _bounded_json(
        _metadata_snapshot(result.metadata)
    )
    finish_reason_json, finish_reason_truncated = _bounded_json(
        result.finish_reason.model_dump(mode="json") if result.finish_reason else None
    )
    metadata_counts = _metadata_counts(result.metadata)

    benchmark_name = ledger_schema.identity_dimension_value(
        body.identity, ledger_schema.IDENTITY_BENCHMARK_NAME
    )
    agent_name = ledger_schema.identity_dimension_value(
        body.identity, ledger_schema.IDENTITY_AGENT_NAME
    )

    event: dict[str, object] = {
        ledger_schema.BASE_PK: ledger_schema.usage_day_pk(day, shard),
        ledger_schema.BASE_SK: f"TS#{completed_at_iso}#USG#{usage_event_id}",
        "entity_type": "usage_event",
        "usage_event_id": usage_event_id,
        "run_id": body.run_id,
        "question_id": body.question_id,
        "query_id": query_id,
        "identity": body.identity,
        ledger_schema.IDENTITY_BENCHMARK_NAME: benchmark_name,
        ledger_schema.IDENTITY_AGENT_NAME: agent_name,
        "api_key_fingerprint": api_key_fingerprint,
        "model": body.model,
        "provider": body.model.partition("/")[0] or "unknown",
        "provider_endpoint": "custom" if config.get("custom_endpoint") else "default",
        "param_group": dimensions.get("ParamGroup")
        or param_group(config, query_config_params, body.token_retry_params),
        "config_hash": config_hash,
        "config_redacted_json": config_json,
        "config_redacted_json_truncated": config_truncated,
        "metadata_json": metadata_json,
        "metadata_json_truncated": metadata_truncated,
        "finish_reason_json": finish_reason_json,
        "finish_reason_json_truncated": finish_reason_truncated,
        "completed_at": completed_at_iso,
        "day": day,
        "usage_shard": shard,
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "metadata_schema_version": DEFAULT_METADATA_SCHEMA_VERSION,
        "normalization_version": DEFAULT_NORMALIZATION_VERSION,
        **metadata_counts,
    }
    if body.run_id:
        event[ledger_schema.RUN_INDEX_PK] = ledger_schema.run_pk(body.run_id, shard)
        event[ledger_schema.RUN_INDEX_SK] = (
            f"TS#{completed_at_iso}#QUESTION#{body.question_id or 'none'}"
            f"#QUERY#{query_id}#USG#{usage_event_id}"
        )
    if query_id:
        event[ledger_schema.QUERY_INDEX_PK] = ledger_schema.query_pk(query_id)
        event[ledger_schema.QUERY_INDEX_SK] = f"USG#{usage_event_id}"
    if api_key_fingerprint:
        event[ledger_schema.API_KEY_DAY_INDEX_PK] = ledger_schema.api_key_day_pk(
            api_key_fingerprint, day, shard
        )
        event[ledger_schema.API_KEY_DAY_INDEX_SK] = (
            f"TS#{completed_at_iso}#USG#{usage_event_id}"
        )
    dimension_sk = ledger_schema.dimension_sort_key(
        completed_at_iso=completed_at_iso,
        run_id=body.run_id,
        question_id=body.question_id,
        query_id=query_id,
        usage_event_id=usage_event_id,
    )
    if benchmark_name:
        event[ledger_schema.BENCHMARK_INDEX_PK] = ledger_schema.benchmark_pk(
            benchmark_name, shard
        )
        event[ledger_schema.BENCHMARK_INDEX_SK] = dimension_sk
    if agent_name:
        event[ledger_schema.AGENT_INDEX_PK] = ledger_schema.agent_pk(agent_name, shard)
        event[ledger_schema.AGENT_INDEX_SK] = dimension_sk
    return event


def _usage_ledger_mode() -> UsageLedgerMode:
    raw_mode = os.environ.get("GATEWAY_USAGE_LEDGER_MODE", "disabled").strip().lower()
    if raw_mode in {"disabled", "off", "false", "0", ""}:
        return "disabled"
    if raw_mode == "shadow":
        return "shadow"
    if raw_mode in {"enforced", "true", "1", "on"}:
        return "enforced"
    raise ValueError("GATEWAY_USAGE_LEDGER_MODE must be disabled, shadow, or enforced")


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_event_id() -> str:
    return f"usg_{uuid.uuid4().hex[:24]}"


def _shard(usage_event_id: str) -> str:
    shard_count = _shard_count()
    value = int(hashlib.sha256(usage_event_id.encode()).hexdigest()[:8], 16)
    return f"{value % shard_count:02d}"


def _shard_count() -> int:
    try:
        shard_count = int(
            os.environ.get("GATEWAY_USAGE_LEDGER_SHARDS", DEFAULT_SHARD_COUNT)
        )
    except ValueError:
        return DEFAULT_SHARD_COUNT
    if shard_count <= 0:
        return DEFAULT_SHARD_COUNT
    return shard_count


def _max_pool_connections() -> int:
    try:
        max_pool_connections = int(
            os.environ.get(
                "GATEWAY_USAGE_LEDGER_MAX_POOL_CONNECTIONS",
                DEFAULT_MAX_POOL_CONNECTIONS,
            )
        )
    except ValueError:
        return DEFAULT_MAX_POOL_CONNECTIONS
    if max_pool_connections <= 0:
        return DEFAULT_MAX_POOL_CONNECTIONS
    return max_pool_connections


def _bounded_json(
    value: object, *, max_length: int = MAX_LEDGER_JSON_LENGTH
) -> tuple[str, bool]:
    sanitized = _sanitize_for_ledger(value)
    try:
        payload = json.dumps(
            sanitized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        payload = json.dumps(str(sanitized), separators=(",", ":"))
    if len(payload) <= max_length:
        return payload, False
    summary = _json_summary(sanitized)
    return json.dumps(summary, sort_keys=True, separators=(",", ":")), True


def _json_summary(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            "truncated": True,
            "top_level_keys": sorted(str(key) for key in mapping),
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return {"truncated": True, "item_count": len(items)}
    return {"truncated": True, "type": type(value).__name__}


def _sanitize_for_ledger(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        sanitized: dict[str, object] = {}
        for key, item in sorted(mapping.items(), key=lambda entry: str(entry[0])):
            text_key = str(key)
            if _should_redact_ledger_key(text_key):
                sanitized[text_key] = "<redacted>"
            else:
                sanitized[text_key] = _sanitize_for_ledger(item)
        return sanitized
    if isinstance(value, list | tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return [_sanitize_for_ledger(item) for item in items]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _sanitize_for_ledger(model_dump(mode="json"))
    return str(value)


def _should_redact_ledger_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return (
        telemetry.is_sensitive_key(key)
        or lowered in telemetry.CONTENT_BEARING_CONFIG_KEYS
    )


def _metadata_snapshot(metadata: QueryResultMetadata) -> dict[str, object]:
    """Return the allowlisted metadata fields safe for the durable ledger."""
    snapshot = _metadata_counts(metadata)
    if metadata.cost is not None:
        snapshot["cost"] = metadata.cost.model_dump(mode="json")

    token_metadata = metadata.extra.get("token_metadata")
    if isinstance(token_metadata, Mapping):
        token_metadata_mapping = cast(Mapping[object, object], token_metadata)
        safe_token_metadata = {
            str(key): value
            for key, value in token_metadata_mapping.items()
            if str(key)
            in {
                "estimated",
                "estimated_with_dynamic_ratio",
                "actual",
                "difference",
                "ratio",
                "dynamic_ratio_used",
            }
            and isinstance(value, bool | int | float)
        }
        if safe_token_metadata:
            snapshot["token_metadata"] = safe_token_metadata
    return snapshot


def _metadata_counts(metadata: QueryResultMetadata) -> dict[str, object]:
    cost_total = metadata.cost.total if metadata.cost is not None else None
    return {
        "input_tokens": metadata.in_tokens,
        "output_tokens": metadata.out_tokens,
        "reasoning_tokens": metadata.reasoning_tokens or 0,
        "cache_read_tokens": metadata.cache_read_tokens or 0,
        "cache_write_tokens": metadata.cache_write_tokens or 0,
        "total_input_tokens": metadata.total_input_tokens,
        "total_output_tokens": metadata.total_output_tokens,
        "duration_seconds": metadata.duration_seconds,
        "cost_usd": _decimal_or_zero(cost_total),
    }


def _decimal_or_zero(value: float | int | None) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))
