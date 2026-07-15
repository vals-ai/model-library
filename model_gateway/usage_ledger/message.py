"""SQS message codec for gateway usage ledger events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal, DecimalException
from typing import Final, cast

import model_gateway.usage_ledger.schema as ledger_schema

MESSAGE_SCHEMA_VERSION: Final = 1
MAX_USAGE_LEDGER_MESSAGE_BYTES: Final = 240_000


class UsageLedgerMessageTooLarge(ValueError):
    """Raised when a usage ledger event cannot fit in a safe SQS body."""


class UsageLedgerMessageError(ValueError):
    """Raised when a usage ledger message envelope is invalid."""


def serialize_usage_event_message(
    event: Mapping[str, object],
    *,
    max_bytes: int = MAX_USAGE_LEDGER_MESSAGE_BYTES,
) -> str:
    """Return a versioned JSON SQS message body for a usage ledger event."""
    message = _serialize_envelope(event)
    if len(message.encode("utf-8")) <= max_bytes:
        return message

    queue_event = _summarize_bulky_fields(event)
    message = _serialize_envelope(queue_event)
    byte_length = len(message.encode("utf-8"))
    if byte_length > max_bytes:
        raise UsageLedgerMessageTooLarge(
            f"Usage ledger message is {byte_length} bytes; max is {max_bytes} bytes"
        )
    return message


def deserialize_usage_event_message(message: str) -> dict[str, object]:
    """Parse a versioned usage ledger SQS message body."""
    try:
        envelope = json.loads(message, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise UsageLedgerMessageError("Usage ledger message is not valid JSON") from exc

    if not isinstance(envelope, dict):
        raise UsageLedgerMessageError("Usage ledger message envelope must be an object")
    envelope_mapping = cast(Mapping[str, object], envelope)
    if envelope_mapping.get("schema_version") != MESSAGE_SCHEMA_VERSION:
        raise UsageLedgerMessageError("Unsupported usage ledger message schema_version")
    event = envelope_mapping.get("event")
    if not isinstance(event, dict):
        raise UsageLedgerMessageError("Usage ledger message event must be an object")
    return _restore_event_types(cast(dict[str, object], event))


def _serialize_envelope(event: Mapping[str, object]) -> str:
    envelope = {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "event": _message_json_safe(event),
    }
    return json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _summarize_bulky_fields(event: Mapping[str, object]) -> dict[str, object]:
    summarized = dict(event)
    for field in ledger_schema.BULKY_LEDGER_FIELDS:
        value = summarized.get(field)
        if value is None:
            continue
        summary = {
            "truncated": True,
            "reason": "sqs_message_size",
            "field": field,
            "original_bytes": len(str(value).encode("utf-8")),
        }
        summarized[field] = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        summarized[f"{field}_truncated"] = True
    return summarized


def _message_json_safe(value: object, *, key_name: str | None = None) -> object:
    if isinstance(value, Decimal):
        if key_name == ledger_schema.COST_USD_FIELD:
            return ledger_schema.format_cost_usd_decimal(value)
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _message_json_safe(item, key_name=str(key))
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_message_json_safe(item, key_name=key_name) for item in items]
    return value


def _restore_event_types(event: dict[str, object]) -> dict[str, object]:
    restored = dict(event)
    for field in ledger_schema.NUMBER_FIELDS:
        value = restored.get(field)
        if isinstance(value, str):
            try:
                numeric_value = Decimal(value)
            except DecimalException as exc:
                raise UsageLedgerMessageError(
                    f"Usage ledger message has invalid numeric field: {field}"
                ) from exc
            if not numeric_value.is_finite():
                raise UsageLedgerMessageError(
                    f"Usage ledger message has invalid numeric field: {field}"
                )
            restored[field] = numeric_value
    return restored
