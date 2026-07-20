"""SQS message codec for gateway usage ledger events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, cast

import model_gateway.usage_ledger.schema as ledger_schema

MESSAGE_SCHEMA_VERSION: Final = 1
MAX_USAGE_LEDGER_MESSAGE_BYTES: Final = 300_000


class UsageLedgerMessageTooLarge(ValueError):
    """Raised when a usage ledger event cannot fit in a safe SQS body."""


@dataclass(frozen=True)
class PreparedUsageEvent:
    event: dict[str, object]
    message: str


def prepare_usage_event_message(
    event: Mapping[str, object],
    *,
    max_bytes: int = MAX_USAGE_LEDGER_MESSAGE_BYTES,
) -> PreparedUsageEvent:
    """Prepare the canonical event shared by direct and SQS ledger paths."""
    message = _serialize_envelope(event)
    byte_length = len(message.encode("utf-8"))
    if byte_length > max_bytes:
        raise UsageLedgerMessageTooLarge(
            f"Usage ledger message is {byte_length} bytes; max is {max_bytes} bytes"
        )
    return PreparedUsageEvent(
        event=deserialize_usage_event_message(message), message=message
    )


def serialize_usage_event_message(
    event: Mapping[str, object],
    *,
    max_bytes: int = MAX_USAGE_LEDGER_MESSAGE_BYTES,
) -> str:
    """Return a versioned JSON SQS message body for a usage ledger event."""
    return prepare_usage_event_message(event, max_bytes=max_bytes).message


def deserialize_usage_event_message(message: str) -> dict[str, object]:
    """Parse a versioned usage ledger SQS message body."""
    envelope = cast(dict[str, object], json.loads(message))
    if envelope["schema_version"] != MESSAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported usage ledger message schema_version")
    return _restore_event_types(cast(dict[str, object], envelope["event"]))


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
            restored[field] = Decimal(value)
    return restored
