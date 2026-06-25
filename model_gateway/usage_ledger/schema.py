"""Canonical DynamoDB usage ledger schema names.

Keep this module dependency-free so runtime code and local scripts can share the
same persisted attribute names without importing gateway services.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

DEFAULT_SHARD_COUNT: Final = 16
MAX_DIMENSION_VALUE_BYTES: Final = 512

BASE_PK: Final = "PK"
BASE_SK: Final = "SK"

RUN_INDEX_NAME: Final = "GSI1"
RUN_INDEX_PK: Final = "GSI1PK"
RUN_INDEX_SK: Final = "GSI1SK"

QUERY_INDEX_NAME: Final = "GSI2"
QUERY_INDEX_PK: Final = "GSI2PK"
QUERY_INDEX_SK: Final = "GSI2SK"

API_KEY_DAY_INDEX_NAME: Final = "GSI3"
API_KEY_DAY_INDEX_PK: Final = "GSI3PK"
API_KEY_DAY_INDEX_SK: Final = "GSI3SK"

BENCHMARK_INDEX_NAME: Final = "GSI4"
BENCHMARK_INDEX_PK: Final = "GSI4PK"
BENCHMARK_INDEX_SK: Final = "GSI4SK"

AGENT_INDEX_NAME: Final = "GSI5"
AGENT_INDEX_PK: Final = "GSI5PK"
AGENT_INDEX_SK: Final = "GSI5SK"

IDENTITY_BENCHMARK_NAME: Final = "benchmark_name"
IDENTITY_AGENT_NAME: Final = "agent_name"

REQUEST_COUNT_FIELD: Final = "request_count"
NUMERIC_LEDGER_FIELDS: Final = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "duration_seconds",
    "cost_usd",
)
NUMBER_FIELDS: Final = (REQUEST_COUNT_FIELD, *NUMERIC_LEDGER_FIELDS)

COMMON_PROJECTION_FIELDS: Final = (
    "usage_event_id",
    "completed_at",
    "run_id",
    "question_id",
    "query_id",
    "api_key_fingerprint",
    "model",
    "provider",
)

INDEX_PROJECTION_FIELDS: Final = (
    *COMMON_PROJECTION_FIELDS,
    "input_tokens",
    "output_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "cost_usd",
)

DIMENSION_PROJECTION_FIELDS: Final = (
    "usage_event_id",
    "completed_at",
    "run_id",
    "question_id",
    "query_id",
    "model",
    "provider",
    "provider_endpoint",
    "param_group",
    "config_hash",
    *NUMERIC_LEDGER_FIELDS,
)

BENCHMARK_PROJECTION_FIELDS: Final = (
    *DIMENSION_PROJECTION_FIELDS,
    IDENTITY_AGENT_NAME,
)

AGENT_PROJECTION_FIELDS: Final = (
    *DIMENSION_PROJECTION_FIELDS,
    IDENTITY_BENCHMARK_NAME,
)

BULKY_LEDGER_FIELDS: Final = (
    "config_redacted_json",
    "metadata_json",
    "finish_reason_json",
)


def format_shard(shard: int | str) -> str:
    if isinstance(shard, int):
        return f"{shard:02d}"
    return shard


def usage_day_pk(day: str, shard: int | str) -> str:
    return f"USAGE#DAY#{day}#S#{format_shard(shard)}"


def run_pk(run_id: str, shard: int | str) -> str:
    return f"RUN#{run_id}#S#{format_shard(shard)}"


def query_pk(query_id: str) -> str:
    return f"QUERY#{query_id}"


def api_key_day_pk(fingerprint: str, day: str, shard: int | str) -> str:
    return f"KEY#{fingerprint}#DAY#{day}#S#{format_shard(shard)}"


def benchmark_pk(benchmark_name: str, shard: int | str) -> str:
    return f"BENCHMARK#{benchmark_name}#S#{format_shard(shard)}"


def agent_pk(agent_name: str, shard: int | str) -> str:
    return f"AGENT#{agent_name}#S#{format_shard(shard)}"


def dimension_sort_key(
    *,
    completed_at_iso: str,
    run_id: str | None,
    question_id: str | None,
    query_id: str,
    usage_event_id: str,
) -> str:
    return (
        f"TS#{completed_at_iso}#RUN#{run_id or 'none'}"
        f"#QUESTION#{question_id or 'none'}#QUERY#{query_id}#USG#{usage_event_id}"
    )


def identity_dimension_value(
    identity: object,
    key: str,
) -> str | None:
    if not isinstance(identity, Mapping):
        return None
    identity_mapping = cast(Mapping[object, object], identity)
    value = identity_mapping.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized.encode("utf-8")) > MAX_DIMENSION_VALUE_BYTES:
        return None
    return normalized
