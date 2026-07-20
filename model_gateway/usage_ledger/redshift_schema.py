"""Canonical Redshift usage analytics schema definitions.

This module is dependency-free so infrastructure, local load scripts, tests, and
operator tooling can share the same warehouse contract.
"""

from __future__ import annotations

import re
from typing import Final, Literal, NamedTuple

WAREHOUSE_SCHEMA: Final = "gateway_usage"
USAGE_EVENTS_STAGING_TABLE_NAME: Final = "usage_events_staging_v2"
USAGE_EVENT_PERFORMANCE_TABLE_NAME: Final = "usage_event_performance_v2"
PERFORMANCE_VARBYTE_MAX_BYTES: Final = 1_048_576
MISSING_DIMENSION_VALUE: Final = "__missing__"
RESERVED_DIMENSION_VALUES: Final[tuple[str, ...]] = (MISSING_DIMENSION_VALUE,)

_SQL_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_BIND_PARAMETER_PATTERN: Final = re.compile(r"^:[A-Za-z_][A-Za-z0-9_]*$")
_SQL_LITERAL_PATTERN: Final = re.compile(r"^'(?:''|[^'])*'$")
_ALLOWED_SQL_EXPRESSIONS: Final[frozenset[str]] = frozenset({"getdate()"})

AggregateGrain = Literal["5m", "1h", "1d"]

AGGREGATE_GRAINS: Final[tuple[AggregateGrain, ...]] = ("5m", "1h", "1d")
USAGE_EVENT_FACT_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "usage_event_id",
    "completed_at",
    "completed_date",
    "completed_hour",
    "run_id",
    "question_id",
    "query_id",
    "query_id_normalized",
    "requested_model_key",
    "provider",
    "provider_endpoint",
    "param_group",
    "config_hash",
    "benchmark_name",
    "agent_name",
    "identity_email",
    "api_key_fingerprint",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "duration_seconds",
    "cost_usd",
    "finish_reason",
    "finish_reason_raw",
    "schema_version",
    "metadata_schema_version",
    "normalization_version",
    "usage_shard",
    "source_pk",
    "source_sk",
    "loaded_at",
)
USAGE_EVENT_PERFORMANCE_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "usage_event_id",
    "performance",
    "performance_truncated",
    "loaded_at",
)
USAGE_EVENTS_STAGING_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "batch_id",
    *USAGE_EVENT_FACT_COLUMN_NAMES,
    *(
        column
        for column in USAGE_EVENT_PERFORMANCE_COLUMN_NAMES
        if column not in {"usage_event_id", "loaded_at"}
    ),
)

AGGREGATE_TABLE_BY_GRAIN: Final[dict[AggregateGrain, str]] = {
    "5m": "usage_agg_5m",
    "1h": "usage_agg_1h",
    "1d": "usage_agg_1d",
}


class AnalyticsDimension(NamedTuple):
    dimension_kind: str
    source_column: str
    fact_column: str
    aggregate_column_type: str
    is_filterable: bool
    is_groupable: bool
    is_in_fast_agg: bool
    display_name: str
    sort_order: int


ANALYTICS_DIMENSIONS: Final[tuple[AnalyticsDimension, ...]] = (
    AnalyticsDimension(
        "provider",
        "provider",
        "provider",
        "varchar(256)",
        True,
        True,
        True,
        "Provider",
        10,
    ),
    AnalyticsDimension(
        "provider_model",
        "provider_model",
        "requested_model_key",
        "varchar(2048)",
        True,
        True,
        True,
        "Provider model",
        20,
    ),
    AnalyticsDimension(
        "benchmark",
        "benchmark_name",
        "benchmark_name",
        "varchar(1024)",
        True,
        True,
        True,
        "Benchmark",
        30,
    ),
    AnalyticsDimension(
        "agent",
        "agent_name",
        "agent_name",
        "varchar(1024)",
        True,
        True,
        True,
        "Agent",
        40,
    ),
    AnalyticsDimension(
        "email",
        "identity_email",
        "identity_email",
        "varchar(1024)",
        True,
        True,
        True,
        "Email",
        50,
    ),
    AnalyticsDimension(
        "api_key_fingerprint",
        "api_key_fingerprint",
        "api_key_fingerprint",
        "varchar(512)",
        True,
        True,
        True,
        "API key fingerprint",
        60,
    ),
)

FAST_AGGREGATE_DIMENSIONS: Final[tuple[str, ...]] = tuple(
    dimension.dimension_kind
    for dimension in ANALYTICS_DIMENSIONS
    if dimension.is_in_fast_agg
)


def _validate_sql_identifier(identifier: str) -> str:
    if not _SQL_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def _validate_sql_expression(expression: str) -> str:
    if _SQL_BIND_PARAMETER_PATTERN.fullmatch(expression):
        return expression
    if _SQL_LITERAL_PATTERN.fullmatch(expression):
        return expression
    if expression in _ALLOWED_SQL_EXPRESSIONS:
        return expression
    raise ValueError(f"Unsafe SQL expression: {expression!r}")


def is_reserved_dimension_value(value: str) -> bool:
    return value in RESERVED_DIMENSION_VALUES


def qualified_table_name(table_name: str, schema: str = WAREHOUSE_SCHEMA) -> str:
    return f"{_validate_sql_identifier(schema)}.{_validate_sql_identifier(table_name)}"


def aggregate_table_name(grain: AggregateGrain) -> str:
    return AGGREGATE_TABLE_BY_GRAIN[grain]


def sql_literal(value: str | bool | int) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def _fast_aggregate_dimensions() -> tuple[AnalyticsDimension, ...]:
    return tuple(
        dimension for dimension in ANALYTICS_DIMENSIONS if dimension.is_in_fast_agg
    )


def _indent(lines: list[str], spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in lines)


def create_schema_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    return f"create schema if not exists {_validate_sql_identifier(schema)};"


def usage_events_column_sql(*, include_batch_id: bool = False) -> str:
    batch_column = "  batch_id varchar(128) not null,\n" if include_batch_id else ""
    return f"""{batch_column}  usage_event_id varchar(128) not null,
  completed_at timestamptz not null,
  completed_date date not null,
  completed_hour timestamp not null,

  run_id varchar(65535),
  question_id varchar(65535),
  query_id varchar(65535),
  query_id_normalized varchar(65535),

  requested_model_key varchar(2048) not null,
  provider varchar(256) not null,
  provider_endpoint varchar(64) not null,
  param_group varchar(512),
  config_hash varchar(256),

  benchmark_name varchar(1024),
  agent_name varchar(1024),
  identity_email varchar(1024),
  api_key_fingerprint varchar(512),

  input_tokens bigint not null default 0,
  output_tokens bigint not null default 0,
  reasoning_tokens bigint not null default 0,
  cache_read_tokens bigint not null default 0,
  cache_write_tokens bigint not null default 0,
  total_input_tokens bigint not null default 0,
  total_output_tokens bigint not null default 0,
  duration_seconds decimal(18,6),
  cost_usd decimal(38,12) not null default 0,

  finish_reason varchar(256),
  finish_reason_raw varchar(2048),

  schema_version integer not null,
  metadata_schema_version integer not null,
  normalization_version varchar(64) not null,
  usage_shard varchar(8),
  source_pk varchar(2048),
  source_sk varchar(2048),
  loaded_at timestamptz not null default getdate()"""


def usage_events_staging_performance_column_sql() -> str:
    return f"""  performance varbyte({PERFORMANCE_VARBYTE_MAX_BYTES}),
  performance_truncated boolean not null default false"""


def usage_events_staging_table_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name(USAGE_EVENTS_STAGING_TABLE_NAME, schema)
    return f"""create table if not exists {table} (
{usage_events_column_sql(include_batch_id=True)},
{usage_events_staging_performance_column_sql()}
)
diststyle auto
sortkey (batch_id, completed_at);"""


def usage_events_table_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name("usage_events", schema)
    return f"""create table if not exists {table} (
{usage_events_column_sql()}
)
diststyle auto
sortkey (completed_at);"""


def usage_event_performance_v2_table_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name(USAGE_EVENT_PERFORMANCE_TABLE_NAME, schema)
    return f"""create table if not exists {table} (
  usage_event_id varchar(128) not null,
  performance varbyte({PERFORMANCE_VARBYTE_MAX_BYTES}),
  performance_truncated boolean not null default false,
  loaded_at timestamptz not null default getdate()
)
diststyle auto
sortkey (usage_event_id);"""


def aggregate_table_ddl(grain: AggregateGrain, schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name(aggregate_table_name(grain), schema)
    dimension_columns = _indent(
        [
            f"{dimension.source_column} {dimension.aggregate_column_type} not null,"
            for dimension in _fast_aggregate_dimensions()
        ]
    )
    return f"""create table if not exists {table} (
  bucket_start_utc timestamptz not null,
  bucket_end_utc timestamptz not null,

{dimension_columns}

  request_count bigint not null,
  input_tokens bigint not null,
  output_tokens bigint not null,
  reasoning_tokens bigint not null,
  cache_read_tokens bigint not null,
  cache_write_tokens bigint not null,
  total_input_tokens bigint not null,
  total_output_tokens bigint not null,
  duration_seconds_sum decimal(38,6) not null,
  duration_observed_count bigint not null,
  cost_usd_sum decimal(38,12) not null,

  data_through_utc timestamptz not null,
  refreshed_at timestamptz not null
)
diststyle auto
sortkey (
  bucket_start_utc,
  benchmark_name,
  agent_name,
  identity_email,
  api_key_fingerprint,
  provider_model
);"""


def aggregate_table_ddl_by_grain(
    schema: str = WAREHOUSE_SCHEMA,
) -> dict[AggregateGrain, str]:
    return {grain: aggregate_table_ddl(grain, schema) for grain in AGGREGATE_GRAINS}


def usage_dimension_values_table_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name("usage_dimension_values", schema)
    return f"""create table if not exists {table} (
  dimension_kind varchar(64) not null,
  dimension_value varchar(2048) not null,
  first_seen_at timestamptz,
  last_seen_at timestamptz,
  request_count_7d bigint,
  request_count_30d bigint,
  cost_usd_30d decimal(38,12),
  updated_at timestamptz not null
)
diststyle auto
sortkey (dimension_kind, dimension_value);"""


def analytics_dimensions_table_ddl(schema: str = WAREHOUSE_SCHEMA) -> str:
    table = qualified_table_name("analytics_dimensions", schema)
    return f"""create table if not exists {table} (
  dimension_kind varchar(64) not null,
  source_column varchar(128) not null,
  is_filterable boolean not null,
  is_groupable boolean not null,
  is_in_fast_agg boolean not null,
  display_name varchar(128) not null,
  sort_order integer not null
)
diststyle auto
sortkey (sort_order, dimension_kind);"""


def _analytics_dimension_row_sql(dimension: AnalyticsDimension) -> str:
    values: tuple[str | bool | int, ...] = (
        dimension.dimension_kind,
        dimension.source_column,
        dimension.is_filterable,
        dimension.is_groupable,
        dimension.is_in_fast_agg,
        dimension.display_name,
        dimension.sort_order,
    )
    return "(" + ", ".join(sql_literal(value) for value in values) + ")"


def analytics_dimensions_seed_statements(
    schema: str = WAREHOUSE_SCHEMA,
) -> tuple[str, str]:
    table = qualified_table_name("analytics_dimensions", schema)
    values = ",\n".join(
        f"  {_analytics_dimension_row_sql(dimension)}"
        for dimension in ANALYTICS_DIMENSIONS
    )
    return (
        f"delete from {table};",
        f"""insert into {table} (
  dimension_kind,
  source_column,
  is_filterable,
  is_groupable,
  is_in_fast_agg,
  display_name,
  sort_order
) values
{values};""",
    )


def aggregate_delete_sql(
    grain: AggregateGrain,
    schema: str = WAREHOUSE_SCHEMA,
    *,
    start_parameter: str = ":start_bucket",
    end_parameter: str = ":end_bucket",
) -> str:
    table = qualified_table_name(aggregate_table_name(grain), schema)
    start_expression = _validate_sql_expression(start_parameter)
    end_expression = _validate_sql_expression(end_parameter)
    return f"""delete from {table}
where bucket_start_utc >= {start_expression}
  and bucket_start_utc < {end_expression};"""


def _bucket_start_expression(grain: AggregateGrain) -> str:
    completed_at_timestamp = "completed_at::timestamp"
    if grain == "5m":
        return (
            f"dateadd(minute, (floor(date_part(minute, {completed_at_timestamp}) / 5) * 5)::integer, "
            f"date_trunc('hour', {completed_at_timestamp}))"
        )
    if grain == "1h":
        return f"date_trunc('hour', {completed_at_timestamp})"
    return f"date_trunc('day', {completed_at_timestamp})"


def _bucket_end_expression(grain: AggregateGrain, bucket_start_expression: str) -> str:
    if grain == "5m":
        return f"dateadd(minute, 5, {bucket_start_expression})"
    if grain == "1h":
        return f"dateadd(hour, 1, {bucket_start_expression})"
    return f"dateadd(day, 1, {bucket_start_expression})"


def _aggregate_dimension_columns() -> list[str]:
    return [dimension.source_column for dimension in _fast_aggregate_dimensions()]


def _aggregate_dimension_insert_columns() -> str:
    return _indent([f"{column}," for column in _aggregate_dimension_columns()])


def _aggregate_dimension_select_columns() -> str:
    return _indent([f"{column}," for column in _aggregate_dimension_columns()])


def _aggregate_dimension_bucket_columns(missing: str) -> str:
    return _indent(
        [
            f"coalesce({dimension.fact_column}, {missing}) as {dimension.source_column},"
            for dimension in _fast_aggregate_dimensions()
        ],
        spaces=4,
    )


def _aggregate_dimension_group_by_columns() -> str:
    columns = _aggregate_dimension_columns()
    return _indent(
        [f"{column}," for column in columns[:-1]] + [columns[-1]],
        spaces=2,
    )


def _reserved_dimension_conflict_predicate() -> str:
    reserved_values = ", ".join(
        sql_literal(value) for value in RESERVED_DIMENSION_VALUES
    )
    checks = [
        f"{dimension.fact_column} in ({reserved_values})"
        for dimension in _fast_aggregate_dimensions()
    ]
    return "\n      or ".join(checks)


def aggregate_reserved_dimension_check_sql(
    schema: str = WAREHOUSE_SCHEMA,
    *,
    start_parameter: str = ":start_bucket",
    end_parameter: str = ":end_bucket",
) -> str:
    fact_table = qualified_table_name("usage_events", schema)
    start_expression = _validate_sql_expression(start_parameter)
    end_expression = _validate_sql_expression(end_parameter)
    return f"""select count(*) as reserved_dimension_conflict_count
from {fact_table}
where completed_at >= {start_expression}
  and completed_at < {end_expression}
  and (
      {_reserved_dimension_conflict_predicate()}
  );"""


def _reserved_dimension_guard_sql(
    fact_table: str,
    start_expression: str,
    end_expression: str,
) -> str:
    return f"""select 1 / case when exists (
    select 1
    from {fact_table}
    where completed_at >= {start_expression}
      and completed_at < {end_expression}
      and (
      {_reserved_dimension_conflict_predicate()}
      )
  ) then 0 else 1 end as reserved_dimension_guard"""


def aggregate_insert_sql(
    grain: AggregateGrain,
    schema: str = WAREHOUSE_SCHEMA,
    *,
    start_parameter: str = ":start_bucket",
    end_parameter: str = ":end_bucket",
    data_through_expression: str = ":data_through_utc",
    refreshed_at_expression: str = "getdate()",
) -> str:
    aggregate_table = qualified_table_name(aggregate_table_name(grain), schema)
    fact_table = qualified_table_name("usage_events", schema)
    start_expression = _validate_sql_expression(start_parameter)
    end_expression = _validate_sql_expression(end_parameter)
    data_through_sql = _validate_sql_expression(data_through_expression)
    refreshed_at_sql = _validate_sql_expression(refreshed_at_expression)
    missing = sql_literal(MISSING_DIMENSION_VALUE)
    bucket_start = _bucket_start_expression(grain)
    bucket_end = _bucket_end_expression(grain, bucket_start)
    dimension_insert_columns = _aggregate_dimension_insert_columns()
    dimension_select_columns = _aggregate_dimension_select_columns()
    dimension_bucket_columns = _aggregate_dimension_bucket_columns(missing)
    dimension_group_by_columns = _aggregate_dimension_group_by_columns()
    reserved_dimension_guard = _reserved_dimension_guard_sql(
        fact_table, start_expression, end_expression
    )
    return f"""insert into {aggregate_table} (
  bucket_start_utc,
  bucket_end_utc,
{dimension_insert_columns}
  request_count,
  input_tokens,
  output_tokens,
  reasoning_tokens,
  cache_read_tokens,
  cache_write_tokens,
  total_input_tokens,
  total_output_tokens,
  duration_seconds_sum,
  duration_observed_count,
  cost_usd_sum,
  data_through_utc,
  refreshed_at
)
select
  bucket_start_utc,
  bucket_end_utc,
{dimension_select_columns}
  count(*) as request_count,
  coalesce(sum(input_tokens), 0) as input_tokens,
  coalesce(sum(output_tokens), 0) as output_tokens,
  coalesce(sum(reasoning_tokens), 0) as reasoning_tokens,
  coalesce(sum(cache_read_tokens), 0) as cache_read_tokens,
  coalesce(sum(cache_write_tokens), 0) as cache_write_tokens,
  coalesce(sum(total_input_tokens), 0) as total_input_tokens,
  coalesce(sum(total_output_tokens), 0) as total_output_tokens,
  coalesce(sum(duration_seconds), 0) as duration_seconds_sum,
  count(duration_seconds) as duration_observed_count,
  coalesce(sum(cost_usd), 0) as cost_usd_sum,
  {data_through_sql} as data_through_utc,
  {refreshed_at_sql} as refreshed_at
from (
  select
    {bucket_start} as bucket_start_utc,
    {bucket_end} as bucket_end_utc,
{dimension_bucket_columns}
    input_tokens,
    output_tokens,
    reasoning_tokens,
    cache_read_tokens,
    cache_write_tokens,
    total_input_tokens,
    total_output_tokens,
    duration_seconds,
    cost_usd
  from {fact_table}
  where completed_at >= {start_expression}
    and completed_at < {end_expression}
) as bucketed
cross join (
  {reserved_dimension_guard}
) as reserved_dimension_guard
group by
  bucket_start_utc,
  bucket_end_utc,
{dimension_group_by_columns};"""


def schema_statement_batches(
    schema: str = WAREHOUSE_SCHEMA,
) -> tuple[tuple[str, ...], ...]:
    aggregate_ddls = tuple(
        aggregate_table_ddl(grain, schema) for grain in AGGREGATE_GRAINS
    )
    singleton_statements = (
        create_schema_ddl(schema),
        usage_events_staging_table_ddl(schema),
        usage_events_table_ddl(schema),
        usage_event_performance_v2_table_ddl(schema),
        *aggregate_ddls,
        usage_dimension_values_table_ddl(schema),
        analytics_dimensions_table_ddl(schema),
    )
    return tuple((statement,) for statement in singleton_statements) + (
        analytics_dimensions_seed_statements(schema),
    )
