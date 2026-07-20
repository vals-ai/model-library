"""Redshift usage ledger load SQL and staging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger.redshift_transform import RedshiftUsageRows

_FACT_COLUMNS = redshift_schema.USAGE_EVENT_FACT_COLUMN_NAMES
_PERFORMANCE_COLUMNS = redshift_schema.USAGE_EVENT_PERFORMANCE_COLUMN_NAMES
_STAGING_COPY_COLUMNS = redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES


@dataclass(frozen=True)
class RedshiftWindowLoad:
    aggregate_start: datetime
    aggregate_end: datetime
    manifest_s3_uri: str
    batch_ids: tuple[str, ...]


def staging_row_from_redshift_rows(rows: RedshiftUsageRows) -> dict[str, object]:
    row = dict(rows.fact)
    row.update(
        {
            key: _json_bytes(value) if key == "performance" else value
            for key, value in rows.performance.items()
            if key != "loaded_at"
        }
    )
    return row


def copy_usage_events_staging_sql(
    *,
    manifest_uri: str,
    iam_role_arn: str,
    schema: str = redshift_schema.WAREHOUSE_SCHEMA,
) -> str:
    table = redshift_schema.qualified_table_name(
        redshift_schema.USAGE_EVENTS_STAGING_TABLE_NAME, schema
    )
    columns = _select_column_list(_STAGING_COPY_COLUMNS, spaces=2, expressions={})
    return f"""copy {table} (
{columns}
)
from {redshift_schema.sql_literal(manifest_uri)}
iam_role {redshift_schema.sql_literal(iam_role_arn)}
format as parquet
manifest;"""


def prepare_window_statements(
    load: RedshiftWindowLoad,
    *,
    schema: str,
) -> list[str]:
    staging = redshift_schema.qualified_table_name(
        redshift_schema.USAGE_EVENTS_STAGING_TABLE_NAME, schema
    )
    return [f"delete from {staging}\nwhere batch_id in ({_batch_id_list(load)});"]


def copy_window_statements(
    load: RedshiftWindowLoad,
    *,
    iam_role_arn: str,
    schema: str,
) -> list[str]:
    return [
        copy_usage_events_staging_sql(
            manifest_uri=load.manifest_s3_uri,
            iam_role_arn=iam_role_arn,
            schema=schema,
        )
    ]


def finalize_window_statements(
    load: RedshiftWindowLoad,
    *,
    schema: str,
    refresh_analytics: bool = True,
) -> list[str]:
    batch_ids = _batch_id_list(load)
    staging = redshift_schema.qualified_table_name(
        redshift_schema.USAGE_EVENTS_STAGING_TABLE_NAME, schema
    )
    analytics = (
        analytics_refresh_statements(
            start=load.aggregate_start,
            end=load.aggregate_end,
            schema=schema,
        )
        if refresh_analytics
        else []
    )
    return [
        merge_usage_events_for_batch_ids_sql(batch_ids, schema),
        merge_usage_event_performance_for_batch_ids_sql(batch_ids, schema),
        *analytics,
        f"delete from {staging}\nwhere batch_id in ({batch_ids});",
    ]


def analytics_refresh_statements(
    *,
    start: datetime,
    end: datetime,
    schema: str,
) -> list[str]:
    data_through = _timestamp_literal(end)
    return [
        *(
            statement
            for grain in redshift_schema.AGGREGATE_GRAINS
            for statement in _aggregate_refresh_statements(
                grain,
                schema,
                start_expression=_timestamp_literal(
                    _aggregate_refresh_start(grain, start)
                ),
                end_expression=_timestamp_literal(
                    _aggregate_refresh_end(grain, start, end)
                ),
                data_through_expression=data_through,
            )
        ),
        *usage_dimension_values_refresh_statements(schema),
    ]


def merge_usage_events_for_batch_ids_sql(
    batch_ids: str,
    schema: str = redshift_schema.WAREHOUSE_SCHEMA,
) -> str:
    return _merge_usage_events_sql(schema, batch_predicate=f"batch_id in ({batch_ids})")


def merge_usage_event_performance_for_batch_ids_sql(
    batch_ids: str,
    schema: str = redshift_schema.WAREHOUSE_SCHEMA,
) -> str:
    return _merge_usage_event_performance_sql(
        schema, batch_predicate=f"batch_id in ({batch_ids})"
    )


def usage_dimension_values_refresh_statements(
    schema: str = redshift_schema.WAREHOUSE_SCHEMA,
) -> tuple[str, str]:
    dimension_values = redshift_schema.qualified_table_name(
        "usage_dimension_values", schema
    )
    facts = redshift_schema.qualified_table_name("usage_events", schema)
    selects = "\n  union all\n".join(
        f"""  select {redshift_schema.sql_literal(dimension.dimension_kind)} as dimension_kind,
         cast({dimension.fact_column} as varchar(2048)) as dimension_value,
         completed_at,
         cost_usd
  from {facts}
  where {dimension.fact_column} is not null"""
        for dimension in redshift_schema.ANALYTICS_DIMENSIONS
        if dimension.is_filterable or dimension.is_groupable
    )
    return (
        f"delete from {dimension_values};",
        f"""insert into {dimension_values} (
  dimension_kind,
  dimension_value,
  first_seen_at,
  last_seen_at,
  request_count_7d,
  request_count_30d,
  cost_usd_30d,
  updated_at
)
select
  dimension_kind,
  dimension_value,
  min(completed_at) as first_seen_at,
  max(completed_at) as last_seen_at,
  sum(case when completed_at >= dateadd(day, -7, getdate()) then 1 else 0 end) as request_count_7d,
  sum(case when completed_at >= dateadd(day, -30, getdate()) then 1 else 0 end) as request_count_30d,
  coalesce(sum(case when completed_at >= dateadd(day, -30, getdate()) then cost_usd else 0 end), 0) as cost_usd_30d,
  getdate() as updated_at
from (
{selects}
) as dimension_source
where dimension_value is not null
  and dimension_value <> {redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)}
group by dimension_kind, dimension_value;""",
    )


def _batch_id_list(load: RedshiftWindowLoad) -> str:
    return ", ".join(
        redshift_schema.sql_literal(batch_id) for batch_id in load.batch_ids
    )


def _merge_usage_events_sql(schema: str, *, batch_predicate: str) -> str:
    target_table_name = "usage_events"
    target = redshift_schema.qualified_table_name(target_table_name, schema)
    source = _batch_source_sql(schema, _FACT_COLUMNS, batch_predicate)
    update_assignments = _merge_update_assignments(_FACT_COLUMNS)
    insert_columns = _column_list(_FACT_COLUMNS)
    insert_values = _source_column_list(_FACT_COLUMNS)
    return f"""merge into {target}
using (
{source}
) as source
on {target_table_name}.usage_event_id = source.usage_event_id
when matched then update set
{update_assignments}
when not matched then insert (
{insert_columns}
) values (
{insert_values}
);"""


def _merge_usage_event_performance_sql(schema: str, *, batch_predicate: str) -> str:
    target_table_name = redshift_schema.USAGE_EVENT_PERFORMANCE_TABLE_NAME
    target = redshift_schema.qualified_table_name(target_table_name, schema)
    source = _batch_source_sql(
        schema,
        _PERFORMANCE_COLUMNS,
        batch_predicate,
    )
    update_assignments = _merge_update_assignments(_PERFORMANCE_COLUMNS)
    insert_columns = _column_list(_PERFORMANCE_COLUMNS)
    insert_values = _source_column_list(_PERFORMANCE_COLUMNS)
    return f"""merge into {target}
using (
{source}
) as source
on {target_table_name}.usage_event_id = source.usage_event_id
when matched then update set
{update_assignments}
when not matched then insert (
{insert_columns}
) values (
{insert_values}
);"""


def _batch_source_sql(
    schema: str,
    columns: tuple[str, ...],
    batch_predicate: str,
    *,
    expressions: dict[str, str] | None = None,
) -> str:
    staging = redshift_schema.qualified_table_name(
        redshift_schema.USAGE_EVENTS_STAGING_TABLE_NAME, schema
    )
    selected_columns = _select_column_list(
        columns, spaces=2, expressions=expressions or {}
    )
    return f"""  select
{selected_columns}
  from {staging}
  where {batch_predicate}"""


def _merge_update_assignments(columns: tuple[str, ...]) -> str:
    mutable_columns = tuple(column for column in columns if column != "usage_event_id")
    return ",\n".join(f"  {column} = source.{column}" for column in mutable_columns)


def _json_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _select_column_list(
    columns: tuple[str, ...],
    *,
    spaces: int = 2,
    expressions: dict[str, str],
) -> str:
    prefix = " " * spaces
    return ",\n".join(
        f"{prefix}{expressions[column]} as {column}"
        if column in expressions
        else f"{prefix}{column}"
        for column in columns
    )


def _column_list(columns: tuple[str, ...], *, spaces: int = 2) -> str:
    prefix = " " * spaces
    return ",\n".join(f"{prefix}{column}" for column in columns)


def _source_column_list(columns: tuple[str, ...]) -> str:
    return ",\n".join(f"  source.{column}" for column in columns)


def _aggregate_refresh_statements(
    grain: redshift_schema.AggregateGrain,
    schema: str,
    *,
    start_expression: str,
    end_expression: str,
    data_through_expression: str,
) -> tuple[str, str, str]:
    return (
        redshift_schema.aggregate_reserved_dimension_check_sql(
            schema,
            start_parameter=start_expression,
            end_parameter=end_expression,
        ),
        redshift_schema.aggregate_delete_sql(
            grain,
            schema,
            start_parameter=start_expression,
            end_parameter=end_expression,
        ),
        redshift_schema.aggregate_insert_sql(
            grain,
            schema,
            start_parameter=start_expression,
            end_parameter=end_expression,
            data_through_expression=data_through_expression,
        ),
    )


def _aggregate_refresh_start(
    grain: redshift_schema.AggregateGrain, start: datetime
) -> datetime:
    if grain == "5m":
        return start
    if grain == "1h":
        return start.replace(minute=0, second=0, microsecond=0)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def _aggregate_refresh_end(
    grain: redshift_schema.AggregateGrain, start: datetime, end: datetime
) -> datetime:
    if grain == "5m":
        return end
    if grain == "1h":
        floor = start.replace(minute=0, second=0, microsecond=0)
        if end <= floor + timedelta(hours=1):
            return floor + timedelta(hours=1)
        end_floor = end.replace(minute=0, second=0, microsecond=0)
        return end_floor if end == end_floor else end_floor + timedelta(hours=1)
    floor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if end <= floor + timedelta(days=1):
        return floor + timedelta(days=1)
    end_floor = end.replace(hour=0, minute=0, second=0, microsecond=0)
    return end_floor if end == end_floor else end_floor + timedelta(days=1)


def _timestamp_literal(value: datetime) -> str:
    return redshift_schema.sql_literal(
        value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
