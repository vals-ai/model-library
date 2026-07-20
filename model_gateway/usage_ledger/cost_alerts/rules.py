from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from typing import Final, Literal, cast

import model_gateway.usage_ledger.redshift_schema as redshift_schema

CostAlertGrain = Literal["1h", "1d"]
CostAlertGroupBy = Literal["global", "provider_model", "benchmark", "identity_email"]
CostAlertBreakdownGroupBy = Literal[
    "provider_model",
    "benchmark",
    "agent",
    "identity_email",
]
CostAlertImageAttachTo = Literal["active", "worsening"]

_ALLOWED_GRAINS: Final[tuple[CostAlertGrain, ...]] = ("1h", "1d")
_ALLOWED_GROUP_BYS: Final[tuple[CostAlertGroupBy, ...]] = (
    "global",
    "provider_model",
    "benchmark",
    "identity_email",
)
_BREAKDOWN_GROUP_BYS: Final[tuple[CostAlertBreakdownGroupBy, ...]] = (
    "provider_model",
    "benchmark",
    "agent",
    "identity_email",
)
_ALLOWED_IMAGE_ATTACH_TO: Final[tuple[CostAlertImageAttachTo, ...]] = (
    "active",
    "worsening",
)
_SLACK_IMAGE_URL_MAX_LENGTH: Final = 3000
_SLACK_IMAGE_ALT_TEXT_MAX_LENGTH: Final = 2000
_DIMENSION_COLUMN_BY_GROUP: Final[dict[CostAlertGroupBy, str | None]] = {
    "global": None,
    "provider_model": "provider_model",
    "benchmark": "benchmark_name",
    "identity_email": "identity_email",
}
_IDLE_DIMENSION_VALUE: Final = "__idle__"
_BREAKDOWN_DIMENSION_SQL_BY_GROUP: Final[dict[CostAlertBreakdownGroupBy, str]] = {
    "provider_model": "provider_model",
    "benchmark": "benchmark_name",
    "agent": "agent_name",
    "identity_email": "identity_email",
}
_PT_TIMEZONE: Final = "America/Los_Angeles"
_TOP_CONTRIBUTORS: Final = 3


@dataclass(frozen=True)
class CostAlertImageConfig:
    url: str
    alt_text: str
    attach_to: tuple[CostAlertImageAttachTo, ...]
    min_threshold_multiple: Decimal | None = None


@dataclass(frozen=True)
class CostAlertRule:
    name: str
    grain: CostAlertGrain
    group_by: CostAlertGroupBy
    threshold_usd: Decimal
    min_previous_usd: Decimal | None = None
    percent_increase: Decimal | None = None
    include_missing: bool = False
    image: CostAlertImageConfig | None = None


@dataclass(frozen=True)
class ComparisonRow:
    bucket_start_utc: datetime
    bucket_end_utc: datetime
    data_through_utc: datetime
    watermark_utc: datetime
    dimension_value: str
    current_request_count: int
    current_eligible_request_count: int
    current_cost_usd: Decimal
    previous_cost_usd: Decimal
    absolute_increase_usd: Decimal
    percent_increase: Decimal | None


@dataclass(frozen=True)
class AlertContributor:
    display_value: str
    current_request_count: int
    current_cost_usd: Decimal
    ignored_for_cost: bool = False


@dataclass(frozen=True)
class AlertBreakdown:
    scope_value: str
    group_by: CostAlertBreakdownGroupBy
    top_contributors: tuple[AlertContributor, ...]


@dataclass(frozen=True)
class AlertCandidate:
    rule_name: str
    grain: CostAlertGrain
    group_by: CostAlertGroupBy
    scope_value: str | None
    bucket_start_utc: datetime
    bucket_end_utc: datetime
    data_through_utc: datetime
    threshold_usd: Decimal
    total_current_cost_usd: Decimal
    total_current_request_count: int
    total_current_eligible_request_count: int
    breakdowns: tuple[AlertBreakdown, ...] = ()
    image: CostAlertImageConfig | None = None


def parse_cost_alert_config_json(raw: str) -> tuple[CostAlertRule, ...]:
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("cost alert config must be valid JSON") from exc
    payload_mapping = _string_key_dict(payload, "cost alert config")
    rules_value = payload_mapping.get("rules")
    rules_list = _object_list(rules_value, "rules")
    if not rules_list:
        raise ValueError("cost alert config must include a non-empty rules list")
    rules = tuple(_parse_rule(_string_key_dict(item, "rule")) for item in rules_list)
    identities = [(rule.name, rule.grain, rule.group_by) for rule in rules]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate cost alert rule identity")
    return rules


def build_rule_query(
    rule: CostAlertRule,
    *,
    ignored_model_keys: Sequence[str] = (),
    schema: str,
) -> str:
    table_name = redshift_schema.qualified_table_name(
        _source_table_name(rule.grain),
        schema,
    )
    bucket_start_sql = _bucket_start_sql(rule.grain)
    bucket_end_sql = _bucket_end_sql(rule.grain)
    dimension_sql = _alert_dimension_sql(rule.group_by)
    dimension_filter = _dimension_filter_sql(rule)
    cost_usd_sql = _cost_usd_sql(ignored_model_keys)
    eligible_request_count_sql = _eligible_request_count_sql(ignored_model_keys)
    dimension_anchor_sql = _dimension_anchor_sql(rule)
    data_through_aggregate = (
        "max(source_data_through_utc)"
        if rule.grain == "1d"
        else "min(source_data_through_utc)"
    )
    return f"""with watermark as (
  select max(agg.data_through_utc) as watermark_utc
  from {table_name} as agg
  where agg.data_through_utc is not null
), evaluation_window as (
{_evaluation_window_sql(rule.grain)}
), bucketed_source as (
  select
    {bucket_start_sql} as alert_bucket_start_utc,
    {bucket_end_sql} as alert_bucket_end_utc,
    {dimension_sql} as alert_dimension_value,
    agg.request_count as source_request_count,
    {eligible_request_count_sql} as source_eligible_request_count,
    {cost_usd_sql} as source_cost_usd,
    agg.data_through_utc as source_data_through_utc,
    evaluation_window.watermark_utc
  from {table_name} as agg
  cross join evaluation_window
  where agg.bucket_start_utc >= evaluation_window.previous_bucket_start_utc
    and agg.bucket_start_utc < evaluation_window.bucket_end_utc
    and agg.bucket_end_utc <= evaluation_window.watermark_utc
    and agg.bucket_end_utc <= agg.data_through_utc
{dimension_filter}), complete_buckets as (
  select
    alert_bucket_start_utc as bucket_start_utc,
    alert_bucket_end_utc as bucket_end_utc,
    alert_dimension_value as dimension_value,
    sum(source_request_count) as current_request_count,
    sum(source_eligible_request_count) as current_eligible_request_count,
    sum(source_cost_usd) as current_cost_usd,
    {data_through_aggregate} as data_through_utc,
    watermark_utc
  from bucketed_source
  group by
    alert_bucket_start_utc,
    alert_bucket_end_utc,
    alert_dimension_value,
    watermark_utc
), current_window as (
  select complete_buckets.*
  from complete_buckets
  join evaluation_window
    on complete_buckets.bucket_start_utc = evaluation_window.bucket_start_utc
), previous_window as (
  select complete_buckets.*
  from complete_buckets
  join evaluation_window
    on complete_buckets.bucket_end_utc = evaluation_window.bucket_start_utc
), dimensions as (
  select dimension_value from current_window
  union
  select dimension_value from previous_window
  union
{dimension_anchor_sql}
)
select
  evaluation_window.bucket_start_utc as bucket_start_utc,
  evaluation_window.bucket_end_utc as bucket_end_utc,
  coalesce(current_window.data_through_utc, previous_window.data_through_utc, evaluation_window.watermark_utc) as data_through_utc,
  evaluation_window.watermark_utc as watermark_utc,
  dimensions.dimension_value as dimension_value,
  coalesce(current_window.current_request_count, 0) as current_request_count,
  coalesce(current_window.current_eligible_request_count, 0) as current_eligible_request_count,
  coalesce(current_window.current_cost_usd, 0) as current_cost_usd,
  coalesce(previous_window.current_cost_usd, 0) as previous_cost_usd,
  coalesce(current_window.current_cost_usd, 0) - coalesce(previous_window.current_cost_usd, 0) as absolute_increase_usd,
  case
    when coalesce(previous_window.current_cost_usd, 0) = 0 then null
    else (
      (coalesce(current_window.current_cost_usd, 0) - coalesce(previous_window.current_cost_usd, 0))
      / coalesce(previous_window.current_cost_usd, 0)
    ) * 100
  end as percent_increase
from dimensions
cross join evaluation_window
left join current_window
  on current_window.dimension_value = dimensions.dimension_value
left join previous_window
  on previous_window.dimension_value = dimensions.dimension_value
order by current_cost_usd desc, dimension_value asc;"""


def build_breakdown_query(
    rule: CostAlertRule,
    *,
    ignored_model_keys: Sequence[str] = (),
    scope_values: Sequence[str] = (),
    schema: str,
) -> str:
    if rule.group_by != "global" and not scope_values:
        raise ValueError("scoped breakdown query requires at least one scope")
    table_name = redshift_schema.qualified_table_name(
        _source_table_name(rule.grain),
        schema,
    )
    scope_sql = _alert_dimension_sql(rule.group_by)
    scope_filter = _breakdown_scope_filter_sql(rule, scope_values)
    cost_usd_sql = _cost_usd_sql(ignored_model_keys)
    ignored_for_cost_sql = _ignored_for_cost_sql(ignored_model_keys)
    current_breakdowns = _breakdown_aggregate_sql("bucketed_source")
    missing_filter = ""
    if not rule.include_missing:
        missing = redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)
        missing_filter = f"\n    and dimension_value <> {missing}"
    return f"""with watermark as (
  select max(agg.data_through_utc) as watermark_utc
  from {table_name} as agg
  where agg.data_through_utc is not null
), evaluation_window as (
{_evaluation_window_sql(rule.grain)}
), bucketed_source as (
  select
    {scope_sql} as alert_scope_value,
    {_normalized_dimension_sql("agg.provider_model")} as provider_model,
    {_normalized_dimension_sql("agg.benchmark_name")} as benchmark_name,
    {_normalized_dimension_sql("agg.agent_name")} as agent_name,
    {_normalized_dimension_sql("agg.identity_email")} as identity_email,
    agg.request_count,
    agg.cost_usd_sum as actual_cost_usd_sum,
    {cost_usd_sql} as cost_usd_sum,
    {ignored_for_cost_sql} as ignored_for_cost
  from {table_name} as agg
  cross join evaluation_window
  where agg.bucket_start_utc >= evaluation_window.bucket_start_utc
    and agg.bucket_start_utc < evaluation_window.bucket_end_utc
    and agg.bucket_end_utc <= evaluation_window.watermark_utc
    and agg.bucket_end_utc <= agg.data_through_utc
{scope_filter}), current_breakdowns as (
{current_breakdowns}
), ranked as (
  select
    *,
    row_number() over (
      partition by alert_scope_value, group_by
      order by current_cost_usd desc, dimension_value asc
    ) as group_rank
  from current_breakdowns
  where current_cost_usd > 0{missing_filter}
)
select
  alert_scope_value,
  group_by,
  dimension_value,
  current_request_count,
  current_cost_usd,
  ignored_for_cost
from ranked
where group_rank <= {_TOP_CONTRIBUTORS}
order by alert_scope_value asc, group_by asc, current_cost_usd desc, dimension_value asc;"""


def _source_table_name(grain: CostAlertGrain) -> str:
    source_grain = "1h" if grain == "1d" else grain
    return redshift_schema.aggregate_table_name(source_grain)


def _bucket_start_sql(grain: CostAlertGrain) -> str:
    if grain == "1d":
        bucket_start_timestamp = "agg.bucket_start_utc::timestamp"
        local_day = (
            "date_trunc('day', "
            f"convert_timezone('UTC', '{_PT_TIMEZONE}', {bucket_start_timestamp}))"
        )
        return f"convert_timezone('{_PT_TIMEZONE}', 'UTC', {local_day})"
    return "agg.bucket_start_utc"


def _bucket_end_sql(grain: CostAlertGrain) -> str:
    if grain == "1d":
        bucket_start_timestamp = "agg.bucket_start_utc::timestamp"
        local_day = (
            "date_trunc('day', "
            f"convert_timezone('UTC', '{_PT_TIMEZONE}', {bucket_start_timestamp}))"
        )
        return (
            f"convert_timezone('{_PT_TIMEZONE}', 'UTC', dateadd(day, 1, {local_day}))"
        )
    return "agg.bucket_end_utc"


def _evaluation_window_sql(grain: CostAlertGrain) -> str:
    if grain == "1d":
        watermark_timestamp = "watermark_utc::timestamp"
        local_day = (
            "date_trunc('day', "
            f"convert_timezone('UTC', '{_PT_TIMEZONE}', {watermark_timestamp}))"
        )
        previous_bucket_start = (
            f"convert_timezone('{_PT_TIMEZONE}', 'UTC', dateadd(day, -1, {local_day}))"
        )
        bucket_start = f"convert_timezone('{_PT_TIMEZONE}', 'UTC', {local_day})"
        bucket_end = (
            f"convert_timezone('{_PT_TIMEZONE}', 'UTC', dateadd(day, 1, {local_day}))"
        )
    else:
        previous_bucket_start = "date_trunc('hour', watermark_utc - interval '2 hours')"
        bucket_start = "date_trunc('hour', watermark_utc - interval '1 hour')"
        bucket_end = "date_trunc('hour', watermark_utc)"
    return f"""  select
    {previous_bucket_start} as previous_bucket_start_utc,
    {bucket_start} as bucket_start_utc,
    {bucket_end} as bucket_end_utc,
    watermark_utc
  from watermark
  where watermark_utc is not null"""


def _dimension_anchor_sql(rule: CostAlertRule) -> str:
    dimension_value = "all" if rule.group_by == "global" else _IDLE_DIMENSION_VALUE
    return f"  select {redshift_schema.sql_literal(dimension_value)} as dimension_value"


def _decoded_result_rows(
    result: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    metadata = _object_list(result.get("ColumnMetadata"), "ColumnMetadata")
    records = _object_list(result.get("Records"), "Records")
    columns = tuple(_metadata_name(item) for item in metadata)
    for record_value in records:
        record = _object_list(record_value, "Redshift record")
        yield {
            column: _cell_value(cell)
            for column, cell in zip(columns, record, strict=True)
        }


def parse_comparison_rows(result: Mapping[str, object]) -> tuple[ComparisonRow, ...]:
    return tuple(
        _comparison_row_from_values(values) for values in _decoded_result_rows(result)
    )


def parse_breakdown_rows(result: Mapping[str, object]) -> tuple[AlertBreakdown, ...]:
    contributors_by_scope_group: dict[
        tuple[str, CostAlertBreakdownGroupBy], list[AlertContributor]
    ] = {}
    for values in _decoded_result_rows(result):
        key = (
            _string_value(values, "alert_scope_value"),
            _breakdown_group_value(values, "group_by"),
        )
        contributors_by_scope_group.setdefault(key, []).append(
            _contributor_from_values(values)
        )
    return tuple(
        AlertBreakdown(
            scope_value=scope_value,
            group_by=group_by,
            top_contributors=tuple(contributors),
        )
        for (scope_value, group_by), contributors in contributors_by_scope_group.items()
    )


def evaluate_rule_rows(
    rule: CostAlertRule,
    rows: Sequence[ComparisonRow],
) -> tuple[AlertCandidate, ...]:
    triggered_rows = sorted(
        (
            row
            for row in rows
            if _include_row(rule, row) and _row_matches_rule(rule, row)
        ),
        key=lambda item: (-item.current_cost_usd, item.dimension_value),
    )
    if rule.group_by == "global":
        triggered_rows = triggered_rows[:1]
    return tuple(
        _candidate_from_row(
            rule,
            row,
            scope_value=None if rule.group_by == "global" else row.dimension_value,
        )
        for row in triggered_rows
    )


def _candidate_from_row(
    rule: CostAlertRule,
    row: ComparisonRow,
    *,
    scope_value: str | None,
) -> AlertCandidate:
    return AlertCandidate(
        rule_name=rule.name,
        grain=rule.grain,
        group_by=rule.group_by,
        scope_value=scope_value,
        bucket_start_utc=row.bucket_start_utc,
        bucket_end_utc=row.bucket_end_utc,
        data_through_utc=row.data_through_utc,
        threshold_usd=rule.threshold_usd,
        total_current_cost_usd=row.current_cost_usd,
        total_current_request_count=row.current_request_count,
        total_current_eligible_request_count=row.current_eligible_request_count,
        image=rule.image,
    )


def display_dimension_value(value: str) -> str:
    if value == redshift_schema.MISSING_DIMENSION_VALUE:
        return "unattributed"
    return value


def _breakdown_aggregate_sql(source_name: str) -> str:
    queries: list[str] = []
    for group_by, column_sql in _BREAKDOWN_DIMENSION_SQL_BY_GROUP.items():
        if group_by == "provider_model":
            current_cost_sql = "sum(actual_cost_usd_sum)"
            ignored_for_cost_sql = "max(ignored_for_cost)"
        else:
            current_cost_sql = "sum(cost_usd_sum)"
            ignored_for_cost_sql = "0"
        queries.append(
            f"""  select
    alert_scope_value,
    {redshift_schema.sql_literal(group_by)} as group_by,
    {column_sql} as dimension_value,
    sum(request_count) as current_request_count,
    {current_cost_sql} as current_cost_usd,
    {ignored_for_cost_sql} as ignored_for_cost
  from {source_name}
  group by
    alert_scope_value,
    dimension_value"""
        )
    return "\n  union all\n".join(queries)


def _parse_rule(raw: Mapping[str, object]) -> CostAlertRule:
    name = _required_string(raw, "name")
    grain = _parse_grain(_required_string(raw, "grain"))
    group_by = _parse_group_by(_required_string(raw, "group_by"))
    threshold_usd = _decimal_from_object(raw.get("threshold_usd"), "threshold_usd")
    min_previous_value = raw.get("min_previous_usd")
    percent_value = raw.get("percent_increase")
    min_previous_usd = (
        None
        if min_previous_value is None
        else _decimal_from_object(min_previous_value, "min_previous_usd")
    )
    percent_increase = (
        None
        if percent_value is None
        else _decimal_from_object(percent_value, "percent_increase")
    )
    include_missing = _bool_with_default(raw, "include_missing", False)
    image = _parse_image_config(raw.get("image"))

    if threshold_usd <= 0:
        raise ValueError("threshold_usd must be greater than zero")
    if min_previous_usd is not None and min_previous_usd < 0:
        raise ValueError("min_previous_usd must be zero or greater")
    if percent_increase is not None:
        if percent_increase <= 0:
            raise ValueError("percent_increase must be greater than zero")
        if min_previous_usd is None:
            raise ValueError(
                "min_previous_usd is required when percent_increase is set"
            )
    return CostAlertRule(
        name=name,
        grain=grain,
        group_by=group_by,
        threshold_usd=threshold_usd,
        min_previous_usd=min_previous_usd,
        percent_increase=percent_increase,
        include_missing=include_missing,
        image=image,
    )


def _parse_image_config(value: object) -> CostAlertImageConfig | None:
    if value is None:
        return None
    raw = _string_key_dict(value, "image")
    url = _optional_string(raw.get("url"), "image.url")
    if url is None:
        raise ValueError("image.url must be a non-empty string")
    if not url.startswith(("http://", "https://")):
        raise ValueError("image.url must start with http:// or https://")
    if len(url) > _SLACK_IMAGE_URL_MAX_LENGTH:
        raise ValueError("image.url must not exceed 3000 characters")
    alt_text = _optional_string(raw.get("alt_text"), "image.alt_text")
    if alt_text is not None and len(alt_text) > _SLACK_IMAGE_ALT_TEXT_MAX_LENGTH:
        raise ValueError("image.alt_text must not exceed 2000 characters")
    attach_to = _parse_image_attach_to(raw.get("attach_to"))
    min_threshold_multiple = _optional_positive_decimal(
        raw.get("min_threshold_multiple"), "image.min_threshold_multiple"
    )
    return CostAlertImageConfig(
        url=url,
        alt_text=alt_text or "cost alert image",
        attach_to=attach_to,
        min_threshold_multiple=min_threshold_multiple,
    )


def _parse_image_attach_to(value: object) -> tuple[CostAlertImageAttachTo, ...]:
    if value is None:
        return ("worsening",)
    raw_values = _object_list(value, "image.attach_to")
    if not raw_values:
        raise ValueError("image.attach_to must include at least one status")
    attach_to: list[CostAlertImageAttachTo] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or raw_value not in _ALLOWED_IMAGE_ATTACH_TO:
            raise ValueError(
                f"image.attach_to must contain only {_ALLOWED_IMAGE_ATTACH_TO}"
            )
        attach_to.append(raw_value)
    return tuple(attach_to)


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_positive_decimal(value: object, field: str) -> Decimal | None:
    if value is None:
        return None
    parsed = _decimal_from_object(value, field)
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _parse_grain(value: str) -> CostAlertGrain:
    if value == "1h":
        return "1h"
    if value == "1d":
        return "1d"
    raise ValueError(f"grain must be one of {_ALLOWED_GRAINS}")


def _parse_group_by(value: str) -> CostAlertGroupBy:
    if value == "global":
        return "global"
    if value == "provider_model":
        return "provider_model"
    if value == "benchmark":
        return "benchmark"
    if value == "identity_email":
        return "identity_email"
    raise ValueError(f"group_by must be one of {_ALLOWED_GROUP_BYS}")


def _required_string(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _decimal_from_object(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def _bool_with_default(raw: Mapping[str, object], field: str, default: bool) -> bool:
    value = raw.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _alert_dimension_sql(group_by: CostAlertGroupBy) -> str:
    column = _DIMENSION_COLUMN_BY_GROUP[group_by]
    if column is None:
        return "'all'"
    return _normalized_dimension_sql(f"agg.{column}")


def _normalized_dimension_sql(column: str) -> str:
    missing = redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)
    return f"coalesce(nullif({column}, ''), {missing})"


def _dimension_filter_sql(rule: CostAlertRule) -> str:
    if rule.group_by == "global" or rule.include_missing:
        return ""
    dimension_sql = _alert_dimension_sql(rule.group_by)
    missing = redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)
    return f"    and {dimension_sql} <> {missing}\n"


def _breakdown_scope_filter_sql(
    rule: CostAlertRule,
    scope_values: Sequence[str],
) -> str:
    if rule.group_by == "global":
        return ""
    dimension_sql = _alert_dimension_sql(rule.group_by)
    literals = ", ".join(redshift_schema.sql_literal(value) for value in scope_values)
    return f"    and {dimension_sql} in ({literals})\n"


def _cost_usd_sql(ignored_model_keys: Sequence[str]) -> str:
    ignored_condition = _ignored_model_condition_sql(ignored_model_keys)
    if ignored_condition is None:
        return "agg.cost_usd_sum"
    return f"""case
      when {ignored_condition} then 0
      else agg.cost_usd_sum
    end"""


def _eligible_request_count_sql(ignored_model_keys: Sequence[str]) -> str:
    ignored_condition = _ignored_model_condition_sql(ignored_model_keys)
    if ignored_condition is None:
        return "agg.request_count"
    return f"""case
      when {ignored_condition} then 0
      else agg.request_count
    end"""


def _ignored_for_cost_sql(ignored_model_keys: Sequence[str]) -> str:
    ignored_condition = _ignored_model_condition_sql(ignored_model_keys)
    if ignored_condition is None:
        return "0"
    return f"""case
      when {ignored_condition} then 1
      else 0
    end"""


def _ignored_model_condition_sql(ignored_model_keys: Sequence[str]) -> str | None:
    if not ignored_model_keys:
        return None
    literals = ", ".join(
        redshift_schema.sql_literal(model_key) for model_key in ignored_model_keys
    )
    return f"agg.provider_model in ({literals})"


def _metadata_name(value: object) -> str:
    metadata = _string_key_dict(value, "Redshift column metadata")
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Redshift column metadata entries must include name")
    return name


def _cell_value(cell: object) -> object:
    cell_mapping = _string_key_dict(cell, "Redshift record cell")
    if cell_mapping.get("isNull"):
        return None
    for key in ("stringValue", "longValue"):
        if key in cell_mapping:
            return cell_mapping[key]
    raise ValueError("Unsupported Redshift Data API cell")


def _datetime_value(values: Mapping[str, object], field: str) -> datetime:
    raw = values.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a timestamp string")
    normalized = raw.replace(" ", "T").replace("Z", "+00:00")
    value = datetime.fromisoformat(normalized)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _comparison_row_from_values(values: Mapping[str, object]) -> ComparisonRow:
    return ComparisonRow(
        bucket_start_utc=_datetime_value(values, "bucket_start_utc"),
        bucket_end_utc=_datetime_value(values, "bucket_end_utc"),
        data_through_utc=_datetime_value(values, "data_through_utc"),
        watermark_utc=_datetime_value(values, "watermark_utc"),
        dimension_value=_string_value(values, "dimension_value"),
        current_request_count=_int_value(values, "current_request_count"),
        current_eligible_request_count=_int_value(
            values, "current_eligible_request_count"
        ),
        current_cost_usd=_decimal_value(values, "current_cost_usd"),
        previous_cost_usd=_decimal_value(values, "previous_cost_usd"),
        absolute_increase_usd=_decimal_value(values, "absolute_increase_usd"),
        percent_increase=_optional_decimal_value(values, "percent_increase"),
    )


def _breakdown_group_value(
    values: Mapping[str, object], field: str
) -> CostAlertBreakdownGroupBy:
    raw = _string_value(values, field)
    if raw in _BREAKDOWN_GROUP_BYS:
        return raw
    raise ValueError(f"{field} must be one of {_BREAKDOWN_GROUP_BYS}")


def _string_value(values: Mapping[str, object], field: str) -> str:
    raw = values.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    return raw


def _int_value(values: Mapping[str, object], field: str) -> int:
    raw = values.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer")
    return raw


def _decimal_value(values: Mapping[str, object], field: str) -> Decimal:
    raw = values.get(field)
    return _decimal_from_object(raw, field)


def _optional_decimal_value(values: Mapping[str, object], field: str) -> Decimal | None:
    raw = values.get(field)
    if raw is None:
        return None
    return _decimal_from_object(raw, field)


def _include_row(rule: CostAlertRule, row: ComparisonRow) -> bool:
    return (
        rule.include_missing
        or row.dimension_value != redshift_schema.MISSING_DIMENSION_VALUE
    )


def _row_matches_rule(rule: CostAlertRule, row: ComparisonRow) -> bool:
    if row.current_cost_usd < rule.threshold_usd:
        return False
    if rule.percent_increase is None:
        return True
    if rule.min_previous_usd is None or row.previous_cost_usd < rule.min_previous_usd:
        return False
    percent_increase = row.percent_increase
    if percent_increase is None and row.previous_cost_usd > 0:
        percent_increase = (
            row.absolute_increase_usd / row.previous_cost_usd
        ) * Decimal("100")
    if percent_increase is None:
        return False
    return percent_increase >= rule.percent_increase


def _contributor_from_values(values: Mapping[str, object]) -> AlertContributor:
    dimension_value = _string_value(values, "dimension_value")
    return AlertContributor(
        display_value=display_dimension_value(dimension_value),
        current_request_count=_int_value(values, "current_request_count"),
        current_cost_usd=_decimal_value(values, "current_cost_usd"),
        ignored_for_cost=_int_value(values, "ignored_for_cost") == 1,
    )


def _string_key_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _object_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return cast(list[object], value)
