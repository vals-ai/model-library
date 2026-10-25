"""Publish current UTC email spend from Redshift's daily usage aggregates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol, cast

from botocore.exceptions import ClientError

from model_gateway.budgets.schema import (
    BudgetPeriod,
    PeriodSpend,
    SpendSnapshot,
    period_bounds,
)
from model_gateway.usage_ledger import (
    gateway_registry,
    redshift_schema,
    redshift_statements,
)
from model_gateway.usage_ledger.schema import canonical_email_value


class SnapshotS3Client(Protocol):
    def get_object(self, **kwargs: object) -> Mapping[str, Any]: ...

    def put_object(self, **kwargs: object) -> Mapping[str, Any]: ...


class SnapshotRedshiftClient(Protocol):
    def execute_statement(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_statement(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_statement_result(self, **kwargs: object) -> Mapping[str, object]: ...


_PERIODS: tuple[BudgetPeriod, ...] = ("daily", "weekly", "monthly")
_IGNORED_MODELS_HASH_KEY = "ignored-for-cost-sha256"


def publish_spend_snapshot(
    *,
    s3_client: SnapshotS3Client,
    redshift_data_client: SnapshotRedshiftClient,
    stage: str,
    bucket: str,
    key: str,
    workgroup: str,
    database: str,
    secret_arn: str,
    schema: str,
    data_through: datetime | None,
    idle: bool = False,
) -> bool:
    """Publish only against the S3 version read before each Redshift query."""
    conflict = False
    while True:
        previous, etag, previous_ignored_hash = _read_existing_snapshot(
            s3_client, bucket, key, stage
        )
        if data_through is None:
            if previous is None:
                return False
            cutoff = previous.data_through
        elif previous is None:
            cutoff = data_through
        elif idle:
            cutoff = previous.data_through
        else:
            cutoff = max(previous.data_through, data_through)
        as_of = datetime.now(UTC)
        if previous is not None:
            as_of = max(as_of, previous.periods["daily"].start)
        bounds: dict[BudgetPeriod, tuple[datetime, datetime]] = {
            period: period_bounds(period, as_of) for period in _PERIODS
        }
        registry = gateway_registry.fetch_live_models()
        if registry is None:
            raise RuntimeError("current model registry is unavailable for budget spend")
        ignored = tuple(
            sorted(
                model_key
                for model_key, model in registry.items()
                if model["metadata"]["ignored_for_cost"]
            )
        )
        ignored_hash = sha256(json.dumps(ignored).encode("utf-8")).hexdigest()
        if (
            idle
            and not conflict
            and previous is not None
            and all(
                (previous.periods[period].start, previous.periods[period].end)
                == bounds[period]
                for period in _PERIODS
            )
            and previous_ignored_hash == ignored_hash
        ):
            return False
        spend = _query_spend(
            redshift_data_client,
            workgroup=workgroup,
            database=database,
            secret_arn=secret_arn,
            sql=_build_spend_query(bounds, schema, ignored),
        )
        snapshot = SpendSnapshot(
            schema_version=1,
            stage=stage,
            revision=previous.revision + 1 if previous is not None else 1,
            generated_at=datetime.now(UTC),
            data_through=cutoff,
            periods={
                period: PeriodSpend(start=start, end=end, spend_usd=spend[period])
                for period, (start, end) in bounds.items()
            },
        )
        condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=snapshot.model_dump_json().encode("utf-8"),
                ContentType="application/json",
                Metadata={_IGNORED_MODELS_HASH_KEY: ignored_hash},
                **condition,
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise
            conflict = True


def _read_existing_snapshot(
    client: SnapshotS3Client, bucket: str, key: str, stage: str
) -> tuple[SpendSnapshot | None, str | None, str | None]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None, None, None
        raise
    snapshot = SpendSnapshot.model_validate_json(response["Body"].read())
    if snapshot.stage != stage:
        raise ValueError("budget snapshot stage does not match publisher stage")
    return (
        snapshot,
        cast(str, response["ETag"]),
        cast(str, response["Metadata"][_IGNORED_MODELS_HASH_KEY]),
    )


def _build_spend_query(
    bounds: dict[BudgetPeriod, tuple[datetime, datetime]],
    schema: str,
    ignored_models: tuple[str, ...],
) -> str:
    table = redshift_schema.qualified_table_name("usage_agg_1d", schema)
    period_rows: list[str] = []
    for period, (start, end) in bounds.items():
        start_sql = redshift_schema.sql_literal(start.isoformat())
        end_sql = redshift_schema.sql_literal(end.isoformat())
        period_rows.append(
            f"  select {redshift_schema.sql_literal(period)}, "
            f"cast({start_sql} as timestamptz), cast({end_sql} as timestamptz)"
        )
    periods = "\n  union all\n".join(period_rows)
    ignored = ""
    if ignored_models:
        models_sql = ", ".join(
            redshift_schema.sql_literal(key) for key in ignored_models
        )
        ignored = f"and agg.provider_model not in ({models_sql})"
    return f"""with budget_periods(period, period_start, period_end) as (
{periods}
)
select p.period, lower(trim(agg.identity_email)) as email, sum(agg.cost_usd_sum) as spend_usd
from {table} agg
join budget_periods p
  on agg.bucket_start_utc >= p.period_start
 and agg.bucket_start_utc < p.period_end
where agg.identity_email <> {redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)}
  and agg.cost_recorded_count > 0
  {ignored}
group by p.period, lower(trim(agg.identity_email))"""


def _query_spend(
    client: SnapshotRedshiftClient,
    *,
    workgroup: str,
    database: str,
    secret_arn: str,
    sql: str,
) -> dict[BudgetPeriod, dict[str, Decimal]]:
    response = client.execute_statement(
        WorkgroupName=workgroup,
        Database=database,
        SecretArn=secret_arn,
        Sql=sql,
        StatementName="usage-budget-spend-snapshot",
    )
    statement_id = str(response["Id"])
    redshift_statements.wait_for_statement(client, statement_id)
    spend: dict[BudgetPeriod, dict[str, Decimal]] = {period: {} for period in _PERIODS}
    next_token: str | None = None
    while True:
        args: dict[str, object] = {"Id": statement_id}
        if next_token is not None:
            args["NextToken"] = next_token
        page = client.get_statement_result(**args)
        for record in cast(list[list[dict[str, object]]], page["Records"]):
            period = cast(BudgetPeriod, record[0]["stringValue"])
            email = canonical_email_value(record[1]["stringValue"])
            assert email is not None
            spend[period][email] = Decimal(str(record[2]["stringValue"]))
        next_token = cast(str | None, page.get("NextToken"))
        if next_token is None:
            return spend
