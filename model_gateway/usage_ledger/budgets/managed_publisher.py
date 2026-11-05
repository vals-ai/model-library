"""Publish managed project and key cap spend from recorded warehouse attribution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import UUID

import sqlalchemy as sa
from botocore.exceptions import ClientError
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.interfaces import BindTyping

from model_gateway.budgets.schema import (
    BudgetPeriod,
    PeriodSpend,
    SpendSnapshot,
    managed_key_spend_id,
    period_bounds,
)
from model_gateway.managed_auth.types import (
    ProjectAlertIdentity,
    ProjectLimitRequest,
    ProjectPolicy,
)
from model_gateway.usage_ledger import (
    gateway_registry,
    redshift_schema,
    redshift_statements,
)


class SnapshotS3Client(Protocol):
    def get_object(self, **kwargs: object) -> Mapping[str, Any]: ...

    def put_object(self, **kwargs: object) -> Mapping[str, Any]: ...


class SnapshotRedshiftClient(Protocol):
    def execute_statement(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_statement(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_statement_result(self, **kwargs: object) -> Mapping[str, object]: ...


def read_existing_snapshot(
    client: SnapshotS3Client, bucket: str, key: str, stage: str
) -> tuple[SpendSnapshot | None, str | None]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None, None
        raise
    snapshot = SpendSnapshot.model_validate_json(response["Body"].read())
    if snapshot.stage != stage:
        raise ValueError("budget snapshot stage does not match publisher stage")
    return snapshot, cast(str, response["ETag"])


def read_alert_publication_token(
    client: SnapshotS3Client, bucket: str, key: str, stage: str
) -> tuple[int, str | None]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return 0, None
        raise
    previous = json.loads(response["Body"].read())
    if previous["stage"] != stage:
        raise ValueError("project alert snapshot stage does not match publisher stage")
    return previous["revision"], cast(str, response["ETag"])


_PERIODS: tuple[BudgetPeriod, ...] = ("daily", "weekly", "monthly")
_TOTAL_ROW = "total"


@dataclass(frozen=True)
class ManagedSpendPublishResult:
    snapshot: SpendSnapshot
    contributors: dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]]
    models: dict[BudgetPeriod, dict[str, dict[str, Decimal]]]
    benchmarks: dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]]


def publish_managed_spend_snapshot(
    *,
    s3_client: SnapshotS3Client,
    redshift_data_client: SnapshotRedshiftClient,
    project_starts: dict[UUID, datetime],
    key_ids: list[UUID],
    stage: str,
    bucket: str,
    key: str,
    workgroup: str,
    database: str,
    secret_arn: str,
    schema: str,
    data_through: datetime | None,
    as_of: datetime,
    idle: bool = False,
) -> ManagedSpendPublishResult | None:
    """Publish admission spend from one successful fact query."""
    registry = gateway_registry.fetch_live_models()
    if registry is None:
        raise RuntimeError("current model registry is unavailable for budget spend")
    ignored = tuple(
        sorted(
            key
            for key, model in registry.items()
            if model["metadata"]["ignored_for_cost"]
        )
    )
    ignored_hash = sha256(json.dumps(ignored).encode("utf-8")).hexdigest()
    while True:
        previous, etag = read_existing_snapshot(s3_client, bucket, key, stage)
        if data_through is None:
            if previous is None:
                return None
            cutoff = previous.data_through
        elif previous is None:
            cutoff = data_through
        elif idle:
            cutoff = previous.data_through
        else:
            cutoff = max(previous.data_through, data_through)
        if previous is not None:
            as_of = max(as_of, previous.periods["daily"].start)
        bounds: dict[BudgetPeriod, tuple[datetime, datetime]] = {
            period: period_bounds(period, as_of) for period in _PERIODS
        }
        project_ids = list(project_starts)
        sql, parameters = _managed_spend_statement(
            bounds, schema, ignored, project_starts, key_ids
        )
        spend, totals, contributors, models, benchmarks = _query_managed_spend(
            redshift_data_client,
            workgroup=workgroup,
            database=database,
            secret_arn=secret_arn,
            sql=sql,
            parameters=parameters,
            project_ids=project_ids,
            key_ids=key_ids,
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
            total_spend_usd=totals,
        )
        condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=snapshot.model_dump_json().encode("utf-8"),
                ContentType="application/json",
                Metadata={"ignored-for-cost-sha256": ignored_hash},
                **condition,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise
            continue
        return ManagedSpendPublishResult(snapshot, contributors, models, benchmarks)


def _build_managed_spend_query(
    bounds: dict[BudgetPeriod, tuple[datetime, datetime]],
    schema: str,
    ignored_models: tuple[str, ...],
    project_ids: list[UUID],
    key_ids: list[UUID],
) -> str:
    table = redshift_schema.qualified_table_name("usage_events", schema)
    periods = "\n  union all\n".join(
        f"  select {redshift_schema.sql_literal(period)}, "
        f"cast({redshift_schema.sql_literal(start.isoformat())} as timestamptz), "
        f"cast({redshift_schema.sql_literal(end.isoformat())} as timestamptz)"
        for period, (start, end) in bounds.items()
    )
    projects = ", ".join(
        redshift_schema.sql_literal(str(project_id)) for project_id in project_ids
    )
    keys = ", ".join(redshift_schema.sql_literal(str(key_id)) for key_id in key_ids)
    project_filter = f"project_id in ({projects})" if projects else "1 = 0"
    key_filter = f"managed_key_id in ({keys})" if keys else "1 = 0"
    project_scope = f"fact.project_id in ({projects})" if projects else "1 = 0"
    key_scope = f"fact.managed_key_id in ({keys})" if keys else "1 = 0"
    ignored = ""
    if ignored_models:
        ignored = (
            "and fact.requested_model_key not in ("
            + ", ".join(redshift_schema.sql_literal(key) for key in ignored_models)
            + ")"
        )
    return f"""with budget_periods(period, period_start, period_end) as (
{periods}
), attributed_facts as (
  select p.period, fact.project_id, fact.managed_key_id, fact.identity_email, fact.requested_model_key, fact.benchmark_name, fact.cost_usd
  from {table} fact
  join budget_periods p
    on fact.completed_at >= p.period_start
   and fact.completed_at < p.period_end
  where fact.cost_usd is not null
    and ({project_scope} or {key_scope})
    {ignored}
)
select period, 'project', project_id, identity_email, sum(cost_usd)
from attributed_facts
where {project_filter}
group by period, project_id, identity_email
union all
select period, 'key', managed_key_id, null, sum(cost_usd)
from attributed_facts
where {key_filter}
group by period, managed_key_id
union all
select period, 'model', project_id, requested_model_key, sum(cost_usd)
from attributed_facts
where {project_filter}
group by period, project_id, requested_model_key
union all
select period, 'benchmark', project_id, benchmark_name, sum(cost_usd)
from attributed_facts
where {project_filter}
group by period, project_id, benchmark_name"""


def _managed_spend_statement(
    bounds: dict[BudgetPeriod, tuple[datetime, datetime]],
    schema: str,
    ignored_models: tuple[str, ...],
    project_starts: dict[UUID, datetime],
    key_ids: list[UUID],
) -> tuple[str, list[dict[str, str]]]:
    sql = _build_managed_spend_query(
        bounds, schema, ignored_models, list(project_starts), key_ids
    )
    if not project_starts:
        return sql, []
    totals, parameters = _total_spend_query(schema, ignored_models, project_starts)
    return "\n".join((sql, "union all", totals)), parameters


def _total_spend_query(
    schema: str, ignored_models: tuple[str, ...], project_starts: dict[UUID, datetime]
) -> tuple[str, list[dict[str, str]]]:
    facts = sa.table(
        "usage_events",
        sa.column("project_id"),
        sa.column("completed_at"),
        sa.column("cost_usd"),
        sa.column("requested_model_key"),
        schema=schema,
    )
    starts = sa.union_all(
        *(
            sa.select(
                sa.cast(sa.literal(str(project_id)), sa.String(128)).label(
                    "project_id"
                ),
                sa.cast(
                    sa.literal(start.isoformat()), sa.DateTime(timezone=True)
                ).label("budget_start"),
            )
            for project_id, start in sorted(project_starts.items())
        )
    ).subquery("starts")
    query = (
        sa.select(
            sa.cast(sa.literal(_TOTAL_ROW), sa.String(16)),
            sa.cast(sa.literal("project"), sa.String(16)),
            facts.c.project_id,
            sa.null(),
            sa.func.sum(facts.c.cost_usd),
        )
        .select_from(
            facts.join(
                starts,
                sa.and_(
                    facts.c.project_id == starts.c.project_id,
                    facts.c.completed_at >= starts.c.budget_start,
                ),
            )
        )
        .where(
            facts.c.cost_usd.is_not(None),
            facts.c.requested_model_key.not_in(
                [sa.cast(sa.literal(model), sa.String(256)) for model in ignored_models]
            ),
        )
        .group_by(facts.c.project_id)
    )
    dialect = postgresql.dialect(paramstyle="named")
    dialect.bind_typing = BindTyping.NONE
    compiled = query.compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True}
    )
    return str(compiled), [
        {"name": name, "value": str(value)} for name, value in compiled.params.items()
    ]


def _query_managed_spend(
    client: SnapshotRedshiftClient,
    *,
    workgroup: str,
    database: str,
    secret_arn: str,
    sql: str,
    parameters: list[dict[str, str]],
    project_ids: list[UUID],
    key_ids: list[UUID],
) -> tuple[
    dict[BudgetPeriod, dict[str, Decimal]],
    dict[str, Decimal],
    dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]],
    dict[BudgetPeriod, dict[str, dict[str, Decimal]]],
    dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]],
]:
    response = client.execute_statement(
        WorkgroupName=workgroup,
        Database=database,
        SecretArn=secret_arn,
        Sql=sql,
        StatementName="usage-managed-budget-spend-snapshot",
        **({"Parameters": parameters} if parameters else {}),
    )
    statement_id = str(response["Id"])
    redshift_statements.wait_for_statement(client, statement_id)
    spend: dict[BudgetPeriod, dict[str, Decimal]] = {
        period: {
            **{str(project_id): Decimal(0) for project_id in project_ids},
            **{managed_key_spend_id(key_id): Decimal(0) for key_id in key_ids},
        }
        for period in _PERIODS
    }
    totals: dict[str, Decimal] = {
        str(project_id): Decimal(0) for project_id in project_ids
    }
    contributors: dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]] = {
        period: {str(project_id): {} for project_id in project_ids}
        for period in _PERIODS
    }
    models: dict[BudgetPeriod, dict[str, dict[str, Decimal]]] = {
        period: {str(project_id): {} for project_id in project_ids}
        for period in _PERIODS
    }
    benchmarks: dict[BudgetPeriod, dict[str, dict[str | None, Decimal]]] = {
        period: {str(project_id): {} for project_id in project_ids}
        for period in _PERIODS
    }
    next_token: str | None = None
    while True:
        args: dict[str, object] = {"Id": statement_id}
        if next_token is not None:
            args["NextToken"] = next_token
        page = client.get_statement_result(**args)
        for record in cast(list[list[dict[str, Any]]], page["Records"]):
            dimension = cast(str, record[1]["stringValue"])
            root_id = cast(str, record[2]["stringValue"])
            amount = Decimal(str(record[4]["stringValue"]))
            if record[0]["stringValue"] == _TOTAL_ROW:
                totals[root_id] += amount
                continue
            period = cast(BudgetPeriod, record[0]["stringValue"])
            if dimension == "project" or dimension == "key":
                spend_id = root_id if dimension == "project" else f"key:{root_id}"
                spend[period][spend_id] += amount
            if dimension == "project":
                email = cast(str | None, record[3].get("stringValue"))
                project_contributors = contributors[period][root_id]
                project_contributors[email] = (
                    project_contributors.get(email, Decimal(0)) + amount
                )
            elif dimension == "model":
                model_name = cast(str, record[3]["stringValue"])
                models[period][root_id][model_name] = amount
            elif dimension == "benchmark":
                benchmark_name = cast(str | None, record[3].get("stringValue"))
                benchmarks[period][root_id][benchmark_name] = amount
        next_token = cast(str | None, page.get("NextToken"))
        if next_token is None:
            return spend, totals, contributors, models, benchmarks


def publish_project_alert_snapshot(
    *,
    s3_client: SnapshotS3Client,
    bucket: str,
    key: str,
    result: ManagedSpendPublishResult,
    policies: tuple[ProjectPolicy, ...],
    identities: dict[UUID, ProjectAlertIdentity],
    extras: dict[UUID, tuple[ProjectLimitRequest, ...]],
    publication_token: tuple[int, str | None],
) -> None:
    projects: list[dict[str, object]] = []
    for policy in policies:
        project_id = policy.project.project_id
        identity = identities.get(project_id)
        if identity is None:
            continue
        period = policy.period
        period_spend = result.snapshot.periods[period]
        projects.append(
            {
                "project_id": str(project_id),
                "name": identity.name,
                "handle": identity.handle,
                "organization_id": policy.project.organization,
                "environment": policy.project.environment,
                "period": period,
                "start": period_spend.start.isoformat(),
                "end": period_spend.end.isoformat(),
                "limit_usd": str(
                    policy.base_limit_usd
                    # CAS can advance the published day past the day used to load extras.
                    + sum(
                        (
                            extra.increase_usd
                            for extra in extras.get(project_id, ())
                            if cast(datetime, extra.period_end)
                            > result.snapshot.periods["daily"].start
                        ),
                        Decimal(0),
                    )
                ),
                "spend_usd": str(period_spend.spend_usd[str(project_id)]),
                "contributors": [
                    {"email": email, "spend_usd": str(amount)}
                    for email, amount in sorted(
                        result.contributors[period][str(project_id)].items(),
                        key=lambda item: (item[0] is not None, item[0] or ""),
                    )
                ],
                "models": [
                    {"name": name, "spend_usd": str(amount)}
                    for name, amount in sorted(
                        result.models[period][str(project_id)].items()
                    )
                ],
                "benchmarks": [
                    {"name": name, "spend_usd": str(amount)}
                    for name, amount in sorted(
                        result.benchmarks[period][str(project_id)].items(),
                        key=lambda item: (item[0] is not None, item[0] or ""),
                    )
                ],
            }
        )
    revision, etag = publication_token
    document = {
        "schema_version": 1,
        "stage": result.snapshot.stage,
        "revision": revision + 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_through": result.snapshot.data_through.isoformat(),
        "projects": projects,
    }
    condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(document).encode("utf-8"),
            ContentType="application/json",
            **condition,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
        }:
            raise
