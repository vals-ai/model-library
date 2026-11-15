"""Publish managed project and key cap spend from recorded warehouse attribution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol, Unpack, cast
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
    ProjectPolicy,
)
from model_gateway.usage_ledger import (
    gateway_registry,
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
_ALERT_BREAKDOWN_PERIOD: BudgetPeriod = "monthly"
PROJECT_ALERT_SCHEMA_VERSION = 2
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


def _inline(value: str) -> sa.BindParameter[str]:
    return sa.bindparam(None, value, type_=sa.String(), literal_execute=True)


def _inline_in(
    column: sa.ColumnElement[Any], values: list[str]
) -> sa.ColumnElement[bool]:
    return column.in_(
        sa.bindparam(
            None, values, type_=sa.String(), expanding=True, literal_execute=True
        )
    )


def _cost_model_key(facts: sa.TableClause) -> sa.ColumnElement[Any]:
    return sa.func.coalesce(facts.c.served_model_key, facts.c.requested_model_key)


def _managed_spend_query(
    bounds: dict[BudgetPeriod, tuple[datetime, datetime]],
    schema: str,
    ignored_models: tuple[str, ...],
    project_ids: list[UUID],
    key_ids: list[UUID],
) -> sa.CompoundSelect[Unpack[tuple[Any, ...]]]:
    facts = sa.table(
        "usage_events",
        sa.column("completed_at"),
        sa.column("project_id"),
        sa.column("managed_key_id"),
        sa.column("identity_email"),
        sa.column("requested_model_key"),
        sa.column("served_model_key"),
        sa.column("benchmark_name"),
        sa.column("cost_usd"),
        schema=schema,
    )
    periods = sa.union_all(
        *(
            sa.select(
                sa.cast(_inline(period), sa.String(16)).label("period"),
                sa.cast(_inline(start.isoformat()), sa.DateTime(timezone=True)).label(
                    "period_start"
                ),
                sa.cast(_inline(end.isoformat()), sa.DateTime(timezone=True)).label(
                    "period_end"
                ),
            )
            for period, (start, end) in bounds.items()
        )
    ).cte("budget_periods")
    projects = [str(project_id) for project_id in project_ids]
    keys = [str(key_id) for key_id in key_ids]
    attributed = (
        sa.select(
            periods.c.period,
            facts.c.project_id,
            facts.c.managed_key_id,
            facts.c.identity_email,
            facts.c.requested_model_key,
            facts.c.benchmark_name,
            facts.c.cost_usd,
        )
        .select_from(
            facts.join(
                periods,
                sa.and_(
                    facts.c.completed_at >= periods.c.period_start,
                    facts.c.completed_at < periods.c.period_end,
                ),
            )
        )
        .where(
            facts.c.cost_usd.is_not(None),
            sa.or_(
                _inline_in(facts.c.project_id, projects),
                _inline_in(facts.c.managed_key_id, keys),
            ),
            sa.not_(_inline_in(_cost_model_key(facts), list(ignored_models))),
        )
        .cte("attributed_facts")
    )
    row = attributed.c
    return sa.union_all(
        *(
            sa.select(
                row.period,
                sa.cast(_inline(dimension), sa.String(16)),
                root,
                detail,
                sa.func.sum(row.cost_usd),
            )
            .where(_inline_in(root, ids))
            .group_by(row.period, root, *grouping)
            for dimension, root, detail, ids, grouping in (
                (
                    "project",
                    row.project_id,
                    row.identity_email,
                    projects,
                    (row.identity_email,),
                ),
                ("key", row.managed_key_id, sa.null(), keys, ()),
                (
                    "model",
                    row.project_id,
                    row.requested_model_key,
                    projects,
                    (row.requested_model_key,),
                ),
                (
                    "benchmark",
                    row.project_id,
                    row.benchmark_name,
                    projects,
                    (row.benchmark_name,),
                ),
            )
        )
    )


def _compile(
    statement: sa.CompoundSelect[Unpack[tuple[Any, ...]]],
) -> tuple[str, list[dict[str, str]]]:
    dialect = postgresql.dialect(paramstyle="named")
    dialect.bind_typing = BindTyping.NONE
    compiled = statement.compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True}
    )
    return str(compiled), [
        {"name": name, "value": str(value)} for name, value in compiled.params.items()
    ]


def _managed_spend_statement(
    bounds: dict[BudgetPeriod, tuple[datetime, datetime]],
    schema: str,
    ignored_models: tuple[str, ...],
    project_starts: dict[UUID, datetime],
    key_ids: list[UUID],
) -> tuple[str, list[dict[str, str]]]:
    periods = _managed_spend_query(
        bounds, schema, ignored_models, list(project_starts), key_ids
    )
    if not project_starts:
        return _compile(periods)
    return _compile(
        sa.union_all(
            *periods.selects, _total_spend_query(schema, ignored_models, project_starts)
        )
    )


def _total_spend_query(
    schema: str, ignored_models: tuple[str, ...], project_starts: dict[UUID, datetime]
) -> sa.Select[Unpack[tuple[Any, ...]]]:
    facts = sa.table(
        "usage_events",
        sa.column("project_id"),
        sa.column("completed_at"),
        sa.column("cost_usd"),
        sa.column("requested_model_key"),
        sa.column("served_model_key"),
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
    return (
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
            _cost_model_key(facts).not_in(
                [sa.cast(sa.literal(model), sa.String(256)) for model in ignored_models]
            ),
        )
        .group_by(facts.c.project_id)
    )


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
    publication_token: tuple[int, str | None],
) -> None:
    projects: list[dict[str, object]] = []
    totals = cast(dict[str, Decimal], result.snapshot.total_spend_usd)
    for policy in policies:
        project_id = policy.project.project_id
        identity = identities.get(project_id)
        if identity is None:
            continue
        projects.append(
            {
                "project_id": str(project_id),
                "name": identity.name,
                "handle": identity.handle,
                "organization_id": policy.project.organization,
                "environment": policy.project.environment,
                "total_started_at": policy.total_started_at.isoformat(),
                "total_limit_usd": str(policy.total_limit_usd),
                "total_spend_usd": str(totals.get(str(project_id), Decimal(0))),
                "contributors": [
                    {"email": email, "spend_usd": str(amount)}
                    for email, amount in sorted(
                        result.contributors[_ALERT_BREAKDOWN_PERIOD][
                            str(project_id)
                        ].items(),
                        key=lambda item: (item[0] is not None, item[0] or ""),
                    )
                ],
                "models": [
                    {"name": name, "spend_usd": str(amount)}
                    for name, amount in sorted(
                        result.models[_ALERT_BREAKDOWN_PERIOD][str(project_id)].items()
                    )
                ],
                "benchmarks": [
                    {"name": name, "spend_usd": str(amount)}
                    for name, amount in sorted(
                        result.benchmarks[_ALERT_BREAKDOWN_PERIOD][
                            str(project_id)
                        ].items(),
                        key=lambda item: (item[0] is not None, item[0] or ""),
                    )
                ],
            }
        )
    revision, etag = publication_token
    document = {
        "schema_version": PROJECT_ALERT_SCHEMA_VERSION,
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
