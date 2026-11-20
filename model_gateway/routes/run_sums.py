"""Model dollars per run and on one project key, read from Redshift `usage_events`."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from uuid import UUID

from boto3 import client as boto3_client  # pyright: ignore[reportUnknownVariableType]
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from model_gateway.budgets.service import BudgetService
from model_gateway.managed_auth.api import machine
from model_gateway.usage_ledger import redshift_statements
from model_gateway.usage_ledger.redshift_schema import GRADER_AGENT_NAME
from model_library.register_models import get_model_registry

RUN_LIMIT = 1000
CUTOFF_LAG = timedelta(minutes=15)
STATEMENT_SECONDS = 15

router = APIRouter(prefix="/admin/managed-auth", tags=["usage"])


class RedshiftClient(Protocol):
    def execute_statement(self, **kwargs: object) -> Any: ...
    def describe_statement(self, **kwargs: object) -> Any: ...
    def cancel_statement(self, **kwargs: object) -> Any: ...
    def get_statement_result(self, **kwargs: object) -> Any: ...


@dataclass(frozen=True)
class UsageRedshift:
    client: RedshiftClient
    workgroup: str
    database: str
    secret_arn: str
    schema: str


class RunSumsRequest(BaseModel):
    run_ids: list[str] = Field(max_length=RUN_LIMIT)
    project_id: UUID | None = None
    project_environment: str


class RunSum(BaseModel):
    run_id: str | None
    amount_usd: str
    grading_usd: str


class RunSumsResponse(BaseModel):
    runs: list[RunSum]
    project_key: list[RunSum]
    through: datetime


def usage_redshift_from_env() -> UsageRedshift | None:
    secret_arn = os.environ.get("GATEWAY_USAGE_REDSHIFT_READER_SECRET_ARN")
    if not secret_arn:
        return None
    return UsageRedshift(
        client=cast(RedshiftClient, boto3_client("redshift-data")),
        workgroup=os.environ["GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME"],
        database=os.environ["GATEWAY_USAGE_REDSHIFT_DATABASE_NAME"],
        secret_arn=secret_arn,
        schema=os.environ["GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME"],
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def ignored_cost_models() -> tuple[str, ...]:
    registry = get_model_registry()
    return tuple(sorted(k for k, c in registry.items() if c.metadata.ignored_for_cost))


def _statement(
    schema: str,
    body: RunSumsRequest,
    ignored: tuple[str, ...],
    through: datetime,
) -> tuple[str, list[dict[str, str]]]:
    parameters = [
        {"name": "through", "value": through.isoformat()},
        {"name": "grader", "value": GRADER_AGENT_NAME},
    ]
    cost = "r.cost_usd"
    if ignored:
        names = [f"ignored_{index}" for index in range(len(ignored))]
        parameters += [{"name": n, "value": v} for n, v in zip(names, ignored)]
        placeholders = ", ".join(f":{name}" for name in names)
        cost = (
            "CASE WHEN COALESCE(r.served_model_key, r.requested_model_key)"
            f" IN ({placeholders}) THEN 0 ELSE r.cost_usd END"
        )
    base = (
        "SELECT '{scope}' AS scope, r.run_id,"
        f" CAST(SUM({cost}) AS VARCHAR(128)) AS amount_usd,"
        f" CAST(SUM(CASE WHEN r.agent_name = :grader THEN {cost} ELSE 0 END)"
        " AS VARCHAR(128)) AS grading_usd"
        f" FROM {schema}.usage_events r"
        " WHERE r.cost_usd IS NOT NULL"
        " AND r.completed_at < CAST(:through AS timestamptz)"
        " AND {scope_filter} GROUP BY r.run_id"
    )
    selects: list[str] = []
    if body.run_ids:
        names = [f"run_{index}" for index in range(len(body.run_ids))]
        parameters += [{"name": n, "value": v} for n, v in zip(names, body.run_ids)]
        run_filter = f"r.run_id IN ({', '.join(f':{name}' for name in names)})"
        selects.append(base.format(scope="run", scope_filter=run_filter))
    if body.project_id is not None:
        parameters += [
            {"name": "project_id", "value": str(body.project_id)},
            {"name": "project_environment", "value": body.project_environment},
        ]
        key_filter = (
            "r.project_id = :project_id"
            " AND r.project_environment = :project_environment"
        )
        selects.append(base.format(scope="key", scope_filter=key_filter))
    return " UNION ALL ".join(selects), parameters


def _read(
    redshift: UsageRedshift, sql: str, parameters: list[dict[str, str]]
) -> list[list[dict[str, Any]]]:
    response = redshift.client.execute_statement(
        WorkgroupName=redshift.workgroup,
        Database=redshift.database,
        SecretArn=redshift.secret_arn,
        Sql=sql,
        StatementName="usage-run-sums",
        Parameters=parameters,
    )
    statement_id = str(response["Id"])
    redshift_statements.wait_for_statement_within(
        redshift.client, statement_id, STATEMENT_SECONDS
    )
    records: list[list[dict[str, Any]]] = []
    args: dict[str, object] = {"Id": statement_id}
    while True:
        page = redshift.client.get_statement_result(**args)
        records += cast(list[list[dict[str, Any]]], page["Records"])
        next_token = page.get("NextToken")
        if next_token is None:
            return records
        args["NextToken"] = next_token


@router.post(
    "/machine/usage/run-sums",
    response_model=RunSumsResponse,
    dependencies=[Depends(machine)],
)
async def run_sums(body: RunSumsRequest, request: Request) -> RunSumsResponse:
    """Billed model dollars for each run, and on the project's key by run."""
    budgets = cast(BudgetService, request.app.state.budget_service)
    exported = await budgets.data_through()
    if exported is None:
        raise HTTPException(503, "Gateway usage is unavailable")
    through = min(utc_now() - CUTOFF_LAG, exported)
    redshift = cast(UsageRedshift | None, request.app.state.usage_redshift)
    if redshift is None:
        raise HTTPException(503, "Gateway usage is unavailable")
    sql, parameters = _statement(redshift.schema, body, ignored_cost_models(), through)
    records: list[list[dict[str, Any]]] = []
    if sql:
        try:
            records = await asyncio.to_thread(_read, redshift, sql, parameters)
        except (
            redshift_statements.RedshiftStatementError,
            BotoCoreError,
            ClientError,
        ) as exc:
            raise HTTPException(503, "Gateway usage is unavailable") from exc
    by_run = {
        run_id: RunSum(run_id=run_id, amount_usd="0", grading_usd="0")
        for run_id in body.run_ids
    }
    project_key: list[RunSum] = []
    for scope, run_id, amount, grading in records:
        row = RunSum(
            run_id=cast(str | None, run_id.get("stringValue")),
            amount_usd=cast(str, amount["stringValue"]),
            grading_usd=cast(str, grading["stringValue"]),
        )
        if scope["stringValue"] == "run" and row.run_id is not None:
            by_run[row.run_id] = row
        else:
            project_key.append(row)
    return RunSumsResponse(
        runs=list(by_run.values()),
        project_key=project_key,
        through=through,
    )
