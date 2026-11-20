"""Model dollars per run and on one project key, read from Redshift `usage_events`."""

from datetime import UTC, datetime
from types import ModuleType
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from model_gateway.usage_ledger.redshift_schema import GRADER_AGENT_NAME

PATH = "/admin/managed-auth/machine/usage/run-sums"
TOKEN = {"Authorization": "Bearer machine-access-key"}
PROJECT = "7d3c2a3e-4c1b-4f43-9a0e-0c6d1f3a9b21"
NOW = datetime(2026, 11, 12, 1, 0, tzinfo=UTC)

pytestmark = pytest.mark.unit


class FakeRedshift:
    def __init__(
        self, records: list[list[dict[str, Any]]], status: str = "FINISHED"
    ) -> None:
        self.records = records
        self.status = status
        self.statements: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.execute_error: Exception | None = None

    def execute_statement(self, **kwargs: Any) -> dict[str, Any]:
        self.statements.append(kwargs)
        if self.execute_error is not None:
            raise self.execute_error
        return {"Id": "statement-1"}

    def describe_statement(self, **_: Any) -> dict[str, Any]:
        return {"Status": self.status, "Error": "boom"}

    def cancel_statement(self, **kwargs: Any) -> dict[str, Any]:
        self.cancelled.append(kwargs["Id"])
        return {"Status": True}

    def get_statement_result(self, **_: Any) -> dict[str, Any]:
        return {"Records": self.records}


class FakeBudgets:
    def __init__(self, data_through: datetime | None = NOW) -> None:
        self.through = data_through

    async def data_through(self) -> datetime | None:
        return self.through


class FakeMachineVerifier:
    async def verify(self, token: str) -> object:
        if token != "machine-access-key":
            raise HTTPException(status_code=401, detail="Invalid machine key")
        return object()


def _record(
    scope: str, run_id: str | None, amount: str, grading: str = "0"
) -> list[dict[str, Any]]:
    return [
        {"stringValue": scope},
        {"isNull": True} if run_id is None else {"stringValue": run_id},
        {"stringValue": amount},
        {"stringValue": grading},
    ]


def _routes() -> ModuleType:
    import model_gateway.routes.run_sums as run_sums

    return run_sums


def _client(
    monkeypatch: pytest.MonkeyPatch,
    redshift: FakeRedshift,
    budgets: FakeBudgets | None = None,
) -> TestClient:
    run_sums = _routes()
    monkeypatch.setattr(run_sums, "utc_now", lambda: NOW)
    monkeypatch.setattr(run_sums, "ignored_cost_models", lambda: ("internal/free",))
    app = FastAPI()
    app.include_router(run_sums.router)
    app.state.managed_machine_verifier = FakeMachineVerifier()
    app.state.budget_service = budgets or FakeBudgets()
    app.state.usage_redshift = run_sums.UsageRedshift(
        client=redshift,
        workgroup="wg",
        database="db",
        secret_arn="arn:secret",
        schema="gateway_usage",
    )
    return TestClient(app)


def _parameters(redshift: FakeRedshift) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in redshift.statements[0].get("Parameters", [])
    }


def test_rejects_missing_machine_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, FakeRedshift([]))

    assert (
        client.post(
            PATH, json={"run_ids": ["r1"], "project_environment": "dev"}
        ).status_code
        == 401
    )


def test_sums_each_run_and_the_project_key_by_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redshift = FakeRedshift(
        [
            _record("run", "r1", "1.50", "0.40"),
            _record("key", None, "0.25"),
            _record("key", "r9", "2", "0.5"),
        ]
    )
    client = _client(monkeypatch, redshift)

    response = client.post(
        PATH,
        json={
            "run_ids": ["r1", "r2"],
            "project_id": PROJECT,
            "project_environment": "dev",
        },
        headers=TOKEN,
    )

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {"run_id": "r1", "amount_usd": "1.50", "grading_usd": "0.40"},
            {"run_id": "r2", "amount_usd": "0", "grading_usd": "0"},
        ],
        "project_key": [
            {"run_id": None, "amount_usd": "0.25", "grading_usd": "0"},
            {"run_id": "r9", "amount_usd": "2", "grading_usd": "0.5"},
        ],
        "through": "2026-11-12T00:45:00Z",
    }


def test_statement_reads_billed_cost_up_to_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redshift = FakeRedshift([])
    client = _client(monkeypatch, redshift)

    client.post(
        PATH,
        json={
            "run_ids": ["r1", "r2"],
            "project_id": PROJECT,
            "project_environment": "dev",
        },
        headers=TOKEN,
    )

    statement = redshift.statements[0]
    sql = " ".join(statement["Sql"].split())
    assert (
        statement["WorkgroupName"],
        statement["Database"],
        statement["SecretArn"],
    ) == ("wg", "db", "arn:secret")
    assert "gateway_usage.usage_events" in sql
    assert (
        "SUM(CASE WHEN COALESCE(r.served_model_key, r.requested_model_key)"
        " IN (:ignored_0) THEN 0 ELSE r.cost_usd END)"
    ) in sql
    assert (
        "SUM(CASE WHEN r.agent_name = :grader THEN CASE WHEN"
        " COALESCE(r.served_model_key, r.requested_model_key)"
        " IN (:ignored_0) THEN 0 ELSE r.cost_usd END ELSE 0 END)"
    ) in sql
    assert "r.cost_usd IS NOT NULL" in sql
    assert sql.count("r.completed_at < CAST(:through AS timestamptz)") == 2
    assert "r.run_id IN (:run_0, :run_1)" in sql
    assert (
        "r.project_id = :project_id AND r.project_environment = :project_environment"
        in sql
    )
    assert _parameters(redshift) == {
        "through": "2026-11-12T00:45:00+00:00",
        "grader": GRADER_AGENT_NAME,
        "ignored_0": "internal/free",
        "run_0": "r1",
        "run_1": "r2",
        "project_id": PROJECT,
        "project_environment": "dev",
    }


def test_without_a_project_reads_runs_only(monkeypatch: pytest.MonkeyPatch) -> None:
    redshift = FakeRedshift([_record("run", "r1", "3")])
    client = _client(monkeypatch, redshift)

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.json()["project_key"] == []
    assert "project_id" not in _parameters(redshift)


def test_cutoff_stops_at_the_last_completed_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = datetime(2026, 11, 12, 0, 30, tzinfo=UTC)
    redshift = FakeRedshift([])
    client = _client(monkeypatch, redshift, FakeBudgets(exported))

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.json()["through"] == "2026-11-12T00:30:00Z"
    assert _parameters(redshift)["through"] == "2026-11-12T00:30:00+00:00"


def test_no_completed_export_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    redshift = FakeRedshift([])
    client = _client(monkeypatch, redshift, FakeBudgets(None))

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.status_code == 503
    assert redshift.statements == []


def test_refuses_more_than_the_run_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, FakeRedshift([]))
    run_ids = [f"r{index}" for index in range(_routes().RUN_LIMIT + 1)]

    response = client.post(
        PATH, json={"run_ids": run_ids, "project_environment": "dev"}, headers=TOKEN
    )

    assert response.status_code == 422


def test_failed_statement_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, FakeRedshift([], status="FAILED"))

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.status_code == 503


def test_stuck_statement_is_cancelled_and_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model_gateway.usage_ledger import redshift_statements

    clock = iter(range(0, 100))
    monkeypatch.setattr(redshift_statements.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(redshift_statements.time, "sleep", lambda _: None)
    redshift = FakeRedshift([], status="STARTED")
    client = _client(monkeypatch, redshift)

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.status_code == 503
    assert redshift.cancelled == ["statement-1"]


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "ThrottlingException"}}, "ExecuteStatement"),
        EndpointConnectionError(endpoint_url="https://redshift-data.test"),
    ],
    ids=["client-error", "endpoint-unreachable"],
)
def test_aws_errors_return_503(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    redshift = FakeRedshift([])
    redshift.execute_error = error
    client = _client(monkeypatch, redshift)

    response = client.post(
        PATH, json={"run_ids": ["r1"], "project_environment": "dev"}, headers=TOKEN
    )

    assert response.status_code == 503


def test_grading_is_a_part_of_the_run_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    redshift = FakeRedshift([_record("run", "r1", "1.50", "0.40")])
    client = _client(monkeypatch, redshift)

    response = client.post(
        PATH,
        json={"run_ids": ["r1", "r2"], "project_environment": "dev"},
        headers=TOKEN,
    )

    sql = " ".join(redshift.statements[0]["Sql"].split())
    grading = sql.split("AS amount_usd,")[1].split("AS grading_usd")[0]
    assert "FROM" not in grading
    assert grading.count("SUM(") == 1
    assert response.json()["runs"] == [
        {"run_id": "r1", "amount_usd": "1.50", "grading_usd": "0.40"},
        {"run_id": "r2", "amount_usd": "0", "grading_usd": "0"},
    ]
