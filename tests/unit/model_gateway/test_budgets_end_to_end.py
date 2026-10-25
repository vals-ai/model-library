"""A spend snapshot moves from control to two query workers before /query is gated."""

import io
import json
from datetime import UTC, datetime, timedelta, tzinfo
from decimal import Decimal
from unittest.mock import patch

import httpx
import pytest
from botocore.exceptions import ClientError

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.routes.budgets as budget_routes
import model_gateway.routes.query as query_routes
from model_gateway.budgets.bumps import BumpSnapshot
from model_gateway.budgets.schema import (
    BudgetPolicy,
    PeriodSpend,
    SpendSnapshot,
    period_bounds,
)
from model_gateway.budgets.service import BudgetService
from model_gateway.runtime_config import (
    AppConfigAgentClient,
    RuntimeConfigManager,
    RuntimeConfigState,
)
from model_library.base import QueryResult
from tests.unit.model_gateway._support import HEADERS, _make_client


class SharedRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.reads = 0

    async def get(self, key: str) -> bytes | None:
        self.reads += 1
        return self.values.get(key)

    async def eval(
        self, script: str, keys: int, key: str, revision: int, payload: bytes
    ) -> int:
        assert keys == 1
        previous = self.values.get(key)
        if (
            previous is not None
            and SpendSnapshot.model_validate_json(previous).revision >= revision
        ):
            return 0
        self.values[key] = payload
        return 1


class SnapshotS3:
    def __init__(self) -> None:
        self.body: bytes | None = None
        self.fail = False

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert (Bucket, Key) == ("landing", "gateway-usage/dev/spend/current.json")
        if self.fail:
            raise RuntimeError("snapshot read failed")
        if self.body is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.body)}


class Ledger:
    enabled = True

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def write_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class Provider:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls

    async def query(self, inputs: object, **kwargs: object) -> QueryResult:
        self.calls.append(inputs)
        return QueryResult(output_text="served", history=[])


def snapshot(now: datetime, revision: int, spend: dict[str, str]) -> SpendSnapshot:
    return SpendSnapshot(
        schema_version=1,
        stage="dev",
        revision=revision,
        generated_at=now,
        data_through=now - timedelta(minutes=1),
        periods={
            period: PeriodSpend(
                start=period_bounds(period, now)[0],
                end=period_bounds(period, now)[1],
                spend_usd={email: Decimal(value) for email, value in spend.items()},
            )
            for period in ("daily", "weekly", "monthly")
        },
    )


@pytest.mark.asyncio
async def test_control_publication_gates_two_workers_and_preserves_last_known_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo) -> datetime:
            return now

    redis = SharedRedis()
    s3 = SnapshotS3()
    control = BudgetService(
        stage="dev",
        redis=redis,
        bucket="landing",
        key="gateway-usage/dev/spend/current.json",
    )
    control._s3 = s3
    workers = [BudgetService(stage="dev", redis=redis) for _ in range(2)]
    ledger = Ledger()
    provider_calls: list[object] = []
    monkeypatch.setattr(
        model_helpers,
        "get_registry_model",
        lambda model, config: Provider(provider_calls),
    )
    with patch.object(gateway_app, "create_usage_ledger_from_env", return_value=ledger):
        clients = [_make_client() for _ in workers]
        query_only = _make_client(runtime_role="query")
        control_only = _make_client(runtime_role="control")
    monkeypatch.setattr(budget_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(query_routes, "datetime", FixedDatetime)
    known_empty_bumps = BumpSnapshot(stage="dev", revision=0, bumps={})
    for client in [*clients, control_only]:
        client.app.state.bump_service.snapshot = known_empty_bumps
    for client, service in zip(clients, workers, strict=True):
        client.app.state.budget_service = service
        client.app.state.runtime_config_state.install(
            budgets=BudgetPolicy(
                period="monthly",
                default_limit_usd=Decimal("1E+4"),
                email_overrides={" Override@Example.com ": Decimal("2.5E+1")},
            )
        )
    control_only.app.state.budget_service = control
    control_only.app.state.runtime_config_state.install(
        budgets=BudgetPolicy(
            period="monthly",
            default_limit_usd=Decimal("1E+4"),
            email_overrides={"override@example.com": Decimal("2.5E+1")},
        ),
        admin_api_key_names=frozenset({"test"}),
    )

    async def ask(index: int, email: str | None) -> httpx.Response:
        body: dict[str, object] = {
            "model": "openai/gpt-4.1-mini",
            "inputs": [{"kind": "text", "text": "hello"}],
        }
        if email is not None:
            body["identity"] = {"email": email}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=clients[index].app),
            base_url="http://test",
        ) as http:
            return await http.post("/query", json=body, headers=HEADERS)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=clients[0].app), base_url="http://test"
    ) as http:
        assert (await http.get("/admin/budgets", headers=HEADERS)).status_code == 403
    clients[0].app.state.runtime_config_state.install(
        admin_api_key_names=frozenset({"test"})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=clients[0].app), base_url="http://test"
    ) as http:
        unavailable = await http.get("/admin/budgets", headers=HEADERS)
    assert unavailable.status_code == 200
    assert unavailable.json()["spend_available"] is False
    assert unavailable.json()["emails"] == [
        {
            "email": "override@example.com",
            "spent_usd": None,
            "limit_usd": "25",
            "limit_source": "override",
            "blocked": False,
        }
    ]

    # No published spend snapshot means requests proceed; a missing email still uses the ledger.
    assert (await ask(0, "default@example.com")).status_code == 200
    assert (await ask(1, None)).status_code == 200
    assert len(ledger.events) == 2
    await control.refresh_control_once()
    s3.body = (
        snapshot(
            now, 11, {"default@example.com": "10000", "override@example.com": "25"}
        )
        .model_dump_json()
        .encode()
    )
    await control.refresh_control_once()
    for worker in workers:
        await worker.refresh_query_once()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=query_only.app), base_url="http://test"
    ) as http:
        assert (await http.get("/admin/budgets", headers=HEADERS)).status_code == 404
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=control_only.app), base_url="http://test"
    ) as http:
        assert (await http.get("/admin/budgets", headers=HEADERS)).status_code == 200
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=clients[0].app), base_url="http://test"
    ) as http:
        control_view = (await http.get("/admin/budgets", headers=HEADERS)).json()
    assert control_view["stage"] == "dev"
    assert control_view["spend_available"] is True
    assert control_view["snapshot_revision"] == 11
    assert control_view["default_limit_usd"] == "10000"
    assert control_view["emails"] == [
        {
            "email": "default@example.com",
            "spent_usd": "10000",
            "limit_usd": "10000",
            "limit_source": "default",
            "blocked": True,
        },
        {
            "email": "override@example.com",
            "spent_usd": "25",
            "limit_usd": "25",
            "limit_source": "override",
            "blocked": True,
        },
    ]
    reads_before = redis.reads
    denied = await ask(0, "DEFAULT@EXAMPLE.COM")
    override = await ask(1, " override@example.com ")
    assert [denied.status_code, override.status_code] == [402, 402]
    assert denied.json()["code"] == override.json()["code"] == "budget_exceeded"
    assert redis.reads == reads_before
    assert len(provider_calls) == len(ledger.events) == 2
    assert (await ask(1, None)).status_code == 200

    # Duplicate and older publications cannot lift the block.
    s3.body = snapshot(now, 11, {}).model_dump_json().encode()
    await control.refresh_control_once()
    for worker in workers:
        await worker.refresh_query_once()
    assert (await ask(0, "default@example.com")).status_code == 402
    # An older publication cannot lift the block, and failed refresh retains current spend.
    s3.body = snapshot(now, 10, {}).model_dump_json().encode()
    await control.refresh_control_once()
    for worker in workers:
        await worker.refresh_query_once()
    assert (await ask(1, "default@example.com")).status_code == 402
    s3.fail = True
    with pytest.raises(RuntimeError, match="snapshot read failed"):
        await control.refresh_control_once()
    assert (await ask(0, "default@example.com")).status_code == 402

    # A live policy edit applies without a warehouse refresh, including a changed period.
    for client in clients:
        client.app.state.runtime_config_state.install(
            budgets=BudgetPolicy(
                period="daily", default_limit_usd=Decimal("10001"), email_overrides={}
            )
        )
    assert (await ask(0, "default@example.com")).status_code == 200
    for client in clients:
        client.app.state.runtime_config_state.install(
            budgets=BudgetPolicy(
                period="monthly", default_limit_usd=Decimal("10000"), email_overrides={}
            )
        )
    assert (await ask(0, "default@example.com")).status_code == 402

    # The old month cannot block a new month before its snapshot arrives.
    next_month = period_bounds("monthly", now)[1]
    assert not workers[0].blocked(
        BudgetPolicy(
            period="monthly", default_limit_usd=Decimal("10000"), email_overrides={}
        ),
        "default@example.com",
        next_month,
    )
    s3.fail = False
    s3.body = (
        snapshot(next_month, 12, {"default@example.com": "10000"})
        .model_dump_json()
        .encode()
    )
    await control.refresh_control_once()
    for worker in workers:
        await worker.refresh_query_once()
    assert workers[0].blocked(
        BudgetPolicy(
            period="monthly", default_limit_usd=Decimal("10000"), email_overrides={}
        ),
        "default@example.com",
        next_month,
    )


@pytest.mark.asyncio
async def test_appconfig_policy_changes_keep_the_last_valid_budget_on_refresh_failure() -> (
    None
):
    from tests.unit.model_gateway.test_runtime_config import (
        MutableAgent,
        _credential_payload,
        _model_payload,
    )

    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Budget", originated_at=now),
        _credential_payload(api_key="test-provider-key"),
    )
    budget_doc = {
        "period": "monthly",
        "default_limit_usd": "10000",
        "email_overrides": {" User@Example.com ": "25"},
    }
    version = 1
    failing = False

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/budgets"):
            if failing:
                return httpx.Response(503)
            return httpx.Response(
                200,
                content=json.dumps(budget_doc).encode(),
                headers={"Configuration-Version": str(version)},
            )
        return agent.transport().handle_request(request)

    source = AppConfigAgentClient(
        base_url="http://127.0.0.1:2772",
        application="model-gateway",
        environment="test",
        access_token="local-token",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    manager = RuntimeConfigManager(
        state=RuntimeConfigState.bootstrap(),
        agent=source,
        model_profile="models",
        credential_profile="credentials",
        pool_profile="pools",
        budget_profile="budgets",
    )
    service = BudgetService(stage="dev", redis=None)
    service.snapshot = snapshot(now, 1, {"user@example.com": "25"})
    try:
        assert (await manager.refresh_once()).failed == frozenset()
        with manager.state.capture() as view:
            assert view.budgets is not None
            assert view.budgets.email_overrides == {"user@example.com": Decimal("25")}
            assert service.blocked(view.budgets, "user@example.com", now)
        budget_doc["email_overrides"] = {"user@example.com": "30"}
        version = 2
        assert (await manager.refresh_once()).failed == frozenset()
        with manager.state.capture() as view:
            assert view.budgets is not None
            assert not service.blocked(view.budgets, "user@example.com", now)
        failing = True
        assert (await manager.refresh_once()).failed == frozenset({"budgets"})
        with manager.state.capture() as view:
            assert view.budgets is not None
            assert view.budgets.email_overrides["user@example.com"] == Decimal("30")
    finally:
        await manager.aclose()
