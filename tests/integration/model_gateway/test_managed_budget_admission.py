"""Managed cap and static-key admission against local PostgreSQL and Redis."""

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from redis.asyncio import Redis

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.request_budget as request_budget
import model_gateway.routes.passthrough as passthrough_routes
from model_gateway.budgets.schema import (
    PeriodSpend, SpendSnapshot, managed_key_spend_id, period_bounds,
)
from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.credentials import create_managed_key_secret
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, KeyLimit, PersonalOwner, ProjectReference, ServiceOwner,
)
from model_library.base.output import QueryResult
from model_library.exceptions import InvalidStructuredOutputError


class CapturedLedger:
    enabled = True

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def write_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class SimulatedProvider:
    delegate = None

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    async def query(self, inputs, **kwargs):
        self.calls += 1
        if self.fail:
            raise InvalidStructuredOutputError(query_result=QueryResult(output_text="partial"))
        return QueryResult(output_text="simulated response")


@pytest.mark.asyncio
async def test_managed_query_gate_and_at_use_attribution(monkeypatch):
    database_url = os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"]
    engine = create_engine(database_url, 4)
    store = ManagedAuthStore(engine)
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    owner = HumanIdentity("https://issuer.example", f"owner-{uuid4()}")
    project = ProjectReference(uuid4(), "organization", "dev")
    other_project = ProjectReference(uuid4(), "organization", "dev")
    uncapped_project = ProjectReference(uuid4(), "organization", "dev")
    key_id, peer_key_id, other_key_id, service_key_id = (uuid4() for _ in range(4))
    secret, verifier = create_managed_key_secret()
    peer_secret, peer_verifier = create_managed_key_secret()
    other_secret, other_verifier = create_managed_key_secret()
    service_secret, service_verifier = create_managed_key_secret()
    service_id = uuid4()
    stage = f"admission-{uuid4()}"
    namespace = f"managed-admission-{uuid4()}"
    owner_key = f"gateway:managed-auth:{namespace}:owner"
    active_key = f"gateway:managed-auth:{namespace}:active"
    owner_token = uuid4().hex
    snapshot_key = f"gateway:budgets:{stage}:managed-spend-v2"
    now = datetime.now(UTC)
    monkeypatch.setenv("MODEL_GATEWAY_AUTH_NAMESPACE", namespace)
    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "combined")
    monkeypatch.setenv("GATEWAY_STAGE", stage)
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_MODE", "disabled")
    monkeypatch.setenv("MODEL_GATEWAY_AUTH_POOL_SIZE", "4")
    monkeypatch.setenv("DESCOPE_PROJECT_ID", "local")
    monkeypatch.setenv("DESCOPE_EXPECTED_ISSUER", owner.issuer)
    monkeypatch.setenv("DESCOPE_ORGANIZATION_ID", "organization")
    monkeypatch.setenv("DESCOPE_MANAGEMENT_CREDENTIAL", "local-only")
    monkeypatch.setenv("DESCOPE_MACHINE_ACCESS_KEY_ID", f"machine-{uuid4()}")
    monkeypatch.setenv("DESCOPE_MACHINE_SERVICE_OWNER_ID", str(uuid4()))

    class Settings:
        def get(self, name: str, default: str = "") -> str:
            return {
                "MODEL_GATEWAY_API_KEYS": json.dumps({"legacy-control": "sk-local"}),
                "MODEL_GATEWAY_HMAC_SECRET": "local-only-secret",
                "OPENAI_API_KEY": "local-provider-key",
            }.get(name, default)

        def unset(self, key: str) -> None:
            pass

    def spend_snapshot(moment: datetime, revision: int, project_spend: Decimal) -> SpendSnapshot:
        return SpendSnapshot(
            schema_version=1, stage=stage, revision=revision, generated_at=moment,
            data_through=moment, periods={
                period: PeriodSpend(
                    start=period_bounds(period, moment)[0],
                    end=period_bounds(period, moment)[1],
                    spend_usd={
                        str(project.project_id): project_spend,
                        str(other_project.project_id): Decimal("10"),
                    },
                ) for period in ("daily", "weekly", "monthly")
            },
            total_spend_usd={
                str(project.project_id): project_spend,
                str(other_project.project_id): Decimal("10"),
            },
        )

    provider = SimulatedProvider()
    ledger = CapturedLedger()
    try:
        assert await redis.set(owner_key, owner_token, nx=True, ex=3600)
        await store.record_human(owner, True, owner)
        await store.create_service(service_id, "project-budgeted-worker", owner)
        for index, (key, target, key_verifier, first_policy) in enumerate((
            (key_id, project, verifier, ("monthly", Decimal("10"))),
            (peer_key_id, project, peer_verifier, None),
            (other_key_id, other_project, other_verifier, ("monthly", Decimal("10"))),
        )):
            request = await store.create_request(
                uuid4(), f"personal-{index}", "test", owner,
                PersonalOwner(owner), None, target, None, None,
            )
            pending = await store.decide_request(
                request.id, owner, key, first_policy, None,
            )
            await store.collect_key(key, pending.revision, key_verifier, owner)
        service_request = await store.create_request(
            uuid4(), "project-budgeted worker", "test", owner,
            ServiceOwner(service_id), None, uncapped_project, None, None,
        )
        service_pending = await store.decide_request(
            service_request.id, owner, service_key_id, ("monthly", Decimal("100")), None,
        )
        await store.collect_key(
            service_key_id, service_pending.revision, service_verifier, owner,
        )
        await redis.set(snapshot_key, spend_snapshot(now, 1, Decimal("9")).model_dump_json())
        with (
            patch.object(gateway_app, "model_library_settings", Settings()),
            patch.object(model_helpers, "get_registry_model", return_value=provider),
        ):
            app = gateway_app.create_app()
            async with app.router.lifespan_context(app):
                publisher = ManagedAuthorityPublisher(
                    engine, redis,
                    HumanReconciler(store, app.state.managed_auth_verifier, owner.issuer),
                    namespace=namespace,
                )
                assert await publisher.publish_once(owner_token)
                app.state.usage_ledger = ledger
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                ) as client:
                    body = {"model": "openai/gpt-4o", "inputs": [{"kind": "text", "text": "hi"}]}

                    async def mint(root_secret: str) -> dict[str, str]:
                        response = await client.post(
                            "/service-auth", headers={"Authorization": f"Bearer {root_secret}"},
                            json={
                                "run_id": f"run-{uuid4()}", "task_id": "task",
                                "attempt_id": "attempt", "allowed_models": ["openai/gpt-4o"],
                                "allowed_operations": ["query"], "ttl_seconds": 300,
                                "identity": {"email": "caller@example.com", "project_id": "forged"},
                            },
                        )
                        assert response.status_code == 200, response.text
                        return {"Authorization": f"Bearer {response.json()['token']}"}

                    headers = await mint(secret)
                    peer_headers = await mint(peer_secret)
                    other_headers = await mint(other_secret)
                    service_headers = await mint(service_secret)
                    success = await client.post("/query", headers=headers, json=body)
                    assert success.status_code == 200, success.text
                    assert success.json()["output_text"] == "simulated response"
                    assert provider.calls == len(ledger.events) == 1
                    event = ledger.events[-1]
                    assert event["managed_key_id"] == str(key_id)
                    assert event["budget_id"] is None
                    assert event["project_id"] == str(project.project_id)
                    assert event["project_org_id"] == project.organization
                    assert event["project_environment"] == project.environment
                    assert event["owner_issuer"] == owner.issuer
                    assert event["owner_subject"] == owner.subject
                    assert event["service_id"] is None
                    assert event["identity_email"] == "caller@example.com"

                    provider.fail = True
                    failed = await client.post("/query", headers=headers, json=body)
                    assert failed.status_code == 200, failed.text
                    assert "error" in failed.json()
                    assert ledger.events[-1]["outcome"] == "error"
                    assert ledger.events[-1]["project_id"] == str(project.project_id)
                    assert ledger.events[-1]["managed_key_id"] == str(key_id)
                    assert provider.calls == len(ledger.events) == 2
                    provider.fail = False

                    await redis.set(
                        snapshot_key, spend_snapshot(now, 2, Decimal("10")).model_dump_json(),
                    )
                    for blocked_headers in (headers, peer_headers, other_headers):
                        denied = await client.post("/query", headers=blocked_headers, json=body)
                        assert denied.status_code == 402, denied.text
                        assert denied.json()["code"] == "budget_exceeded"
                    assert provider.calls == len(ledger.events) == 2

                    pending_increase = await store.create_limit_request(
                        uuid4(), project, "temporary", Decimal("5"), "burst", owner,
                    )
                    assert pending_increase.status == "pending"
                    assert (await client.post("/query", headers=peer_headers, json=body)).status_code == 402
                    policy = await store.get_project_policy(project.project_id)
                    approved = await store.approve_limit_request(
                        pending_increase.id, owner, policy.revision,
                    )
                    assert approved.status == "approved"
                    assert approved.period_start <= datetime.now(UTC) < approved.period_end
                    assert await publisher.publish_once(owner_token)
                    shared = await client.post("/query", headers=peer_headers, json=body)
                    assert shared.status_code == 200, shared.text
                    assert ledger.events[-1]["managed_key_id"] == str(peer_key_id)
                    assert ledger.events[-1]["project_id"] == str(project.project_id)
                    assert (await client.post("/query", headers=headers, json=body)).status_code == 200
                    assert ledger.events[-1]["managed_key_id"] == str(key_id)
                    assert ledger.events[-1]["project_id"] == str(project.project_id)
                    assert (await client.post("/query", headers=other_headers, json=body)).status_code == 402
                    assert provider.calls == len(ledger.events) == 4
                    switched_spend = spend_snapshot(now, 3, Decimal("15"))
                    weekly = switched_spend.periods["weekly"]
                    switched_spend.periods["weekly"] = weekly.model_copy(update={
                        "spend_usd": {
                            str(project.project_id): Decimal("10"),
                            str(other_project.project_id): Decimal("10"),
                        },
                    })
                    switched_spend = switched_spend.model_copy(update={
                        "total_spend_usd": {
                            str(project.project_id): Decimal("15"),
                            str(other_project.project_id): Decimal("10"),
                        },
                    })
                    await redis.set(snapshot_key, switched_spend.model_dump_json())
                    assert (await client.post("/query", headers=peer_headers, json=body)).status_code == 402
                    policy = await store.get_project_policy(project.project_id)
                    switched_policy = await store.update_project_policy(
                        project.project_id, policy.revision, "weekly", Decimal("10"), owner,
                    )
                    assert switched_policy.period == "weekly"
                    assert await publisher.publish_once(owner_token)
                    switched_authority = await app.state.managed_authority.by_key_id(key_id)
                    assert switched_authority.project_limit.period == "weekly"
                    assert switched_authority.project_limit.temporary_increases[0].period_start == approved.period_start
                    assert switched_authority.project_limit.temporary_increases[0].period_end == approved.period_end
                    switched_refused = await client.post("/query", headers=peer_headers, json=body)
                    assert switched_refused.status_code == 402, switched_refused.text
                    assert provider.calls == len(ledger.events) == 4

                    future = approved.period_end + timedelta(seconds=1)
                    await redis.set(
                        snapshot_key, spend_snapshot(future, 4, Decimal("10")).model_dump_json(),
                    )
                    class FutureDatetime(datetime):
                        @classmethod
                        def now(cls, tz=None):
                            return future

                    with patch.object(request_budget, "datetime", FutureDatetime):
                        kept = await client.post("/query", headers=peer_headers, json=body)
                        assert kept.status_code == 200, kept.text
                    assert provider.calls == len(ledger.events) == 5

                    permanent = await store.create_limit_request(
                        uuid4(), project, "permanent", Decimal("5"), "higher base", owner,
                    )
                    policy = await store.get_project_policy(project.project_id)
                    await store.approve_limit_request(permanent.id, owner, policy.revision)
                    raised = await store.get_project_policy(project.project_id)
                    assert (raised.base_limit_usd, raised.total_limit_usd) == (Decimal("10"), Decimal("20"))
                    assert await publisher.publish_once(owner_token)
                    with patch.object(request_budget, "datetime", FutureDatetime):
                        permanent_allowed = await client.post("/query", headers=headers, json=body)
                        assert permanent_allowed.status_code == 200, permanent_allowed.text
                        assert (await client.post("/query", headers=other_headers, json=body)).status_code == 402
                    assert ledger.events[-1]["project_id"] == str(project.project_id)
                    assert ledger.events[-1]["budget_id"] is None

                    await redis.delete(snapshot_key)
                    assert (await client.post("/query", headers=headers, json=body)).status_code == 503
                    await redis.set(snapshot_key, "not a snapshot")
                    assert (await client.post("/query", headers=headers, json=body)).status_code == 503
                    missing = spend_snapshot(now, 6, Decimal("0"))
                    weekly = missing.periods["weekly"]
                    missing.periods["weekly"] = weekly.model_copy(update={
                        "spend_usd": {str(other_project.project_id): Decimal("10")},
                    })
                    await redis.set(snapshot_key, missing.model_dump_json())
                    absent = await client.post("/query", headers=headers, json=body)
                    assert absent.status_code == 200, absent.text
                    service_allowed = await client.post("/query", headers=service_headers, json=body)
                    assert service_allowed.status_code == 200, service_allowed.text
                    event = ledger.events[-1]
                    assert event["managed_key_id"] == str(service_key_id)
                    assert event["budget_id"] is None
                    assert event["project_id"] == str(uncapped_project.project_id)
                    assert event["project_org_id"] == uncapped_project.organization
                    assert event["project_environment"] == uncapped_project.environment
                    assert event["service_id"] == str(service_id)
                    assert event["owner_issuer"] is None
                    assert event["owner_subject"] is None
                    assert event["identity_email"] == "caller@example.com"

                    # The recorded root spend precedes the admin's cap; children share it.
                    root = await store.get_key(key_id)
                    service = await store.get_key(service_key_id)
                    assert root is not None and service is not None
                    root = await store.update_key_limit(
                        key_id, root.revision, KeyLimit("monthly", Decimal("4")), owner,
                    )
                    service = await store.update_key_limit(
                        service_key_id, service.revision, KeyLimit("daily", Decimal("2")), owner,
                    )
                    assert root is not None and service is not None
                    assert await publisher.publish_once(owner_token)
                    root_child_headers = await mint(secret)

                    def capped_snapshot(
                        revision: int, project_spend: Decimal,
                        root_spend: Decimal, service_spend: Decimal,
                    ) -> SpendSnapshot:
                        snapshot = spend_snapshot(now, revision, project_spend)
                        for period in ("daily", "weekly", "monthly"):
                            current = snapshot.periods[period]
                            snapshot.periods[period] = current.model_copy(update={
                                "spend_usd": {
                                    **current.spend_usd,
                                    managed_key_spend_id(key_id): (
                                        root_spend if period == "monthly" else Decimal("30")
                                    ),
                                    managed_key_spend_id(service_key_id): (
                                        service_spend if period == "daily" else Decimal("30")
                                    ),
                                },
                            })
                        return snapshot

                    capped = capped_snapshot(7, Decimal("9"), Decimal("4"), Decimal("2"))
                    await redis.set(snapshot_key, capped.model_dump_json())
                    before = provider.calls
                    for root_headers in (headers, root_child_headers):
                        blocked_root = await client.post("/query", headers=root_headers, json=body)
                        assert blocked_root.status_code == 402, blocked_root.text
                        assert blocked_root.json()["code"] == "budget_exceeded"
                        assert blocked_root.json()["message"] == "Managed key budget exceeded"
                    blocked_service = await client.post("/query", headers=service_headers, json=body)
                    assert blocked_service.status_code == 402, blocked_service.text
                    assert blocked_service.json()["message"] == "Managed key budget exceeded"
                    assert (await client.post("/query", headers=peer_headers, json=body)).status_code == 200
                    assert provider.calls == before + 1

                    project_blocked = capped_snapshot(8, Decimal("20"), Decimal("4"), Decimal("2"))
                    await redis.set(snapshot_key, project_blocked.model_dump_json())
                    for root_headers in (headers, root_child_headers, peer_headers):
                        blocked_project = await client.post("/query", headers=root_headers, json=body)
                        assert blocked_project.status_code == 402, blocked_project.text
                        assert blocked_project.json()["message"] == "Project total budget of $20 is spent"
                    assert provider.calls == before + 1
                    managed_passthrough = await client.post(
                        "/service-auth", headers={"Authorization": f"Bearer {secret}"},
                        json={
                            "run_id": f"managed-passthrough-{uuid4()}", "task_id": "task",
                            "attempt_id": "attempt", "allowed_models": ["openai/gpt-4o"],
                            "allowed_operations": ["passthrough"], "ttl_seconds": 300,
                            "identity": {"email": "caller@example.com"},
                        },
                    )
                    assert managed_passthrough.status_code == 200, managed_passthrough.text
                    managed_blocked = await client.post(
                        "/passthrough/openai/v1/responses",
                        headers={"Authorization": f"Bearer {managed_passthrough.json()['token']}"},
                        json={"model": "gpt-4o", "input": "hello"},
                    )
                    assert managed_blocked.status_code == 402, managed_blocked.text
                    assert provider.calls == len(ledger.events) == before + 1

                    # Monthly root and daily service caps ignore spend in the other periods.
                    admitted = capped_snapshot(9, Decimal("9"), Decimal("3"), Decimal("1"))
                    await redis.set(snapshot_key, admitted.model_dump_json())
                    for admitted_headers in (headers, root_child_headers, peer_headers, service_headers):
                        response = await client.post("/query", headers=admitted_headers, json=body)
                        assert response.status_code == 200, response.text
                    assert provider.calls == before + 5

                    no_key_counter = capped_snapshot(10, Decimal("9"), Decimal("3"), Decimal("1"))
                    monthly = no_key_counter.periods["monthly"]
                    no_key_counter.periods["monthly"] = monthly.model_copy(update={
                        "spend_usd": {
                            identifier: spend for identifier, spend in monthly.spend_usd.items()
                            if identifier != managed_key_spend_id(key_id)
                        },
                    })
                    await redis.set(snapshot_key, no_key_counter.model_dump_json())
                    assert (await client.post("/query", headers=headers, json=body)).status_code == 200
                    await redis.delete(snapshot_key)
                    assert (await client.post("/query", headers=service_headers, json=body)).status_code == 503
                    await redis.set(snapshot_key, "not a snapshot")
                    assert (await client.post("/query", headers=headers, json=body)).status_code == 503

                    await redis.set(snapshot_key, capped.model_dump_json())
                    root = await store.update_key_limit(key_id, root.revision, None, owner)
                    assert root is not None and root.key_limit is None
                    assert await publisher.publish_once(owner_token)
                    assert (await client.post("/query", headers=root_child_headers, json=body)).status_code == 200
                    assert (await client.post("/query", headers=service_headers, json=body)).status_code == 402
                    service = await store.update_key_limit(service_key_id, service.revision, None, owner)
                    assert service is not None and service.key_limit is None
                    assert await publisher.publish_once(owner_token)
                    await redis.delete(snapshot_key)
                    assert (await client.post("/query", headers=service_headers, json=body)).status_code == 503
                    await redis.set(snapshot_key, project_blocked.model_dump_json())
                    static_headers = {"Authorization": "Bearer sk-local"}
                    static_body = {
                        **body,
                        "identity": {"email": "caller@example.com"},
                    }
                    static_response = await client.post(
                        "/query", headers=static_headers, json=static_body,
                    )
                    assert static_response.status_code == 200, static_response.text
                    assert ledger.events[-1]["identity_email"] == "caller@example.com"
                    assert ledger.events[-1].get("managed_key_id") is None
                    assert (await client.get("/budget", headers=static_headers)).status_code == 404

                    legacy_mint = await client.post(
                        "/service-auth", headers=static_headers,
                        json={
                            "run_id": f"legacy-{uuid4()}", "task_id": "task",
                            "attempt_id": "attempt", "allowed_models": ["openai/gpt-4o"],
                            "allowed_operations": ["passthrough"], "ttl_seconds": 300,
                            "identity": {"email": "caller@example.com"},
                        },
                    )
                    assert legacy_mint.status_code == 200, legacy_mint.text
                    await app.state.passthrough_client.aclose()
                    app.state.passthrough_client = httpx.AsyncClient(
                        transport=httpx.MockTransport(
                            lambda request: httpx.Response(
                                200, json={"id": "response", "usage": {
                                    "input_tokens": 1, "output_tokens": 1,
                                }},
                            )
                        ),
                    )
                    with patch.object(passthrough_routes, "model_library_settings", Settings()):
                        passed = await client.post(
                            "/passthrough/openai/v1/responses",
                            headers={"Authorization": f"Bearer {legacy_mint.json()['token']}"},
                            json={"model": "gpt-4o", "input": "hello"},
                        )
                    assert passed.status_code == 200, passed.text
                    assert ledger.events[-1]["identity_email"] == "caller@example.com"
                    assert ledger.events[-1].get("managed_key_id") is None
    finally:
        await redis.delete(snapshot_key, owner_key, active_key)
        await redis.aclose()
        await engine.dispose()
