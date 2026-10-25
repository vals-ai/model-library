"""Temporary grants from admin S3 commit through Redis to query-time enforcement."""

import asyncio
import io
import json
from datetime import UTC, datetime, timedelta, tzinfo
from decimal import Decimal
from types import SimpleNamespace
from threading import Barrier, Lock
from typing import cast
from unittest.mock import patch

import httpx
import pytest
from botocore.exceptions import ClientError

import model_gateway.app as gateway_app
import model_gateway.budgets.bump_service as bump_service_module
import model_gateway.model_helpers as model_helpers
import model_gateway.routes.budgets as budget_routes
import model_gateway.routes.query as query_routes
from model_gateway.budgets.bump_service import BumpService
from model_gateway.budgets.bumps import (
    BudgetGrant,
    BumpDocument,
    BumpSnapshot,
    LegacyBumpSnapshot,
    LegacyBudgetGrant,
    effective_limit,
)
from model_gateway.budgets.override_service import OverrideService
from model_gateway.budgets.schema import BudgetPolicy, period_bounds
from model_gateway.budgets.service import BudgetService
from model_gateway.runtime_config import AppConfigAgentClient
from tests.unit.model_gateway._support import HEADERS, _make_client
from tests.unit.model_gateway.test_budgets_end_to_end import Ledger, Provider, snapshot


class SharedRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def eval(
        self, script: str, keys: int, key: str, revision: int, payload: bytes
    ) -> int:
        assert keys == 1
        previous = self.values.get(key)
        if previous is not None and json.loads(previous)["revision"] >= revision:
            return 0
        self.values[key] = payload
        return 1


class BumpS3:
    def __init__(self) -> None:
        self.body: bytes | None = None
        self.etag = '"0"'
        self.fail = False
        self.conflict = False

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert (Bucket, Key) == ("landing", "gateway-usage/dev/budgets/bumps.json")
        if self.fail:
            raise RuntimeError("S3 unavailable")
        if self.body is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.body), "ETag": self.etag}

    def put_object(self, **kwargs: object) -> None:
        assert (kwargs["Bucket"], kwargs["Key"]) == (
            "landing",
            "gateway-usage/dev/budgets/bumps.json",
        )
        condition = (
            kwargs.get("IfMatch") == self.etag
            if self.body is not None
            else kwargs.get("IfNoneMatch") == "*"
        )
        if not condition or self.conflict:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "conflict"}},
                "PutObject",
            )
        self.body = cast(bytes, kwargs["Body"])
        self.etag = f'"{BumpDocument.model_validate_json(self.body).revision}"'


@pytest.mark.asyncio
async def test_bump_commit_distribution_conflict_and_period_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [datetime(2026, 10, 22, 12, tzinfo=UTC)]

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo) -> datetime:
            return clock[0]

    redis = SharedRedis()
    s3 = BumpS3()
    control_bumps = BumpService(
        "dev", redis, "landing", "gateway-usage/dev/budgets/bumps.json"
    )
    control_bumps._s3 = s3
    worker_bumps = BumpService("dev", redis)
    spend = BudgetService(stage="dev", redis=None)
    spend.bump_service = worker_bumps
    spend.snapshot = snapshot(clock[0], 1, {"alice@example.com": "100"})
    policy = BudgetPolicy(
        period="daily", default_limit_usd=Decimal("100"), email_overrides={}
    )
    calls: list[object] = []
    monkeypatch.setattr(
        model_helpers, "get_registry_model", lambda model, config: Provider(calls)
    )
    with patch.object(
        gateway_app, "create_usage_ledger_from_env", return_value=Ledger()
    ):
        control = _make_client(runtime_role="control")
        worker = _make_client(runtime_role="query")
    monkeypatch.setattr(budget_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(query_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(bump_service_module, "datetime", FixedDatetime)
    control.app.state.budget_service = spend
    control.app.state.bump_service = control_bumps
    control.app.state.runtime_config_state.install(
        budgets=policy, admin_api_key_names=frozenset({"test"})
    )
    worker.app.state.budget_service = spend
    worker.app.state.runtime_config_state.install(budgets=policy)
    start, end = period_bounds("daily", clock[0])
    body = {
        "email": " ALICE@EXAMPLE.COM ",
        "expected_revision": 0,
        "period": "daily",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "actor": {"user_id": "operator-1", "email": "operator@example.com"},
        "reason": "On-call capacity",
    }
    request = {
        "model": "openai/gpt-4.1-mini",
        "inputs": [{"kind": "text", "text": "hello"}],
        "identity": {"email": "alice@example.com"},
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control.app), base_url="http://test"
        ) as admin,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=worker.app), base_url="http://test"
        ) as query,
    ):
        s3.fail = True
        unknown = await admin.get("/admin/budgets", headers=HEADERS)
        assert unknown.status_code == 503
        assert unknown.json()["code"] == "budget_bumps_unavailable"
        paired_unknown = await admin.get("/admin/budgets/bumps", headers=HEADERS)
        assert paired_unknown.status_code == 503
        assert paired_unknown.json()["code"] == "budget_bumps_unavailable"
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 200
        s3.fail = False
        initial = (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()
        assert {
            key: initial[key]
            for key in (
                "stage",
                "period",
                "period_start",
                "period_end",
                "revision",
                "emails",
            )
        } == {
            "stage": "dev",
            "period": "daily",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "revision": 0,
            "emails": [],
        }
        assert "available" not in initial
        assert initial["budget"]["period_start"] == initial["period_start"]
        assert initial["budget"]["emails"][0]["limit_usd"] == "100"
        assert not spend.blocked(policy, "alice@example.com", clock[0])
        await control_bumps.refresh_control_once()
        await worker_bumps.refresh_query_once()
        known_empty = (await admin.get("/admin/budgets", headers=HEADERS)).json()
        assert known_empty["emails"][0]["limit_usd"] == "100"
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 402
        for method in ("post", "put", "delete"):
            payload = (
                {**body, "limit_usd": "150", "expected_base_limit_usd": "100"}
                if method == "put"
                else body
            )
            for missing in ({"reason": " "}, {"reason": None}, {"actor": None}):
                invalid = await admin.request(
                    method.upper(),
                    "/admin/budgets/bumps",
                    json={**payload, **missing},
                    headers=HEADERS,
                )
                assert invalid.status_code == 400
                assert invalid.json()["code"] == "invalid_request"
            for key in ("reason", "actor"):
                missing_field = {
                    name: value for name, value in payload.items() if name != key
                }
                missing_response = await admin.request(
                    method.upper(),
                    "/admin/budgets/bumps",
                    json=missing_field,
                    headers=HEADERS,
                )
                assert missing_response.status_code == 400
                assert missing_response.json()["code"] == "invalid_request"
        assert s3.body is None
        first = await admin.post("/admin/budgets/bumps", json=body, headers=HEADERS)
        assert first.status_code == 200
        assert first.json()["limit_usd"] == "5100"
        paired = (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()
        assert paired["revision"] == first.json()["revision"]
        assert (
            paired["emails"][0]["limit_usd"]
            == paired["budget"]["emails"][0]["limit_usd"]
            == "5100"
        )
        assert (
            await admin.post("/admin/budgets/bumps", json=body, headers=HEADERS)
        ).status_code == 409
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 402
        await control_bumps.refresh_control_once()
        await worker_bumps.refresh_query_once()
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 200
        assert len(calls) == 2
        first_payload = redis.values[worker_bumps.redis_key]
        second = await admin.post(
            "/admin/budgets/bumps",
            json={**body, "expected_revision": 1},
            headers=HEADERS,
        )
        assert second.status_code == 200
        assert second.json()["bump_usd"] == "10000"
        await control_bumps.refresh_control_once()
        await worker_bumps.refresh_query_once()
        redis.values[worker_bumps.redis_key] = first_payload
        await worker_bumps.refresh_query_once()
        assert worker_bumps.snapshot is not None and worker_bumps.snapshot.revision == 2
        s3.conflict = True
        assert (
            await admin.post(
                "/admin/budgets/bumps",
                json={**body, "expected_revision": 2},
                headers=HEADERS,
            )
        ).status_code == 409
        assert len(BumpDocument.model_validate_json(cast(bytes, s3.body)).events) == 2
        s3.conflict = False
        view = (await admin.get("/admin/budgets", headers=HEADERS)).json()
        assert view["emails"][0]["limit_usd"] == "10100"
        s3.fail = True
        assert (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()[
            "revision"
        ] == 2
        retained = (await admin.get("/admin/budgets", headers=HEADERS)).json()
        assert retained["emails"][0]["limit_usd"] == "10100"
        assert (
            await admin.get(
                "/admin/budgets/bumps/history",
                params={
                    "email": "alice@example.com",
                    "period": "daily",
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                },
                headers=HEADERS,
            )
        ).status_code == 503
        clock[0] += timedelta(days=1)
        new_start, new_end = period_bounds("daily", clock[0])
        spend.snapshot = snapshot(clock[0], 2, {"alice@example.com": "100"})
        assert worker_bumps.snapshot is not None and worker_bumps.snapshot.revision == 2
        assert (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()[
            "emails"
        ] == []
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 402
        assert (
            await admin.post(
                "/admin/budgets/bumps",
                json={**body, "expected_revision": 2},
                headers=HEADERS,
            )
        ).status_code == 409
        s3.fail = False
        renewed = await admin.post(
            "/admin/budgets/bumps",
            json={
                **body,
                "expected_revision": 2,
                "period_start": new_start.isoformat(),
                "period_end": new_end.isoformat(),
            },
            headers=HEADERS,
        )
        assert renewed.status_code == 200
        assert renewed.json()["bump_usd"] == "5000"
        assert renewed.json()["revision"] == 3
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 402
        await control_bumps.refresh_control_once()
        await worker_bumps.refresh_query_once()
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 200
        assert len(calls) == 3
        bump_only = await admin.post(
            "/admin/budgets/bumps",
            json={
                **body,
                "email": " BOB@EXAMPLE.COM ",
                "expected_revision": 3,
                "period_start": new_start.isoformat(),
                "period_end": new_end.isoformat(),
            },
            headers=HEADERS,
        )
        assert bump_only.status_code == 200
        assert bump_only.json()["email"] == "bob@example.com"
        assert bump_only.json()["revision"] == 4
        paired_bump_only = (
            await admin.get("/admin/budgets/bumps", headers=HEADERS)
        ).json()
        legacy_bump_only = (await admin.get("/admin/budgets", headers=HEADERS)).json()
        assert paired_bump_only["revision"] == 4
        assert [row["email"] for row in paired_bump_only["emails"]] == [
            "alice@example.com",
            "bob@example.com",
        ]
        assert [row["email"] for row in paired_bump_only["budget"]["emails"]] == [
            "alice@example.com",
            "bob@example.com",
        ]
        assert [row["email"] for row in legacy_bump_only["emails"]] == [
            "alice@example.com",
            "bob@example.com",
        ]
        assert paired_bump_only["emails"][1] == {
            "email": "bob@example.com",
            "base_limit_usd": "100",
            "bump_usd": "5000",
            "limit_usd": "5100",
        }
        assert (
            paired_bump_only["budget"]["emails"][1]
            == legacy_bump_only["emails"][1]
            == {
                "email": "bob@example.com",
                "spent_usd": "0",
                "limit_usd": "5100",
                "limit_source": "default",
                "blocked": False,
            }
        )
        current_history_query = {
            "email": "alice@example.com",
            "period": "daily",
            "period_start": new_start.isoformat(),
            "period_end": new_end.isoformat(),
        }
        old_history = (
            await admin.get(
                "/admin/budgets/bumps/history",
                params={
                    **current_history_query,
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                },
                headers=HEADERS,
            )
        ).json()
        assert [event["revision"] for event in old_history["events"]] == [1, 2]
        assert old_history["events"][0]["actor"] == body["actor"]
        assert old_history["events"][0]["reason"] == body["reason"]
        assert old_history["events"][0]["base_limit_usd"] == "100"
        assert old_history["events"][0]["old_bump_usd"] == "0"
        assert old_history["events"][0]["new_bump_usd"] == "5000"
        assert (
            await admin.get(
                "/admin/budgets/bumps/history",
                params={
                    **current_history_query,
                    "period_end": (new_end + timedelta(days=1)).isoformat(),
                },
                headers=HEADERS,
            )
        ).status_code == 400
        assert (
            await admin.put(
                "/admin/budgets/bumps",
                json={
                    **body,
                    "expected_revision": 4,
                    "period_start": new_start.isoformat(),
                    "period_end": new_end.isoformat(),
                    "limit_usd": "99.99",
                    "expected_base_limit_usd": "100",
                },
                headers=HEADERS,
            )
        ).status_code == 400
        custom = await admin.put(
            "/admin/budgets/bumps",
            json={
                **body,
                "expected_revision": 4,
                "period_start": new_start.isoformat(),
                "period_end": new_end.isoformat(),
                "limit_usd": "100.000000000000000000001",
                "expected_base_limit_usd": "100",
                "reason": "Exact extra",
            },
            headers=HEADERS,
        )
        assert custom.status_code == 200
        assert custom.json()["bump_usd"] == "0.000000000000000000001"
        await control_bumps.refresh_control_once()
        assert b'"events"' not in redis.values[worker_bumps.redis_key]
        await worker_bumps.refresh_query_once()
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 200
        reduced = await admin.put(
            "/admin/budgets/bumps",
            json={
                **body,
                "expected_revision": 5,
                "period_start": new_start.isoformat(),
                "period_end": new_end.isoformat(),
                "limit_usd": "100",
                "expected_base_limit_usd": "100",
                "reason": "Reduce capacity",
            },
            headers=HEADERS,
        )
        assert reduced.status_code == 200
        assert reduced.json()["bump_usd"] == "0"
        await control_bumps.refresh_control_once()
        await worker_bumps.refresh_query_once()
        assert (
            await query.post("/query", json=request, headers=HEADERS)
        ).status_code == 402
        removed = await admin.request(
            "DELETE",
            "/admin/budgets/bumps",
            json={
                **body,
                "email": "bob@example.com",
                "expected_revision": 6,
                "period_start": new_start.isoformat(),
                "period_end": new_end.isoformat(),
                "reason": "Remove unused allowance",
            },
            headers=HEADERS,
        )
        assert removed.status_code == 200
        assert removed.json()["limit_usd"] == "100"
        assert (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()[
            "emails"
        ] == []
        current_history = (
            await admin.get(
                "/admin/budgets/bumps/history",
                params=current_history_query,
                headers=HEADERS,
            )
        ).json()
        assert (
            current_history["history_started_at"] == old_history["history_started_at"]
        )
        assert [event["revision"] for event in current_history["events"]] == [3, 5, 6]
        assert [event["new_bump_usd"] for event in current_history["events"]] == [
            "5000",
            "0.000000000000000000001",
            "0",
        ]
        bob_history = (
            await admin.get(
                "/admin/budgets/bumps/history",
                params={
                    **current_history_query,
                    "email": "bob@example.com",
                },
                headers=HEADERS,
            )
        ).json()
        assert [event["revision"] for event in bob_history["events"]] == [4, 7]


@pytest.mark.asyncio
async def test_put_rejects_stale_base_without_reinterpreting_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo) -> datetime:
            return now

    start, end = period_bounds("daily", now)
    s3 = BumpS3()
    initial = BumpDocument(
        stage="dev",
        revision=1,
        history_started_at=now,
        events=[],
        bumps={
            "alice@example.com": BudgetGrant(
                period="daily",
                period_start=start,
                period_end=end,
                bump_usd=Decimal("10"),
            )
        },
    )
    s3.body = initial.model_dump_json().encode()
    s3.etag = '"1"'
    service = BumpService(
        "dev", None, "landing", "gateway-usage/dev/budgets/bumps.json"
    )
    service._s3 = s3
    with patch.object(
        gateway_app, "create_usage_ledger_from_env", return_value=Ledger()
    ):
        control = _make_client(runtime_role="control")
    monkeypatch.setattr(budget_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(bump_service_module, "datetime", FixedDatetime)
    control.app.state.bump_service = service
    control.app.state.budget_service = BudgetService(stage="dev", redis=None)
    policy_200 = BudgetPolicy(
        period="daily", default_limit_usd=Decimal("200"), email_overrides={}
    )
    control.app.state.runtime_config_state.install(
        budgets=policy_200, admin_api_key_names=frozenset({"test"})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=control.app), base_url="http://test"
    ) as admin:
        paired = (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()
        assert paired["revision"] == 1
        assert paired["emails"][0] == {
            "email": "alice@example.com",
            "base_limit_usd": "200",
            "bump_usd": "10",
            "limit_usd": "210",
        }
        assert paired["budget"]["period_start"] == paired["period_start"]
        body = {
            "email": "alice@example.com",
            "expected_revision": paired["revision"],
            "period": paired["period"],
            "period_start": paired["period_start"],
            "period_end": paired["period_end"],
            "limit_usd": "250",
            "expected_base_limit_usd": "200",
            "actor": {"user_id": "operator-1", "email": None},
            "reason": "Capacity",
        }
        for invalid in (None, "-1", "NaN", "Infinity", "1e2", 200):
            payload = {**body, "expected_base_limit_usd": invalid}
            invalid_response = await admin.put(
                "/admin/budgets/bumps", json=payload, headers=HEADERS
            )
            assert invalid_response.status_code == 400
            assert invalid_response.json()["code"] == "invalid_request"
        missing_base = await admin.put(
            "/admin/budgets/bumps",
            json={
                key: value
                for key, value in body.items()
                if key != "expected_base_limit_usd"
            },
            headers=HEADERS,
        )
        assert missing_base.status_code == 400
        assert missing_base.json()["code"] == "invalid_request"
        assert s3.body == initial.model_dump_json().encode()
        control.app.state.runtime_config_state.install(
            budgets=BudgetPolicy(
                period="daily", default_limit_usd=Decimal("100"), email_overrides={}
            ),
            admin_api_key_names=frozenset({"test"}),
        )
        conflict = await admin.put("/admin/budgets/bumps", json=body, headers=HEADERS)
        assert (
            conflict.status_code == 409
            and conflict.json()["code"] == "budget_bump_conflict"
        )
        assert s3.body == initial.model_dump_json().encode()
        assert (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()[
            "revision"
        ] == 1
        control.app.state.runtime_config_state.install(
            budgets=policy_200, admin_api_key_names=frozenset({"test"})
        )
        accepted = await admin.put("/admin/budgets/bumps", json=body, headers=HEADERS)
        assert accepted.status_code == 200
        assert accepted.json()["base_limit_usd"] == "200"
        assert accepted.json()["bump_usd"] == "50"
        assert accepted.json()["limit_usd"] == "250"
        durable = BumpDocument.model_validate_json(cast(bytes, s3.body))
        assert durable.revision == 2
        assert durable.bumps["alice@example.com"].bump_usd == 50
        assert len(durable.events) == 1 and durable.events[0].base_limit_usd == 200


@pytest.mark.asyncio
async def test_legacy_snapshot_preserved_until_first_reasoned_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo) -> datetime:
            return now

    start, end = period_bounds("daily", now)
    s3 = BumpS3()
    old = LegacyBumpSnapshot(
        stage="dev",
        revision=8,
        bumps={
            "alice@example.com": LegacyBudgetGrant(
                period="daily", period_start=start, period_end=end, count=2
            ),
            "bob@example.com": LegacyBudgetGrant(
                period="daily", period_start=start, period_end=end, count=1
            ),
        },
    )
    s3.body = old.model_dump_json().encode()
    s3.etag = '"8"'
    redis = SharedRedis()
    service = BumpService(
        "dev", redis, "landing", "gateway-usage/dev/budgets/bumps.json"
    )
    service._s3 = s3
    worker = BumpService("dev", redis)
    with patch.object(
        gateway_app, "create_usage_ledger_from_env", return_value=Ledger()
    ):
        control = _make_client(runtime_role="control")
    monkeypatch.setattr(budget_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(bump_service_module, "datetime", FixedDatetime)
    control.app.state.bump_service = service
    control.app.state.budget_service = BudgetService(stage="dev", redis=None)
    control.app.state.runtime_config_state.install(
        budgets=BudgetPolicy(
            period="daily", default_limit_usd=Decimal("100"), email_overrides={}
        ),
        admin_api_key_names=frozenset({"test"}),
    )
    await service.refresh_control_once()
    await worker.refresh_query_once()
    assert s3.body == old.model_dump_json().encode()
    assert redis.values[worker.redis_key] == s3.body
    assert isinstance(worker.snapshot, LegacyBumpSnapshot)
    body = {
        "email": "alice@example.com",
        "period": "daily",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "expected_revision": 8,
        "actor": {"user_id": "operator-1", "email": None},
        "reason": "Adjustment",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=control.app), base_url="http://test"
    ) as admin:
        initial = (await admin.get("/admin/budgets/bumps", headers=HEADERS)).json()
        assert [(row["email"], row["bump_usd"]) for row in initial["emails"]] == [
            ("alice@example.com", "10000"),
            ("bob@example.com", "5000"),
        ]
        history_query = {
            key: body[key] for key in ("email", "period", "period_start", "period_end")
        }
        history_before = (
            await admin.get(
                "/admin/budgets/bumps/history", params=history_query, headers=HEADERS
            )
        ).json()
        assert (
            history_before["history_started_at"] is None
            and history_before["events"] == []
        )
        invalid_reason = await admin.post(
            "/admin/budgets/bumps", json={**body, "reason": " "}, headers=HEADERS
        )
        assert invalid_reason.status_code == 400
        assert invalid_reason.json()["code"] == "invalid_request"
        assert s3.body == old.model_dump_json().encode()
        changed = await admin.put(
            "/admin/budgets/bumps",
            json={
                **body,
                "limit_usd": "120.25",
                "expected_base_limit_usd": "100",
            },
            headers=HEADERS,
        )
        assert changed.status_code == 200
        assert changed.json()["bump_usd"] == "20.25"
        durable = BumpDocument.model_validate_json(cast(bytes, s3.body))
        assert durable.revision == 9
        assert durable.bumps["bob@example.com"].bump_usd == 5000
        assert [event.revision for event in durable.events] == [9]
        assert durable.events[0].old_bump_usd == 10000
        history_after = (
            await admin.get(
                "/admin/budgets/bumps/history", params=history_query, headers=HEADERS
            )
        ).json()
        assert (
            history_after["history_started_at"]
            == durable.history_started_at.isoformat()
        )
        assert len(history_after["events"]) == 1
    await service.refresh_control_once()
    await worker.refresh_query_once()
    assert isinstance(worker.snapshot, BumpSnapshot)
    assert b'"events"' not in redis.values[worker.redis_key]
    assert worker.snapshot.bumps["bob@example.com"].bump_usd == 5000


@pytest.mark.asyncio
async def test_independent_control_writers_race_one_s3_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo) -> datetime:
            return now

    start, end = period_bounds("daily", now)

    class RacingS3(BumpS3):
        def __init__(self) -> None:
            super().__init__()
            self.body = (
                LegacyBumpSnapshot(stage="dev", revision=1, bumps={})
                .model_dump_json()
                .encode("utf-8")
            )
            self.etag = '"1"'
            self.read_barrier = Barrier(2)
            self.commit_lock = Lock()

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            observed = super().get_object(Bucket=Bucket, Key=Key)
            self.read_barrier.wait(timeout=5)
            return observed

        def put_object(self, **kwargs: object) -> None:
            with self.commit_lock:
                super().put_object(**kwargs)

    s3 = RacingS3()
    policy = BudgetPolicy(
        period="daily", default_limit_usd=Decimal("100"), email_overrides={}
    )
    with patch.object(
        gateway_app, "create_usage_ledger_from_env", return_value=Ledger()
    ):
        first = _make_client(runtime_role="control")
        second = _make_client(runtime_role="control")
    monkeypatch.setattr(budget_routes, "datetime", FixedDatetime)
    monkeypatch.setattr(bump_service_module, "datetime", FixedDatetime)
    for client in (first, second):
        bump_service = BumpService(
            "dev", None, "landing", "gateway-usage/dev/budgets/bumps.json"
        )
        bump_service._s3 = s3
        client.app.state.bump_service = bump_service
        client.app.state.runtime_config_state.install(
            budgets=policy, admin_api_key_names=frozenset({"test"})
        )
    body = {
        "email": "alice@example.com",
        "expected_revision": 1,
        "period": "daily",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "actor": {"user_id": "operator-2", "email": None},
        "reason": "Emergency capacity",
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first.app), base_url="http://test"
        ) as first_admin,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second.app), base_url="http://test"
        ) as second_admin,
    ):
        responses = await asyncio.gather(
            first_admin.post("/admin/budgets/bumps", json=body, headers=HEADERS),
            second_admin.post("/admin/budgets/bumps", json=body, headers=HEADERS),
        )
    assert sorted(response.status_code for response in responses) == [200, 409]
    durable = BumpDocument.model_validate_json(cast(bytes, s3.body))
    assert durable.revision == 2
    assert durable.bumps["alice@example.com"].bump_usd == 5000
    assert len(durable.events) == 1 and durable.events[0].revision == 2


def test_exact_effective_limit_with_long_policy_value() -> None:
    base = Decimal("123456789012345678901234567890.123456789012345678901")
    assert effective_limit(base, Decimal("0.000000000000000000001")) == Decimal(
        "123456789012345678901234567890.123456789012345678902"
    )
    huge = Decimal("1" + "0" * 4301)
    assert effective_limit(huge, Decimal(1)) == Decimal("1" + "0" * 4300 + "1")


class BudgetAppConfig:
    def __init__(self) -> None:
        self.contents = {
            1: b'{"period":"daily","default_limit_usd":"100","email_overrides":{"bob@example.com":"200"}}'
        }
        self.versions = [{"VersionNumber": 1, "Description": "initial policy"}]
        self.deployments = [
            {
                "ConfigurationProfileId": "budget-profile",
                "ConfigurationVersion": "1",
                "DeploymentNumber": 1,
                "State": "COMPLETE",
            }
        ]
        self.served_version = 1
        self.creates: list[dict[str, object]] = []
        self.fail_start = False

    def list_hosted_configuration_versions(self, **kwargs: object) -> dict[str, object]:
        offset = int(cast(str, kwargs.get("NextToken", "0")))
        result: dict[str, object] = {"Items": self.versions[offset : offset + 1]}
        if offset + 1 < len(self.versions):
            result["NextToken"] = str(offset + 1)
        return result

    def list_deployments(self, **kwargs: object) -> dict[str, object]:
        offset = int(cast(str, kwargs.get("NextToken", "0")))
        result: dict[str, object] = {"Items": self.deployments[offset : offset + 1]}
        if offset + 1 < len(self.deployments):
            result["NextToken"] = str(offset + 1)
        return result

    def get_hosted_configuration_version(
        self, **kwargs: object
    ) -> dict[str, io.BytesIO]:
        return {
            "Content": io.BytesIO(self.contents[cast(int, kwargs["VersionNumber"])])
        }

    def create_hosted_configuration_version(self, **kwargs: object) -> dict[str, int]:
        assert kwargs["LatestVersionNumber"] == self.versions[-1]["VersionNumber"]
        version = len(self.versions) + 1
        self.contents[version] = cast(bytes, kwargs["Content"])
        self.versions.append(
            {"VersionNumber": version, "Description": kwargs["Description"]}
        )
        self.creates.append(kwargs)
        return {"VersionNumber": version}

    def start_deployment(self, **kwargs: object) -> dict[str, object]:
        assert (
            kwargs["LatestDeploymentNumber"] == self.deployments[-1]["DeploymentNumber"]
        )
        if self.fail_start:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConflictException",
                        "Message": "deployment conflict",
                    }
                },
                "StartDeployment",
            )
        deployment = {
            "ConfigurationProfileId": "budget-profile",
            "ConfigurationVersion": kwargs["ConfigurationVersion"],
            "DeploymentNumber": len(self.deployments) + 1,
            "State": "DEPLOYING",
        }
        self.deployments.append(deployment)
        return deployment

    def complete(self) -> None:
        self.deployments[-1]["State"] = "COMPLETE"
        self.served_version = int(self.deployments[-1]["ConfigurationVersion"])

    async def fetch(self, profile: str) -> SimpleNamespace:
        assert profile == "budget-profile"
        return SimpleNamespace(
            content=self.contents[self.served_version], version=str(self.served_version)
        )

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_permanent_override_audit_tokens_and_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appconfig = BudgetAppConfig()
    service = OverrideService(
        stage="dev",
        application="app",
        environment="env",
        profile="budget-profile",
        strategy="immediate",
        agent=cast(AppConfigAgentClient, appconfig),
        client=appconfig,
    )
    monkeypatch.setattr(gateway_app, "override_service_from_env", lambda stage: service)
    with patch.object(
        gateway_app, "create_usage_ledger_from_env", return_value=Ledger()
    ):
        control = _make_client(runtime_role="control")
    control.app.state.runtime_config_state.install(
        admin_api_key_names=frozenset({"test"})
    )
    now = datetime.now(UTC)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=control.app), base_url="http://test"
    ) as admin:
        initial = (
            await admin.get(
                "/admin/budgets/overrides",
                params={"email": " ALICE@EXAMPLE.COM "},
                headers=HEADERS,
            )
        ).json()
        assert (
            initial["email"] == "alice@example.com"
            and initial["base_limit_usd"] == "100"
        )
        assert initial["deployment"] == {
            "number": 1,
            "configuration_version": 1,
            "state": "COMPLETE",
        }
        body = {
            "email": "alice@example.com",
            "expected_configuration_version": 1,
            "expected_latest_version": 1,
            "expected_deployment_number": 1,
            "actor": {"user_id": "operator-1", "email": "operator@example.com"},
            "reason": "Ongoing capacity",
            "base_limit_usd": "120.000000000000000000001",
        }
        for invalid in ({"reason": " "}, {"actor": None}, {"base_limit_usd": "-1"}):
            assert (
                await admin.put(
                    "/admin/budgets/overrides",
                    json={**body, **invalid},
                    headers=HEADERS,
                )
            ).status_code == 400
        assert (
            await admin.put(
                "/admin/budgets/overrides",
                json={**body, "reason": "x" * 1025},
                headers=HEADERS,
            )
        ).json()["code"] == "override_reason_too_long"
        assert not appconfig.creates
        saved = await admin.put("/admin/budgets/overrides", json=body, headers=HEADERS)
        assert saved.status_code == 200 and saved.json()["state"] == "DEPLOYING"
        assert appconfig.creates[0]["LatestVersionNumber"] == 1
        assert json.loads(cast(bytes, appconfig.creates[0]["Content"]))[
            "email_overrides"
        ] == {"bob@example.com": "200", "alice@example.com": body["base_limit_usd"]}
        assert (
            await admin.put("/admin/budgets/overrides", json=body, headers=HEADERS)
        ).status_code == 409
        pending = (
            await admin.get(
                "/admin/budgets/overrides",
                params={"email": body["email"]},
                headers=HEADERS,
            )
        ).json()
        assert (
            pending["base_limit_usd"] == "100"
            and pending["deployment"]["state"] == "DEPLOYING"
        )
        appconfig.complete()
        appconfig.deployments.append(
            {
                "ConfigurationProfileId": "model-profile",
                "ConfigurationVersion": "8",
                "DeploymentNumber": 3,
                "State": "COMPLETE",
            }
        )
        current = (
            await admin.get(
                "/admin/budgets/overrides",
                params={"email": body["email"]},
                headers=HEADERS,
            )
        ).json()
        assert (
            current["base_limit_usd"] == body["base_limit_usd"]
            and current["configuration_version"] == 2
        )
        assert current["deployment"] == {
            "number": 2,
            "configuration_version": 2,
            "state": "COMPLETE",
        }
        assert current["latest_deployment_number"] == 3
        bounds = period_bounds("daily", now)
        allowance = BumpService("dev", None)
        allowance.snapshot = BumpSnapshot(
            stage="dev",
            revision=1,
            bumps={
                "alice@example.com": BudgetGrant(
                    period="daily",
                    period_start=bounds[0],
                    period_end=bounds[1],
                    bump_usd=Decimal("50"),
                ),
            },
        )
        spend = BudgetService(stage="dev", redis=None)
        spend.snapshot = snapshot(now, 1, {"alice@example.com": "160"})
        spend.bump_service = allowance
        assert not spend.blocked(
            BudgetPolicy.model_validate_json(appconfig.contents[2]), body["email"], now
        )
        remove = {key: body[key] for key in ("email", "actor", "reason")}
        remove.update(
            expected_configuration_version=2,
            expected_latest_version=2,
            expected_deployment_number=3,
        )
        assert (
            await admin.request(
                "DELETE",
                "/admin/budgets/overrides",
                json={**remove, "expected_configuration_version": 1},
                headers=HEADERS,
            )
        ).status_code == 409
        removed = await admin.request(
            "DELETE", "/admin/budgets/overrides", json=remove, headers=HEADERS
        )
        assert (
            removed.status_code == 200 and removed.json()["configuration_version"] == 3
        )
        appconfig.versions.append(
            {"VersionNumber": 99, "Description": json.dumps({"description": "unrelated"})}
        )
        history = (
            await admin.get(
                "/admin/budgets/overrides/history",
                params={
                    "email": body["email"],
                    "start": (now - timedelta(minutes=1)).isoformat(),
                    "end": (now + timedelta(minutes=1)).isoformat(),
                },
                headers=HEADERS,
            )
        ).json()
        appconfig.versions.pop()
        assert [event["action"] for event in history["events"]] == ["set", "remove"]
        assert [event["deployment"]["state"] for event in history["events"]] == [
            "COMPLETE",
            "DEPLOYING",
        ]
        assert history["events"][0]["actor"] == body["actor"]
        assert history["events"][0]["new_base_limit_usd"] == body["base_limit_usd"]
        assert history["events"][1]["new_base_limit_usd"] == "100"
        appconfig.complete()
        reset = (
            await admin.get(
                "/admin/budgets/overrides",
                params={"email": body["email"]},
                headers=HEADERS,
            )
        ).json()
        assert reset["override_limit_usd"] is None and reset["base_limit_usd"] == "100"
        assert json.loads(appconfig.contents[3])["email_overrides"] == {
            "bob@example.com": "200"
        }
        assert spend.blocked(
            BudgetPolicy.model_validate_json(appconfig.contents[3]), body["email"], now
        )
        assert effective_limit(
            Decimal(current["base_limit_usd"]), Decimal("50")
        ) == Decimal("170.000000000000000000001")
        assert effective_limit(
            Decimal(reset["base_limit_usd"]), Decimal("50")
        ) == Decimal("150")
        original_fetch = appconfig.fetch

        async def mismatched_fetch(profile: str) -> SimpleNamespace:
            return SimpleNamespace(content=appconfig.contents[2], version="3")

        monkeypatch.setattr(appconfig, "fetch", mismatched_fetch)
        mismatch = await admin.get(
            "/admin/budgets/overrides", params={"email": body["email"]}, headers=HEADERS
        )
        assert mismatch.status_code == 409
        monkeypatch.setattr(appconfig, "fetch", original_fetch)
        attempt = {
            **body,
            "expected_configuration_version": 3,
            "expected_latest_version": 3,
            "expected_deployment_number": 4,
            "base_limit_usd": "140",
        }
        failed = await admin.put(
            "/admin/budgets/overrides", json=attempt, headers=HEADERS
        )
        assert failed.status_code == 200 and failed.json()["configuration_version"] == 4
        appconfig.deployments[-1]["State"] = "ROLLED_BACK"
        rolled_back = (
            await admin.get(
                "/admin/budgets/overrides",
                params={"email": body["email"]},
                headers=HEADERS,
            )
        ).json()
        assert rolled_back["base_limit_usd"] == "100"
        assert rolled_back["deployment"] == {
            "number": 5,
            "configuration_version": 4,
            "state": "ROLLED_BACK",
        }
        next_attempt = {
            **attempt,
            "expected_latest_version": 4,
            "expected_deployment_number": 5,
            "base_limit_usd": "150",
        }
        recovered = await admin.put(
            "/admin/budgets/overrides", json=next_attempt, headers=HEADERS
        )
        assert (
            recovered.status_code == 200
            and recovered.json()["configuration_version"] == 5
        )
        assert json.loads(appconfig.contents[5])["email_overrides"] == {
            "bob@example.com": "200",
            "alice@example.com": "150",
        }
        after_failure = (
            await admin.get(
                "/admin/budgets/overrides/history",
                params={
                    "email": body["email"],
                    "start": (now - timedelta(minutes=1)).isoformat(),
                    "end": (now + timedelta(minutes=1)).isoformat(),
                },
                headers=HEADERS,
            )
        ).json()
        assert [event["deployment"]["state"] for event in after_failure["events"]] == [
            "COMPLETE",
            "COMPLETE",
            "ROLLED_BACK",
            "DEPLOYING",
        ]
        appconfig.complete()
        appconfig.fail_start = True
        orphan = {
            **attempt,
            "expected_configuration_version": 5,
            "expected_latest_version": 5,
            "expected_deployment_number": 6,
            "base_limit_usd": "160",
        }
        assert (
            await admin.put("/admin/budgets/overrides", json=orphan, headers=HEADERS)
        ).status_code == 409
        saved_only = (
            await admin.get(
                "/admin/budgets/overrides/history",
                params={
                    "email": body["email"],
                    "start": (now - timedelta(minutes=1)).isoformat(),
                    "end": (now + timedelta(minutes=1)).isoformat(),
                },
                headers=HEADERS,
            )
        ).json()
        assert saved_only["events"][-1]["configuration_version"] == 6
        assert saved_only["events"][-1]["deployment"] is None
        appconfig.fail_start = False
        fresh = {**orphan, "expected_latest_version": 6, "base_limit_usd": "170"}
        assert (
            await admin.put("/admin/budgets/overrides", json=fresh, headers=HEADERS)
        ).status_code == 200
        assert json.loads(appconfig.contents[7])["email_overrides"] == {
            "bob@example.com": "200",
            "alice@example.com": "170",
        }
