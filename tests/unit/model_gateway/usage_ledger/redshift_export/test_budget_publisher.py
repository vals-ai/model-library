"""Recorded daily aggregates become one complete, ordered budget document."""

from collections.abc import Callable
import io
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from model_gateway.budgets.schema import SpendSnapshot, period_bounds
from model_gateway.usage_ledger.budgets import publisher


class Clock(datetime):
    current = datetime(2026, 10, 21, 12, tzinfo=UTC)

    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return cls.current


class SourceWarehouse:
    """Execute the generated aggregate query against source rows, not canned results."""

    def __init__(self) -> None:
        self.database = sqlite3.connect(":memory:")
        self.database.execute("attach database ':memory:' as gateway_usage")
        self.database.execute(
            "create table gateway_usage.usage_agg_1d ("
            "bucket_start_utc text, identity_email text, provider_model text, "
            "cost_usd_sum real, cost_recorded_count integer)"
        )
        self.rows: list[tuple[object, ...]] = []
        self.executions = 0
        self.on_query: Callable[[], None] | None = None

    def add(
        self, day: str, email: str, model: str, cost: float, recorded: int = 1
    ) -> None:
        self.database.execute(
            "insert into gateway_usage.usage_agg_1d values (?, ?, ?, ?, ?)",
            (day, email, model, cost, recorded),
        )

    def execute_statement(self, **kwargs: object) -> dict[str, str]:
        self.executions += 1
        sql = re.sub(r"cast\('([^']+)' as timestamptz\)", r"'\1'", str(kwargs["Sql"]))
        self.rows = self.database.execute(sql).fetchall()
        if self.on_query is not None:
            callback, self.on_query = self.on_query, None
            callback()
        return {"Id": "snapshot-query"}

    def describe_statement(self, **kwargs: object) -> dict[str, str]:
        return {"Status": "FINISHED"}

    def get_statement_result(self, **kwargs: object) -> dict[str, object]:
        return {
            "Records": [
                [{"stringValue": str(value)} for value in row] for row in self.rows
            ]
        }


class DocumentStore:
    def __init__(self) -> None:
        self.body: bytes | None = None
        self.etag = '"1"'
        self.metadata: dict[str, str] = {}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        if self.body is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {
            "Body": io.BytesIO(self.body),
            "ETag": self.etag,
            "Metadata": self.metadata,
        }

    def put_object(self, **kwargs: object) -> dict[str, object]:
        if (self.body is None and kwargs.get("IfNoneMatch") != "*") or (
            self.body is not None and kwargs.get("IfMatch") != self.etag
        ):
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "stale"}},
                "PutObject",
            )
        self.body = bytes(kwargs["Body"])
        self.metadata = dict(kwargs["Metadata"])
        self.etag = '"3"'
        return {}

    def snapshot(self) -> SpendSnapshot:
        assert self.body is not None
        return SpendSnapshot.model_validate_json(self.body)


def publish(
    store: DocumentStore,
    warehouse: SourceWarehouse,
    now: datetime,
    *,
    idle: bool = False,
) -> bool:
    return publisher.publish_spend_snapshot(
        s3_client=store,
        redshift_data_client=warehouse,
        stage="dev",
        bucket="landing",
        key="gateway-usage/dev/spend/current.json",
        workgroup="workgroup",
        database="usage",
        secret_arn="secret",
        schema="gateway_usage",
        data_through=now - timedelta(minutes=2),
        idle=idle,
    )


def test_source_rows_publish_recorded_nonignored_spend_and_idle_rollover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Clock.current = datetime(2026, 10, 21, 12, tzinfo=UTC)
    monkeypatch.setattr(publisher, "datetime", Clock)
    now = datetime(2026, 10, 21, 12, tzinfo=UTC)
    warehouse, store = SourceWarehouse(), DocumentStore()
    warehouse.add("2026-10-21T00:00:00+00:00", " User@Example.com ", "billable", 6.25)
    warehouse.add("2026-10-20T00:00:00+00:00", "user@example.com", "billable", 1.25)
    warehouse.add("2026-10-21T00:00:00+00:00", "user@example.com", "internal", 99)
    warehouse.add(
        "2026-10-21T00:00:00+00:00", "user@example.com", "billable", 50, recorded=0
    )
    warehouse.add("2026-09-30T00:00:00+00:00", "user@example.com", "billable", 100)
    registry = {
        "billable": {"metadata": {"ignored_for_cost": False}},
        "internal": {"metadata": {"ignored_for_cost": True}},
    }
    with patch.object(
        publisher.gateway_registry, "fetch_live_models", return_value=registry
    ):
        assert publish(store, warehouse, now)
        first = store.snapshot()
        assert first.periods["daily"].spend_usd == {"user@example.com": Decimal("6.25")}
        assert first.periods["weekly"].spend_usd == {"user@example.com": Decimal("7.5")}
        assert first.periods["monthly"].spend_usd == {
            "user@example.com": Decimal("7.5")
        }
        assert first.data_through == now - timedelta(minutes=2)
        assert not publish(store, warehouse, now, idle=True)
        assert warehouse.executions == 1

        # A recalculation can lower spend; an idle month change emits empty complete periods.
        warehouse.database.execute("delete from gateway_usage.usage_agg_1d")
        assert publish(store, warehouse, now)
        assert store.snapshot().periods["monthly"].spend_usd == {}
        new_month = datetime(2026, 11, 1, tzinfo=UTC)
        Clock.current = new_month
        assert publish(store, warehouse, new_month, idle=True)
        rolled = store.snapshot()
        for period in ("daily", "weekly", "monthly"):
            assert rolled.periods[period].spend_usd == {}
            assert (
                rolled.periods[period].start,
                rolled.periods[period].end,
            ) == period_bounds(period, new_month)
        assert rolled.data_through == now - timedelta(minutes=2)


def test_idle_export_recalculates_spend_when_ignored_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Clock.current = datetime(2026, 10, 21, 12, tzinfo=UTC)
    monkeypatch.setattr(publisher, "datetime", Clock)
    warehouse, store = SourceWarehouse(), DocumentStore()
    warehouse.add("2026-10-21T00:00:00+00:00", "user@example.com", "billable", 4)
    warehouse.add("2026-10-21T00:00:00+00:00", "user@example.com", "internal", 9)
    registry = {
        "billable": {"metadata": {"ignored_for_cost": False}},
        "internal": {"metadata": {"ignored_for_cost": False}},
    }
    monkeypatch.setattr(
        publisher.gateway_registry, "fetch_live_models", lambda: registry
    )
    now = Clock.current

    assert publish(store, warehouse, now)
    assert store.snapshot().periods["daily"].spend_usd == {
        "user@example.com": Decimal("13")
    }
    assert not publish(store, warehouse, now, idle=True)
    assert warehouse.executions == 1

    registry["internal"]["metadata"]["ignored_for_cost"] = True
    assert publish(store, warehouse, now, idle=True)
    assert store.snapshot().periods["daily"].spend_usd == {
        "user@example.com": Decimal("4")
    }
    assert warehouse.executions == 2
    assert not publish(store, warehouse, now, idle=True)
    assert warehouse.executions == 2

    registry["internal"]["metadata"]["ignored_for_cost"] = False
    assert publish(store, warehouse, now, idle=True)
    assert store.snapshot().periods["daily"].spend_usd == {
        "user@example.com": Decimal("13")
    }
    assert warehouse.executions == 3


def test_conflicted_publisher_requeries_and_preserves_lower_cost_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Clock.current = datetime(2026, 10, 21, 12, tzinfo=UTC)
    monkeypatch.setattr(publisher, "datetime", Clock)
    now = datetime(2026, 10, 21, 12, tzinfo=UTC)
    slow, fast, store = SourceWarehouse(), SourceWarehouse(), DocumentStore()
    slow.add("2026-10-21T00:00:00+00:00", "user@example.com", "billable", 3)
    fast.add("2026-10-21T00:00:00+00:00", "user@example.com", "billable", 5)
    registry = {"billable": {"metadata": {"ignored_for_cost": False}}}
    winning: list[SpendSnapshot] = []

    def complete_fast_query() -> None:
        assert publish(store, fast, now)
        winning.append(store.snapshot())
        slow.database.execute("update gateway_usage.usage_agg_1d set cost_usd_sum = 2")

    slow.on_query = complete_fast_query
    with patch.object(
        publisher.gateway_registry, "fetch_live_models", return_value=registry
    ):
        assert publish(store, slow, now)
    assert winning[0].periods["monthly"].spend_usd == {"user@example.com": Decimal("5")}
    assert slow.executions == 2
    assert store.snapshot().revision == winning[0].revision + 1
    assert store.snapshot().periods["monthly"].spend_usd == {
        "user@example.com": Decimal("2")
    }


def test_new_calendar_day_wins_over_exporter_start_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_midnight = datetime(2026, 10, 21, 23, 59, tzinfo=UTC)
    after_midnight = datetime(2026, 10, 22, 0, 1, tzinfo=UTC)
    Clock.current = before_midnight
    monkeypatch.setattr(publisher, "datetime", Clock)
    store, warehouse = DocumentStore(), SourceWarehouse()
    warehouse.add("2026-10-21T00:00:00+00:00", "user@example.com", "billable", 10)
    with patch.object(
        publisher.gateway_registry,
        "fetch_live_models",
        return_value={"billable": {"metadata": {"ignored_for_cost": False}}},
    ):
        assert publish(store, warehouse, before_midnight)
        warehouse.add("2026-10-22T00:00:00+00:00", "user@example.com", "billable", 2)
        Clock.current = after_midnight
        assert publish(store, warehouse, before_midnight)
    result = store.snapshot()
    for period in ("daily", "weekly", "monthly"):
        assert (
            result.periods[period].start,
            result.periods[period].end,
        ) == period_bounds(period, after_midnight)
    assert result.periods["daily"].spend_usd == {"user@example.com": Decimal("2")}
    assert result.periods["weekly"].spend_usd == {"user@example.com": Decimal("12")}
    assert result.data_through == before_midnight - timedelta(minutes=2)


def test_published_snapshot_preserves_redshift_decimal_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Clock.current = datetime(2026, 10, 21, 12, tzinfo=UTC)
    monkeypatch.setattr(publisher, "datetime", Clock)
    warehouse, store = SourceWarehouse(), DocumentStore()
    warehouse.add("2026-10-21T00:00:00+00:00", "user@example.com", "billable", 1)
    monkeypatch.setattr(
        publisher.gateway_registry,
        "fetch_live_models",
        lambda: {"billable": {"metadata": {"ignored_for_cost": False}}},
    )
    exact = "10000000000000000.000000000001"
    with patch.object(
        warehouse,
        "get_statement_result",
        return_value={
            "Records": [
                [
                    {"stringValue": period},
                    {"stringValue": "user@example.com"},
                    {"stringValue": exact},
                ]
                for period in ("daily", "weekly", "monthly")
            ]
        },
    ):
        assert publish(store, warehouse, Clock.current)
    snapshot = store.snapshot()
    assert str(snapshot.periods["daily"].spend_usd["user@example.com"]) == exact
