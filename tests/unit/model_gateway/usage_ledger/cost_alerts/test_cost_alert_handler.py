from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, cast

import pytest

import model_gateway.usage_ledger.cost_alerts.handler as cost_alert_handler
import model_gateway.usage_ledger.cost_alerts.rules as cost_alerts


class FakeStateTable:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}
        self.put_calls: list[dict[str, object]] = []

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = cast(dict[str, object], kwargs["Key"])
        item = self.items.get(cast(str, key["PK"]))
        return {} if item is None else {"Item": dict(item)}

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(dict(kwargs))
        assert set(kwargs) == {"Item"}
        item = cast(dict[str, object], kwargs["Item"])
        self.items[cast(str, item["PK"])] = dict(item)
        return {}


class FakeSlackSender:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def post_payload(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)


def _rule(name: str = "dev-email-1d") -> cost_alerts.CostAlertRule:
    return cost_alerts.CostAlertRule(
        name=name,
        grain="1d",
        group_by="identity_email",
        threshold_usd=Decimal("1000"),
    )


def _row(
    dimension: str,
    cost: str,
    requests: int = 1,
    eligible_requests: int | None = None,
) -> cost_alerts.ComparisonRow:
    current = Decimal(cost)
    return cost_alerts.ComparisonRow(
        bucket_start_utc=datetime(2026, 7, 10, 7, tzinfo=UTC),
        bucket_end_utc=datetime(2026, 7, 11, 7, tzinfo=UTC),
        data_through_utc=datetime(2026, 7, 10, 20, tzinfo=UTC),
        watermark_utc=datetime(2026, 7, 10, 20, tzinfo=UTC),
        dimension_value=dimension,
        current_request_count=requests,
        current_eligible_request_count=(
            requests if eligible_requests is None else eligible_requests
        ),
        current_cost_usd=current,
        previous_cost_usd=Decimal("0"),
        absolute_increase_usd=current,
        percent_increase=None,
    )


def _install_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    table: FakeStateTable,
    sender: FakeSlackSender,
    rows: tuple[cost_alerts.ComparisonRow, ...],
) -> tuple[list[cost_alerts.CostAlertRule], list[tuple[str, ...]]]:
    config = json.dumps(
        {
            "rules": [
                {
                    "name": "dev-email-1d",
                    "grain": "1d",
                    "group_by": "identity_email",
                    "threshold_usd": "1000",
                }
            ]
        }
    )
    monkeypatch.setenv("GATEWAY_COST_ALERT_CONFIG_PARAMETER_NAME", "config")
    monkeypatch.setenv("GATEWAY_COST_ALERT_STATE_TABLE_NAME", "state")
    monkeypatch.setenv("GATEWAY_COST_ALERT_SLACK_WEBHOOK_SECRET_NAME", "slack")
    monkeypatch.setenv("GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME", "workgroup")
    monkeypatch.setenv("GATEWAY_USAGE_REDSHIFT_DATABASE_NAME", "database")
    monkeypatch.setenv("GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME", "gateway_usage")
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setattr(cost_alert_handler, "_parameter_value", lambda _name: config)
    monkeypatch.setattr(cost_alert_handler, "_secret_value", lambda _name: "webhook")
    monkeypatch.setattr(cost_alert_handler, "_state_table", lambda _name: table)
    monkeypatch.setattr(cost_alert_handler, "_slack_sender", lambda _url: sender)
    monkeypatch.setattr(cost_alert_handler, "_redshift_client", object)
    ignored_model_keys = ("provider/ignored",)
    monkeypatch.setattr(
        cost_alert_handler,
        "_ignored_for_cost_model_keys",
        lambda: ignored_model_keys,
    )
    queried_rules: list[cost_alerts.CostAlertRule] = []
    ignored_model_key_calls: list[tuple[str, ...]] = []

    def query_rules(
        **kwargs: object,
    ) -> tuple[tuple[cost_alerts.CostAlertRule, dict[str, object]], ...]:
        rules = cast(tuple[cost_alerts.CostAlertRule, ...], kwargs["rules"])
        queried_rules.extend(rules)
        ignored_model_key_calls.append(
            tuple(cast(Sequence[str], kwargs["ignored_model_keys"]))
        )
        return tuple((rule, {}) for rule in rules)

    monkeypatch.setattr(cost_alert_handler, "_query_rules", query_rules)
    monkeypatch.setattr(
        cost_alert_handler.cost_alerts,
        "parse_comparison_rows",
        lambda _result: rows,
    )

    def query_breakdowns(**kwargs: object) -> dict[str, object]:
        ignored_model_key_calls.append(
            tuple(cast(Sequence[str], kwargs["ignored_model_keys"]))
        )
        return {"ColumnMetadata": [], "Records": []}

    monkeypatch.setattr(cost_alert_handler, "_query_breakdowns", query_breakdowns)
    return queried_rules, ignored_model_key_calls


def test_ignored_for_cost_model_keys_come_from_generated_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "all_models.json"
    registry_path.write_text(
        json.dumps(
            {
                "provider/included": {"metadata": {"ignored_for_cost": False}},
                "provider/ignored": {"metadata": {"ignored_for_cost": True}},
            }
        )
    )
    monkeypatch.setattr(cost_alert_handler, "_MODEL_REGISTRY_PATH", registry_path)

    ignored_model_keys = cost_alert_handler._ignored_for_cost_model_keys()

    assert list(ignored_model_keys) == ["provider/ignored"]


def test_handler_posts_and_persists_each_breached_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeStateTable()
    sender = FakeSlackSender()
    queried, ignored_model_key_calls = _install_handler(
        monkeypatch,
        table=table,
        sender=sender,
        rows=(
            _row("connor@example.com", "1400", 140),
            _row("hung@example.com", "1300", 130),
            _row("devin@example.com", "800", 80),
        ),
    )

    cost_alert_handler.handler({}, object())

    assert queried == [_rule()]
    assert [list(keys) for keys in ignored_model_key_calls] == [
        ["provider/ignored"],
        ["provider/ignored"],
    ]
    assert len(sender.payloads) == 2
    assert len(table.items) == 2
    assert all(
        set(item)
        == {
            "PK",
            "last_alerted_bucket",
            "last_notified_level",
            "updated_at",
            "ttl",
        }
        for item in table.items.values()
    )
    assert all(call.keys() == {"Item"} for call in table.put_calls)


def test_same_bucket_suppresses_duplicate_and_posts_worsening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeStateTable()
    sender = FakeSlackSender()
    rows = [_row("user@example.com", "1100")]
    _install_handler(
        monkeypatch,
        table=table,
        sender=sender,
        rows=tuple(rows),
    )

    cost_alert_handler.handler({}, object())
    assert len(sender.payloads) == 1

    cost_alert_handler.handler({}, object())
    assert len(sender.payloads) == 1
    rows[0] = _row("user@example.com", "1600")
    monkeypatch.setattr(
        cost_alert_handler.cost_alerts,
        "parse_comparison_rows",
        lambda _result: tuple(rows),
    )
    cost_alert_handler.handler({}, object())

    assert len(sender.payloads) == 2
    assert "Worsening" in json.dumps(sender.payloads[1])


def test_slack_post_happens_before_direct_state_write() -> None:
    sender = FakeSlackSender()

    class OrderedTable(FakeStateTable):
        def put_item(self, **kwargs: object) -> dict[str, object]:
            assert len(sender.payloads) == 1
            return super().put_item(**kwargs)

    candidate = cost_alerts.evaluate_rule_rows(
        _rule(),
        (_row("user@example.com", "1400"),),
    )[0]

    cost_alert_handler._process_breached_candidate(
        table=OrderedTable(),
        sender=cast(Any, sender),
        stage="dev",
        rule=_rule(),
        candidate=candidate,
        notification_kind="active",
        notified_level=1,
    )


def test_breakdown_enrichment_preserves_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = cost_alerts.evaluate_rule_rows(
        _rule(),
        (_row("first@example.com", "2000"), _row("second@example.com", "1500")),
    )
    monkeypatch.setattr(
        cost_alert_handler,
        "_query_breakdowns",
        lambda **_kwargs: {"ColumnMetadata": [], "Records": []},
    )

    enriched = cost_alert_handler._add_required_breakdowns(
        redshift_client=object(),
        workgroup_name="workgroup",
        database_name="database",
        ignored_model_keys=(),
        schema="gateway_usage",
        rule=_rule(),
        candidates=(first, second),
    )

    assert [candidate.scope_value for candidate in enriched] == [
        "first@example.com",
        "second@example.com",
    ]


class ConcurrentRedshiftClient:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.started_count_at_poll: list[int] = []

    def execute_statement(self, **kwargs: object) -> dict[str, object]:
        statement_id = f"statement-{len(self.started)}"
        self.started.append(cast(str, kwargs["StatementName"]))
        return {"Id": statement_id}

    def describe_statement(self, **_kwargs: object) -> dict[str, object]:
        self.started_count_at_poll.append(len(self.started))
        return {"Status": "FINISHED"}

    def get_statement_result(self, **_kwargs: object) -> dict[str, object]:
        return {"ColumnMetadata": [], "Records": []}


def test_rule_queries_run_eight_at_a_time_and_yield_in_config_order() -> None:
    client = ConcurrentRedshiftClient()
    rules = tuple(_rule(f"rule-{index}") for index in range(9))

    results = tuple(
        cost_alert_handler._query_rules(
            redshift_client=client,
            workgroup_name="workgroup",
            database_name="database",
            ignored_model_keys=(),
            schema="gateway_usage",
            rules=rules,
        )
    )

    assert client.started_count_at_poll[0] == 8
    assert [rule.name for rule, _result in results] == [
        f"rule-{index}" for index in range(9)
    ]


class PaginatedRedshiftClient:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, object]] = []
        self.result_calls: list[dict[str, object]] = []
        self.pages = [
            {
                "ColumnMetadata": [{"name": "value"}],
                "Records": [[{"longValue": 1}]],
                "NextToken": "next",
            },
            {
                "ColumnMetadata": [{"name": "value"}],
                "Records": [[{"longValue": 2}]],
            },
        ]

    def execute_statement(self, **kwargs: object) -> dict[str, object]:
        self.execute_calls.append(kwargs)
        return {"Id": "statement"}

    def describe_statement(self, **_kwargs: object) -> dict[str, object]:
        return {"Status": "FINISHED"}

    def get_statement_result(self, **kwargs: object) -> dict[str, object]:
        self.result_calls.append(kwargs)
        return self.pages.pop(0)


def test_query_statement_collects_all_result_pages() -> None:
    client = PaginatedRedshiftClient()

    result = cost_alert_handler._query_statement(
        redshift_client=client,
        workgroup_name="workgroup",
        database_name="database",
        sql="select 1",
        statement_name="test",
    )

    assert client.execute_calls == [
        {
            "WorkgroupName": "workgroup",
            "Database": "database",
            "Sql": "select 1",
            "StatementName": "test",
        }
    ]
    assert client.result_calls == [
        {"Id": "statement"},
        {"Id": "statement", "NextToken": "next"},
    ]
    assert result == {
        "ColumnMetadata": [{"name": "value"}],
        "Records": [[{"longValue": 1}], [{"longValue": 2}]],
    }
