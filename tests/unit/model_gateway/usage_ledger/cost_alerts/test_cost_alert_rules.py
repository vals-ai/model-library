from datetime import UTC, datetime
from decimal import Decimal
import json

import pytest

from model_gateway.usage_ledger.cost_alerts import rules as cost_alerts

_SCHEMA = "gateway_usage"


def _rule(
    *,
    grain: cost_alerts.CostAlertGrain = "1d",
    group_by: cost_alerts.CostAlertGroupBy = "identity_email",
    threshold_usd: str = "1000",
    include_missing: bool = False,
) -> cost_alerts.CostAlertRule:
    return cost_alerts.CostAlertRule(
        name="test-rule",
        grain=grain,
        group_by=group_by,
        threshold_usd=Decimal(threshold_usd),
        include_missing=include_missing,
    )


def _row(
    dimension_value: str,
    *,
    current_cost: str,
    previous_cost: str = "0",
    current_requests: int = 1,
) -> cost_alerts.ComparisonRow:
    current = Decimal(current_cost)
    previous = Decimal(previous_cost)
    return cost_alerts.ComparisonRow(
        bucket_start_utc=datetime(2026, 3, 8, 8, tzinfo=UTC),
        bucket_end_utc=datetime(2026, 3, 9, 7, tzinfo=UTC),
        data_through_utc=datetime(2026, 3, 8, 20, tzinfo=UTC),
        watermark_utc=datetime(2026, 3, 8, 20, tzinfo=UTC),
        dimension_value=dimension_value,
        current_request_count=current_requests,
        current_cost_usd=current,
        previous_cost_usd=previous,
        absolute_increase_usd=current - previous,
        percent_increase=None
        if previous == 0
        else ((current - previous) / previous) * 100,
    )


def _config_with_image(*, url: str, alt_text: str) -> str:
    return json.dumps(
        {
            "rules": [
                {
                    "name": "image-rule",
                    "grain": "1h",
                    "group_by": "global",
                    "threshold_usd": "10",
                    "image": {"url": url, "alt_text": alt_text},
                }
            ]
        }
    )


def test_config_rejects_duplicate_incident_identities() -> None:
    rule = {
        "name": "duplicate",
        "grain": "1d",
        "group_by": "identity_email",
        "threshold_usd": "1000",
    }

    with pytest.raises(ValueError, match="duplicate cost alert rule identity"):
        cost_alerts.parse_cost_alert_config_json(json.dumps({"rules": [rule, rule]}))


@pytest.mark.parametrize("threshold", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_non_finite_thresholds(threshold: str) -> None:
    config = {
        "rules": [
            {
                "name": "non-finite",
                "grain": "1h",
                "group_by": "global",
                "threshold_usd": threshold,
            }
        ]
    }

    with pytest.raises(ValueError, match="threshold_usd must be finite"):
        cost_alerts.parse_cost_alert_config_json(json.dumps(config))


def test_config_ignores_unknown_fields() -> None:
    config = {
        "rules": [
            {
                "name": "tolerant",
                "grain": "1h",
                "group_by": "global",
                "threshold_usd": "10",
                "unknown_rule_field": True,
                "image": {
                    "url": "https://example.com/alert.png",
                    "unknown_image_field": True,
                },
            }
        ],
        "unknown_root_field": True,
    }

    parsed = cost_alerts.parse_cost_alert_config_json(json.dumps(config))

    rule = parsed[0]
    assert rule.name == "tolerant"
    assert rule.include_missing is False
    assert rule.image is not None
    assert rule.image.alt_text == "cost alert image"
    assert rule.image.attach_to == ("worsening",)


def test_image_config_accepts_exact_slack_limits() -> None:
    url_prefix = "https://example.com/"
    url = url_prefix + "u" * (3000 - len(url_prefix))
    alt_text = "a" * 2000

    parsed = cost_alerts.parse_cost_alert_config_json(
        _config_with_image(url=url, alt_text=alt_text)
    )

    assert parsed[0].image is not None
    assert parsed[0].image.url == url
    assert parsed[0].image.alt_text == alt_text


@pytest.mark.parametrize(
    ("url_length", "alt_text_length", "expected_error"),
    [
        (3001, 1, "image.url must not exceed 3000 characters"),
        (
            len("https://example.com/image.png"),
            2001,
            "image.alt_text must not exceed 2000 characters",
        ),
    ],
)
def test_image_config_rejects_values_above_slack_limits(
    url_length: int,
    alt_text_length: int,
    expected_error: str,
) -> None:
    url_prefix = "https://example.com/"
    url = url_prefix + "u" * (url_length - len(url_prefix))

    with pytest.raises(ValueError, match=expected_error):
        cost_alerts.parse_cost_alert_config_json(
            _config_with_image(url=url, alt_text="a" * alt_text_length)
        )


def test_queries_normalize_blank_dimensions_to_missing() -> None:
    rule = _rule(include_missing=True)

    rule_sql = cost_alerts.build_rule_query(rule, schema=_SCHEMA)
    breakdown_sql = cost_alerts.build_breakdown_query(
        rule,
        scope_values=("__missing__",),
        schema=_SCHEMA,
    )

    normalized_email = "coalesce(nullif(agg.identity_email, ''), '__missing__')"
    assert f"{normalized_email} as alert_dimension_value" in rule_sql
    assert f"{normalized_email} as alert_scope_value" in breakdown_sql
    assert f"{normalized_email} as identity_email" in breakdown_sql


def test_queries_use_the_configured_redshift_schema() -> None:
    rule = _rule()

    comparison_sql = cost_alerts.build_rule_query(
        rule,
        schema="gateway_usage_prod",
    )
    breakdown_sql = cost_alerts.build_breakdown_query(
        rule,
        scope_values=("user@example.com",),
        schema="gateway_usage_prod",
    )

    for sql in (comparison_sql, breakdown_sql):
        assert "from gateway_usage_prod.usage_agg_1h" in sql
        assert "max(data_through_utc) as watermark_utc" in sql
        assert "load_batches" not in sql
        assert "from gateway_usage_prod.usage_agg_1h as agg" in sql
        assert "gateway_usage.load_batches" not in sql
        assert "gateway_usage.usage_agg_1h" not in sql


def test_daily_query_aggregates_before_joining_and_uses_current_pt_day() -> None:
    sql = cost_alerts.build_rule_query(_rule(), schema=_SCHEMA)

    assert "bucketed_source as (" in sql
    assert "as alert_bucket_start_utc" in sql
    assert "as alert_bucket_end_utc" in sql
    assert "as alert_dimension_value" in sql
    assert "group by\n    alert_bucket_start_utc" in sql
    assert "complete_buckets.bucket_end_utc = evaluation_window.bucket_start_utc" in sql
    assert "evaluation_window.bucket_start_utc - interval '1 day'" not in sql
    assert "watermark_utc::timestamp" in sql
    assert "convert_timezone('America/Los_Angeles', 'UTC', dateadd(day, -1," in sql
    assert "convert_timezone('America/Los_Angeles', 'UTC', dateadd(day, 1," in sql
    assert "+ interval '1 day'" not in sql


def test_queries_bound_source_rows_before_aggregation() -> None:
    rule = _rule()
    comparison_sql = cost_alerts.build_rule_query(rule, schema=_SCHEMA)
    breakdown_sql = cost_alerts.build_breakdown_query(
        rule,
        scope_values=("user@example.com",),
        schema=_SCHEMA,
    )

    for sql in (comparison_sql, breakdown_sql):
        assert "cross join evaluation_window" in sql
        assert "agg.bucket_start_utc < evaluation_window.bucket_end_utc" in sql
    assert (
        "agg.bucket_start_utc >= evaluation_window.previous_bucket_start_utc"
        in comparison_sql
    )
    assert "agg.bucket_start_utc >= evaluation_window.bucket_start_utc" in breakdown_sql


def test_hourly_query_uses_previous_bucket_end_adjacency() -> None:
    sql = cost_alerts.build_rule_query(_rule(grain="1h"), schema=_SCHEMA)

    assert "complete_buckets.bucket_end_utc = evaluation_window.bucket_start_utc" in sql
    assert "date_trunc('hour', watermark_utc - interval '1 hour')" in sql


@pytest.mark.parametrize(
    "group_by",
    ["provider_model", "benchmark", "identity_email"],
)
def test_scoped_evaluation_returns_one_truthful_candidate_per_breach(
    group_by: cost_alerts.CostAlertGroupBy,
) -> None:
    rule = _rule(group_by=group_by)
    rows = (
        _row("connor@example.com", current_cost="1400", current_requests=140),
        _row("hung@example.com", current_cost="1300", current_requests=130),
        _row("devin@example.com", current_cost="800", current_requests=80),
    )

    candidates = cost_alerts.evaluate_rule_rows(rule, rows)

    assert [candidate.scope_value for candidate in candidates] == [
        "connor@example.com",
        "hung@example.com",
    ]
    assert [candidate.total_current_cost_usd for candidate in candidates] == [
        Decimal("1400"),
        Decimal("1300"),
    ]
    assert [candidate.total_current_request_count for candidate in candidates] == [
        140,
        130,
    ]


def test_global_evaluation_returns_one_global_candidate() -> None:
    candidates = cost_alerts.evaluate_rule_rows(
        _rule(group_by="global"),
        (_row("all", current_cost="1400", current_requests=140),),
    )

    assert len(candidates) == 1
    assert candidates[0].scope_value is None
    assert candidates[0].total_current_cost_usd == Decimal("1400")


@pytest.mark.parametrize(
    ("current_cost", "expected_count"),
    [("999.99", 0), ("1000", 1), ("1000.01", 1)],
)
def test_threshold_boundary_is_inclusive(
    current_cost: str,
    expected_count: int,
) -> None:
    candidates = cost_alerts.evaluate_rule_rows(
        _rule(),
        (_row("connor@example.com", current_cost=current_cost),),
    )

    assert len(candidates) == expected_count


def test_percent_rule_requires_minimum_baseline_and_percent_increase() -> None:
    rule = cost_alerts.CostAlertRule(
        name="percent",
        grain="1h",
        group_by="provider_model",
        threshold_usd=Decimal("100"),
        min_previous_usd=Decimal("50"),
        percent_increase=Decimal("20"),
    )

    candidates = cost_alerts.evaluate_rule_rows(
        rule,
        (
            _row("below-baseline", current_cost="120", previous_cost="10"),
            _row("below-percent", current_cost="119", previous_cost="100"),
            _row("triggered", current_cost="120", previous_cost="100"),
        ),
    )

    assert [candidate.scope_value for candidate in candidates] == ["triggered"]


def test_missing_scope_follows_include_missing() -> None:
    row = _row("__missing__", current_cost="1400")

    assert cost_alerts.evaluate_rule_rows(_rule(), (row,)) == ()
    assert (
        cost_alerts.evaluate_rule_rows(_rule(include_missing=True), (row,))[
            0
        ].scope_value
        == "__missing__"
    )


def test_config_rejects_resolved_image_attachment() -> None:
    config = {
        "rules": [
            {
                "name": "email-hour",
                "grain": "1h",
                "group_by": "identity_email",
                "threshold_usd": "1000",
                "image": {
                    "url": "https://example.com/cost.png",
                    "attach_to": ["resolved"],
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="attach_to"):
        cost_alerts.parse_cost_alert_config_json(json.dumps(config))


def test_breakdown_query_batches_selected_scopes_and_shares_window() -> None:
    sql = cost_alerts.build_breakdown_query(
        _rule(),
        scope_values=("connor@example.com", "hung@example.com"),
        schema=_SCHEMA,
    )

    assert "alert_scope_value" in sql
    assert "partition by alert_scope_value, group_by" in sql
    assert "where group_rank <= 3" in sql
    assert "agg.bucket_start_utc >= evaluation_window.bucket_start_utc" in sql
    assert "previous_window as" not in sql
    assert "previous_breakdowns as" not in sql
    assert "full outer join" not in sql
    assert "previous_cost_usd" not in sql
    assert "absolute_increase_usd" not in sql
    assert "percent_increase" not in sql
    assert "'connor@example.com'" in sql
    assert "'hung@example.com'" in sql
    assert "current_bucket as" not in sql


def test_breakdown_parser_keeps_scopes_independent() -> None:
    metadata = [
        {"name": "alert_scope_value"},
        {"name": "group_by"},
        {"name": "dimension_value"},
        {"name": "current_request_count"},
        {"name": "current_cost_usd"},
    ]

    def record(scope: str, model: str, cost: str) -> list[dict[str, object]]:
        values: list[object] = [
            scope,
            "provider_model",
            model,
            10,
            cost,
        ]
        return [
            {"longValue": value}
            if isinstance(value, int)
            else {"isNull": True}
            if value is None
            else {"stringValue": str(value)}
            for value in values
        ]

    breakdowns = cost_alerts.parse_breakdown_rows(
        {
            "ColumnMetadata": metadata,
            "Records": [
                record("connor@example.com", "model-a", "10"),
                record("hung@example.com", "model-b", "20"),
            ],
        }
    )

    assert [(item.scope_value, item.group_by) for item in breakdowns] == [
        ("connor@example.com", "provider_model"),
        ("hung@example.com", "provider_model"),
    ]


def test_comparison_parser_decodes_data_api_values() -> None:
    columns = (
        "bucket_start_utc",
        "bucket_end_utc",
        "data_through_utc",
        "watermark_utc",
        "dimension_value",
        "current_request_count",
        "current_cost_usd",
        "previous_cost_usd",
        "absolute_increase_usd",
        "percent_increase",
    )
    values: tuple[object, ...] = (
        "2026-07-10 00:00:00-07:00",
        "2026-07-11T07:00:00Z",
        "2026-07-10 20:00:00+00:00",
        "2026-07-10 20:05:00+00:00",
        "anthropic/model",
        42,
        "12.50",
        "0",
        "12.50",
        None,
    )
    record = [
        {"longValue": value}
        if isinstance(value, int)
        else {"isNull": True}
        if value is None
        else {"stringValue": str(value)}
        for value in values
    ]

    rows = cost_alerts.parse_comparison_rows(
        {
            "ColumnMetadata": [{"name": column} for column in columns],
            "Records": [record],
        }
    )

    assert rows == (
        cost_alerts.ComparisonRow(
            bucket_start_utc=datetime.fromisoformat("2026-07-10T00:00:00-07:00"),
            bucket_end_utc=datetime(2026, 7, 11, 7, tzinfo=UTC),
            data_through_utc=datetime(2026, 7, 10, 20, tzinfo=UTC),
            watermark_utc=datetime(2026, 7, 10, 20, 5, tzinfo=UTC),
            dimension_value="anthropic/model",
            current_request_count=42,
            current_cost_usd=Decimal("12.50"),
            previous_cost_usd=Decimal("0"),
            absolute_increase_usd=Decimal("12.50"),
            percent_increase=None,
        ),
    )
