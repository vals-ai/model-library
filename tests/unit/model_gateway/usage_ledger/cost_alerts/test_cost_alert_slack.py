from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
import json
import traceback
from typing import cast

import pytest

from model_gateway.usage_ledger.cost_alerts import rules as cost_alerts
from model_gateway.usage_ledger.cost_alerts import slack


def _candidate(
    *,
    scope_value: str,
    current_cost: str,
    eligible_requests: int = 140,
    group_by: cost_alerts.CostAlertGroupBy = "identity_email",
    breakdowns: tuple[cost_alerts.AlertBreakdown, ...] = (),
) -> cost_alerts.AlertCandidate:
    return cost_alerts.AlertCandidate(
        rule_name=f"dev-{group_by}-1d",
        grain="1d",
        group_by=group_by,
        scope_value=scope_value,
        bucket_start_utc=datetime(2026, 7, 10, 7, tzinfo=UTC),
        bucket_end_utc=datetime(2026, 7, 11, 7, tzinfo=UTC),
        data_through_utc=datetime(2026, 7, 10, 20, tzinfo=UTC),
        threshold_usd=Decimal("1000"),
        total_current_cost_usd=Decimal(current_cost),
        total_current_request_count=140,
        total_current_eligible_request_count=eligible_requests,
        breakdowns=breakdowns,
    )


def _fallback_text(payload: dict[str, object]) -> str:
    attachments = cast(list[dict[str, object]], payload["attachments"])
    return cast(str, attachments[0]["fallback"])


def _section_texts(payload: dict[str, object]) -> list[str]:
    attachments = cast(list[dict[str, object]], payload["attachments"])
    blocks = cast(list[dict[str, object]], attachments[0]["blocks"])
    return [
        cast(str, cast(dict[str, object], block["text"])["text"])
        for block in blocks
        if block["type"] == "section" and "text" in block
    ]


def test_active_fallback_uses_explicit_scope_and_scoped_total() -> None:
    payload = slack.build_alert_payload(
        stage="dev",
        candidate=_candidate(
            scope_value="connor@example.com",
            current_cost="1400",
        ),
    )

    fallback_text = _fallback_text(payload)
    assert "Email: connor@example.com" in fallback_text
    assert "$1400.00 / $1000.00" in fallback_text


@pytest.mark.parametrize(
    ("group_by", "scope_value", "expected_scope"),
    [
        ("provider_model", "anthropic/model", "Model: anthropic/model"),
        ("benchmark", "safety-eval", "Benchmark: safety-eval"),
        ("identity_email", "connor@example.com", "Email: connor@example.com"),
    ],
)
def test_block_payloads_use_explicit_scope_for_every_scoped_group(
    group_by: cost_alerts.CostAlertGroupBy,
    scope_value: str,
    expected_scope: str,
) -> None:
    candidate = _candidate(
        scope_value=scope_value,
        current_cost="1400",
        group_by=group_by,
    )
    payloads = (
        slack.build_alert_payload(stage="dev", candidate=candidate),
        slack.build_worsening_payload(stage="dev", candidate=candidate),
    )

    for payload in payloads:
        attachments = cast(list[dict[str, object]], payload["attachments"])
        blocks = cast(list[dict[str, object]], attachments[0]["blocks"])
        assert expected_scope in json.dumps(blocks)


def test_active_payload_renders_current_breakdown() -> None:
    candidate = _candidate(
        scope_value="connor@example.com",
        current_cost="1400",
        breakdowns=(
            cost_alerts.AlertBreakdown(
                scope_value="connor@example.com",
                group_by="provider_model",
                top_contributors=(
                    cost_alerts.AlertContributor(
                        display_value="anthropic/model",
                        current_request_count=70,
                        current_cost_usd=Decimal("700"),
                    ),
                ),
            ),
        ),
    )

    payload = slack.build_alert_payload(stage="dev", candidate=candidate)

    assert "Top models" in json.dumps(payload)
    assert "anthropic/model" in json.dumps(payload)
    assert "$700.00" in json.dumps(payload)
    assert "50%" in json.dumps(payload)


def test_cost_per_request_excludes_ignored_model_requests() -> None:
    candidate = _candidate(
        scope_value="connor@example.com",
        current_cost="1400",
        eligible_requests=70,
    )

    payload_text = json.dumps(
        slack.build_alert_payload(stage="dev", candidate=candidate)
    )

    assert "140 reqs" in payload_text
    assert "$20.000/req" in payload_text
    assert "$10.000/req" not in payload_text


def test_active_payload_keeps_ignored_model_in_rank_and_labels_it() -> None:
    ignored_contributor = cost_alerts.AlertContributor(
        display_value="openai/gpt-4o",
        current_request_count=21_231,
        current_cost_usd=Decimal("14931.68"),
        ignored_for_cost=True,
    )
    candidate = _candidate(
        scope_value="connor@example.com",
        current_cost="2000",
        breakdowns=(
            cost_alerts.AlertBreakdown(
                scope_value="connor@example.com",
                group_by="provider_model",
                top_contributors=(
                    ignored_contributor,
                    cost_alerts.AlertContributor(
                        display_value="anthropic/model",
                        current_request_count=70,
                        current_cost_usd=Decimal("1200"),
                    ),
                ),
            ),
        ),
    )

    payload = slack.build_alert_payload(stage="dev", candidate=candidate)
    breakdown = next(
        text for text in _section_texts(payload) if text.startswith("*Top models*")
    )

    assert "1. *openai/gpt-4o* — *$14931.68* · 21,231 reqs [Ignored]" in breakdown
    assert "2. *anthropic/model* — *$1200.00* (60%) · 70 reqs" in breakdown
    ignored_line = breakdown.splitlines()[1]
    assert "%" not in ignored_line
    assert "openai/gpt-4o $14931.68 [Ignored]" in _fallback_text(payload)


def test_image_attachment_targets_active_and_worsening_only() -> None:
    image = cost_alerts.CostAlertImageConfig(
        url="https://example.com/cost.png",
        alt_text="cost alert",
        attach_to=("worsening",),
        min_threshold_multiple=Decimal("1.25"),
    )
    candidate = replace(
        _candidate(scope_value="connor@example.com", current_cost="1400"),
        image=image,
    )

    active = json.dumps(slack.build_alert_payload(stage="dev", candidate=candidate))
    worsening = json.dumps(
        slack.build_worsening_payload(stage="dev", candidate=candidate)
    )

    assert '"type": "image"' not in active
    assert '"type": "image"' in worsening


def test_image_attachment_respects_minimum_threshold_multiple() -> None:
    image = cost_alerts.CostAlertImageConfig(
        url="https://example.com/cost.png",
        alt_text="cost alert",
        attach_to=("active",),
        min_threshold_multiple=Decimal("1.5"),
    )
    candidate = replace(
        _candidate(scope_value="connor@example.com", current_cost="1400"),
        image=image,
    )

    payload = json.dumps(slack.build_alert_payload(stage="dev", candidate=candidate))

    assert '"type": "image"' not in payload


def test_dashboard_action_is_omitted_when_url_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slack, "_dashboard_url", "")

    payload = json.dumps(
        slack.build_alert_payload(
            stage="dev",
            candidate=_candidate(scope_value="user@example.com", current_cost="1400"),
        )
    )

    assert '"type": "actions"' not in payload
    assert "Open Dashboard" not in payload


def test_missing_scope_fallback_uses_human_readable_label() -> None:
    payload = slack.build_alert_payload(
        stage="dev",
        candidate=_candidate(scope_value="__missing__", current_cost="1400"),
    )

    fallback_text = _fallback_text(payload)
    assert "Email: unattributed" in fallback_text
    assert "__missing__" not in fallback_text


def test_breakdown_section_respects_slack_text_limit() -> None:
    contributors = tuple(
        cost_alerts.AlertContributor(
            display_value=f"{index}{'m' * 2047}",
            current_request_count=10,
            current_cost_usd=Decimal("100"),
        )
        for index in range(3)
    )
    candidate = _candidate(
        scope_value="connor@example.com",
        current_cost="300",
        breakdowns=(
            cost_alerts.AlertBreakdown(
                scope_value="connor@example.com",
                group_by="provider_model",
                top_contributors=contributors,
            ),
        ),
    )

    sections = _section_texts(
        slack.build_alert_payload(stage="dev", candidate=candidate)
    )
    breakdown = next(text for text in sections if text.startswith("*Top models*"))

    assert len(breakdown) == 3000
    assert breakdown.endswith("…")
    assert all(len(text) <= 3000 for text in sections)


def test_malformed_webhook_url_is_redacted() -> None:
    capability_token = "SECRET-CAPABILITY"
    webhook_url = f"hooks.slack.com/services/T000/B000/{capability_token}"

    with pytest.raises(slack.SlackWebhookError) as error:
        slack._post_webhook(webhook_url, b"{}", 0.1)

    rendered = "".join(traceback.format_exception(error.value))
    assert capability_token not in rendered
    assert webhook_url not in rendered
    assert error.value.__cause__ is None


def test_webhook_transport_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_token = "SECRET-CAPABILITY"
    webhook_url = f"https://hooks.slack.com/services/T000/B000/{capability_token}"

    def fail_post(_request: object, *, timeout: float) -> None:
        raise OSError(f"transport failed for {webhook_url} after {timeout}")

    monkeypatch.setattr(slack.urllib.request, "urlopen", fail_post)

    with pytest.raises(slack.SlackWebhookError) as error:
        slack._post_webhook(webhook_url, b"{}", 0.1)

    rendered = "".join(traceback.format_exception(error.value))
    assert capability_token not in rendered
    assert webhook_url not in rendered
    assert error.value.__cause__ is None
