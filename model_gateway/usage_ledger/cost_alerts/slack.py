from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import http.client
import json
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from model_gateway.usage_ledger.cost_alerts.rules import (
    AlertBreakdown,
    AlertCandidate,
    AlertContributor,
    CostAlertBreakdownGroupBy,
    CostAlertImageAttachTo,
    display_dimension_value,
)

SlackBlock = dict[str, object]
SlackPayload = dict[str, object]

_BREAKDOWN_ORDER: tuple[CostAlertBreakdownGroupBy, ...] = (
    "provider_model",
    "benchmark",
    "agent",
    "identity_email",
)
_HORIZON_LABEL_BY_GRAIN: dict[str, str] = {
    "1h": "Hourly",
    "1d": "Daily",
}
_HORIZON_ICON_BY_GRAIN: dict[str, str] = {
    "1h": "⏱️",
    "1d": "📅",
}
_ACTIVE_COLOR = "#ECB22E"
_WORSENING_COLOR = "#FF7A00"
_SEVERE_COLOR = "#E01E5A"
_CRITICAL_COLOR = "#7C3AED"
_SECTION_TEXT_MAX_LENGTH = 3000
_dashboard_url = ""
_PACIFIC = ZoneInfo("America/Los_Angeles")


class SlackWebhookError(RuntimeError):
    pass


@dataclass
class SlackWebhookSender:
    webhook_url: str
    post: Callable[[str, bytes, float], int] | None = None
    timeout_seconds: float = 5.0

    def post_payload(self, payload: SlackPayload) -> None:
        body = _json_payload(payload)
        if self.post is not None:
            status_code = self.post(self.webhook_url, body, self.timeout_seconds)
        else:
            status_code = _post_webhook(self.webhook_url, body, self.timeout_seconds)
        if status_code < 200 or status_code >= 300:
            raise SlackWebhookError(f"Slack webhook returned HTTP {status_code}")


def build_alert_payload(*, stage: str, candidate: AlertCandidate) -> SlackPayload:
    return _build_active_payload(
        alert_status="active",
        status_emoji=":rotating_light:",
        stage=stage,
        candidate=candidate,
    )


def build_worsening_payload(*, stage: str, candidate: AlertCandidate) -> SlackPayload:
    return _build_active_payload(
        alert_status="worsening",
        status_emoji=":warning:",
        stage=stage,
        candidate=candidate,
    )


def _build_active_payload(
    *,
    alert_status: CostAlertImageAttachTo,
    status_emoji: str,
    stage: str,
    candidate: AlertCandidate,
) -> SlackPayload:
    fallback_text = _alert_fallback_text(
        status_emoji=status_emoji,
        stage=stage,
        candidate=candidate,
    )
    blocks = [
        _header(_alert_header_text(status_emoji=status_emoji, candidate=candidate)),
        _section(
            "\n".join(
                [
                    _scope_summary_text(candidate),
                    _alert_summary_plain_text(
                        status_emoji=status_emoji,
                        candidate=candidate,
                    ),
                    _context_text(stage=stage),
                ]
            )
        ),
    ]
    image_block = _configured_image(alert_status=alert_status, candidate=candidate)
    if image_block is not None:
        blocks.append(image_block)
    blocks.extend(
        [
            _section_fields(
                [
                    f"*Traffic*\n{candidate.total_current_request_count:,} reqs",
                    f"*Cost / req*\n{_format_cost_per_request(candidate.total_current_cost_usd, candidate.total_current_request_count)}",
                ]
            ),
            _section(f"*Window:* {_format_window(candidate)}"),
        ]
    )
    if _dashboard_url:
        blocks.append(_dashboard_action())
    blocks.append(_divider())
    blocks.extend(_breakdown_blocks(candidate))
    return _attachment_payload(
        fallback_text=fallback_text,
        color=_alert_color(status_emoji=status_emoji, candidate=candidate),
        blocks=blocks,
    )


def _post_webhook(webhook_url: str, body: bytes, timeout_seconds: float) -> int:
    try:
        request = urllib.request.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return int(getattr(response, "status", response.getcode()))
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (OSError, ValueError, http.client.HTTPException):
        raise SlackWebhookError("Slack webhook request failed") from None


def _json_payload(payload: SlackPayload) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _alert_fallback_text(
    *, status_emoji: str, stage: str, candidate: AlertCandidate
) -> str:
    lines = [
        _alert_header_text(status_emoji=status_emoji, candidate=candidate),
        f"{_plain_scope_summary_text(candidate)} · {_alert_summary_plain_text(status_emoji=status_emoji, candidate=candidate)} · {_context_text(stage=stage)} · rule {_escape_mrkdwn(candidate.rule_name)}",
    ]
    lines.extend(_breakdown_fallback_lines(candidate))
    return "\n".join(lines)


def _alert_color(*, status_emoji: str, candidate: AlertCandidate) -> str:
    if status_emoji != ":warning:":
        return _ACTIVE_COLOR
    threshold_multiple = _threshold_multiple(candidate)
    if threshold_multiple >= Decimal("3"):
        return _CRITICAL_COLOR
    if threshold_multiple >= Decimal("2"):
        return _SEVERE_COLOR
    return _WORSENING_COLOR


def _attachment_payload(
    *, fallback_text: str, color: str, blocks: list[SlackBlock]
) -> SlackPayload:
    return {
        "attachments": [{"color": color, "fallback": fallback_text, "blocks": blocks}]
    }


def _header(text: str) -> SlackBlock:
    return {"type": "header", "text": {"type": "plain_text", "text": text}}


def _section(text: str) -> SlackBlock:
    if len(text) > _SECTION_TEXT_MAX_LENGTH:
        text = f"{text[: _SECTION_TEXT_MAX_LENGTH - 1]}…"
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _section_fields(fields: list[str]) -> SlackBlock:
    return {
        "type": "section",
        "fields": [{"type": "mrkdwn", "text": field} for field in fields],
    }


def _divider() -> SlackBlock:
    return {"type": "divider"}


def _dashboard_action() -> SlackBlock:
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Open Dashboard"},
                "url": _dashboard_url,
            }
        ],
    }


def _configured_image(
    *, alert_status: CostAlertImageAttachTo, candidate: AlertCandidate
) -> SlackBlock | None:
    image = candidate.image
    if image is None:
        return None
    if alert_status not in image.attach_to:
        return None
    if (
        image.min_threshold_multiple is not None
        and _threshold_multiple(candidate) < image.min_threshold_multiple
    ):
        return None
    return {"type": "image", "image_url": image.url, "alt_text": image.alt_text}


def _threshold_multiple(candidate: AlertCandidate) -> Decimal:
    return candidate.total_current_cost_usd / candidate.threshold_usd


def _alert_header_text(*, status_emoji: str, candidate: AlertCandidate) -> str:
    label = _horizon_label(candidate)
    header = (
        f"{_horizon_icon(candidate)} {label} | Triggered | "
        f"{_format_usd(candidate.total_current_cost_usd)} / {_format_usd(candidate.threshold_usd)} limit"
    )
    if status_emoji == ":warning:":
        header = (
            f"{_horizon_icon(candidate)} {label} | Worsening: {_format_threshold_multiple(candidate)}x | "
            f"{_format_usd(candidate.total_current_cost_usd)} / {_format_usd(candidate.threshold_usd)} limit"
        )
    return header


def _scope_summary_text(candidate: AlertCandidate) -> str:
    return f"*{_escape_mrkdwn(_plain_scope_summary_text(candidate))}*"


def _plain_scope_summary_text(candidate: AlertCandidate) -> str:
    label = _scope_label(candidate.group_by)
    value = _scope_value(candidate)
    if value is None:
        return label
    return f"{label}: {value}"


def _scope_label(group_by: str) -> str:
    if group_by == "provider_model":
        return "Model"
    if group_by == "benchmark":
        return "Benchmark"
    if group_by == "identity_email":
        return "Email"
    return "Global threshold"


def _scope_value(candidate: AlertCandidate) -> str | None:
    if candidate.scope_value is None:
        return None
    return display_dimension_value(candidate.scope_value)


def _alert_summary_plain_text(*, status_emoji: str, candidate: AlertCandidate) -> str:
    threshold_delta = candidate.total_current_cost_usd - candidate.threshold_usd
    if status_emoji == ":warning:":
        return f"*{_format_signed_usd(threshold_delta)} over limit*"
    return (
        f"*{_format_signed_usd(threshold_delta)} over limit* · "
        f"{_format_percent_over_limit(candidate)} over"
    )


def _context_text(*, stage: str) -> str:
    return _escape_mrkdwn(stage)


def _horizon_icon(candidate: AlertCandidate) -> str:
    return _HORIZON_ICON_BY_GRAIN[candidate.grain]


def _horizon_label(candidate: AlertCandidate) -> str:
    return _HORIZON_LABEL_BY_GRAIN[candidate.grain]


def _breakdown_blocks(candidate: AlertCandidate) -> list[SlackBlock]:
    breakdowns = _breakdown_map(candidate)
    blocks: list[SlackBlock] = []
    for group_by in _BREAKDOWN_ORDER:
        breakdown = breakdowns.get(group_by)
        if breakdown is None or not breakdown.top_contributors:
            continue
        contributors = _contributors_text(
            breakdown.top_contributors,
            total_cost=candidate.total_current_cost_usd,
        )
        blocks.append(_section(f"*Top {_breakdown_label(group_by)}*\n{contributors}"))
    return blocks


def _breakdown_fallback_lines(candidate: AlertCandidate) -> list[str]:
    breakdowns = _breakdown_map(candidate)
    lines: list[str] = []
    for group_by in _BREAKDOWN_ORDER:
        breakdown = breakdowns.get(group_by)
        contributors: tuple[AlertContributor, ...] = ()
        if breakdown is not None:
            contributors = breakdown.top_contributors
        lines.append(
            f"Top {_breakdown_label(group_by)}: {_compact_contributors_text(contributors)}"
        )
    return lines


def _breakdown_map(
    candidate: AlertCandidate,
) -> dict[CostAlertBreakdownGroupBy, AlertBreakdown]:
    return {breakdown.group_by: breakdown for breakdown in candidate.breakdowns}


def _breakdown_label(group_by: CostAlertBreakdownGroupBy) -> str:
    if group_by == "provider_model":
        return "models"
    if group_by == "benchmark":
        return "benchmarks"
    if group_by == "agent":
        return "agents"
    if group_by == "identity_email":
        return "emails"
    raise ValueError(f"Unsupported breakdown group: {group_by}")


def _compact_contributors_text(contributors: tuple[AlertContributor, ...]) -> str:
    if not contributors:
        return "No data"
    return "; ".join(
        f"{contributor.display_value} {_format_usd(contributor.current_cost_usd)}"
        for contributor in contributors
    )


def _contributors_text(
    contributors: tuple[AlertContributor, ...], *, total_cost: Decimal
) -> str:
    lines: list[str] = []
    for index, contributor in enumerate(contributors, start=1):
        lines.append(
            f"{index}. *{_escape_mrkdwn(contributor.display_value)}* — "
            f"*{_format_usd(contributor.current_cost_usd)}* "
            f"({_format_share(contributor.current_cost_usd, total_cost)}) · "
            f"{contributor.current_request_count:,} reqs"
        )
    return "\n".join(lines)


def _escape_mrkdwn(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_window(candidate: AlertCandidate) -> str:
    start_pt = candidate.bucket_start_utc.astimezone(_PACIFIC)
    if candidate.grain == "1d":
        end_pt = min(
            candidate.data_through_utc,
            candidate.bucket_end_utc,
        ).astimezone(_PACIFIC)
    else:
        end_pt = candidate.bucket_end_utc.astimezone(_PACIFIC)
    if start_pt.date() == end_pt.date():
        return f"{start_pt.strftime('%b %-d, %H:%M')} → {end_pt.strftime('%H:%M')} PT"
    return (
        f"{start_pt.strftime('%b %-d, %H:%M')} → {end_pt.strftime('%b %-d, %H:%M')} PT"
    )


def _format_usd(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"


def _format_threshold_multiple(candidate: AlertCandidate) -> str:
    return f"{_threshold_multiple(candidate).quantize(Decimal('0.01'))}"


def _format_percent_over_limit(candidate: AlertCandidate) -> str:
    return _format_signed_percent((_threshold_multiple(candidate) - 1) * Decimal("100"))


def _format_signed_percent(value: Decimal) -> str:
    formatted = f"{abs(value).quantize(Decimal('1'))}%"
    if value >= 0:
        return f"+{formatted}"
    return f"-{formatted}"


def _format_signed_usd(value: Decimal) -> str:
    amount = _format_usd(abs(value))
    if value >= 0:
        return f"+{amount}"
    return f"-{amount}"


def _format_cost_per_request(cost: Decimal, request_count: int) -> str:
    if request_count <= 0:
        return "n/a"
    return f"${(cost / Decimal(request_count)).quantize(Decimal('0.000'))}/req"


def _format_share(cost: Decimal, total_cost: Decimal) -> str:
    if total_cost <= 0:
        return "0%"
    return f"{((cost / total_cost) * Decimal('100')).quantize(Decimal('1'))}%"
