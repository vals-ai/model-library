"""Scheduled Lambda that evaluates gateway cost alerts and posts to Slack."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, Literal, cast

import boto3  # pyright: ignore[reportMissingImports]

import model_gateway.usage_ledger.cost_alerts.rules as cost_alerts
import model_gateway.usage_ledger.cost_alerts.slack as cost_alert_slack

_QUERY_POLL_INTERVAL_SECONDS = 1.0
_MAX_CONCURRENT_RULE_QUERIES = 8
_STATE_TTL_DAYS = 30
_ESCALATION_STEP = Decimal("0.25")
_MODEL_REGISTRY_PATH: Final = (
    Path(__file__).resolve().parents[3] / "model_library" / "config" / "all_models.json"
)


@dataclass(frozen=True)
class AlertState:
    pk: str
    last_alerted_bucket: datetime
    last_notified_level: int


def _ignored_for_cost_model_keys() -> tuple[str, ...]:
    try:
        registry = cast(
            dict[str, dict[str, Any]],
            json.loads(_MODEL_REGISTRY_PATH.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load generated model registry: {_MODEL_REGISTRY_PATH}"
        ) from exc
    return tuple(
        model_key
        for model_key, config in registry.items()
        if cast(dict[str, object], config["metadata"])["ignored_for_cost"]
    )


def handler(_event: Mapping[str, object], _context: object) -> None:
    rules = cost_alerts.parse_cost_alert_config_json(
        _parameter_value(_env("GATEWAY_COST_ALERT_CONFIG_PARAMETER_NAME"))
    )
    table = _state_table(_env("GATEWAY_COST_ALERT_STATE_TABLE_NAME"))
    sender = _slack_sender(
        _secret_value(_env("GATEWAY_COST_ALERT_SLACK_WEBHOOK_SECRET_NAME"))
    )
    workgroup_name = _env("GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME")
    database_name = _env("GATEWAY_USAGE_REDSHIFT_DATABASE_NAME")
    schema = _env("GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME")
    stage = _env("GATEWAY_STAGE")
    ignored_model_keys = _ignored_for_cost_model_keys()
    redshift_client = _redshift_client()
    query_results = _query_rules(
        redshift_client=redshift_client,
        workgroup_name=workgroup_name,
        database_name=database_name,
        ignored_model_keys=ignored_model_keys,
        schema=schema,
        rules=rules,
    )
    for rule, comparison_result in query_results:
        _process_rule(
            table=table,
            sender=sender,
            redshift_client=redshift_client,
            workgroup_name=workgroup_name,
            database_name=database_name,
            ignored_model_keys=ignored_model_keys,
            schema=schema,
            stage=stage,
            rule=rule,
            comparison_result=comparison_result,
        )


def _process_rule(
    *,
    table: Any,
    sender: cost_alert_slack.SlackWebhookSender,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    ignored_model_keys: Sequence[str],
    stage: str,
    rule: cost_alerts.CostAlertRule,
    comparison_result: Mapping[str, object],
    schema: str,
) -> None:
    decisions_by_scope: dict[
        str | None, tuple[Literal["active", "worsening"], int]
    ] = {}
    notification_candidates: list[cost_alerts.AlertCandidate] = []
    for candidate in cost_alerts.evaluate_rule_rows(
        rule, cost_alerts.parse_comparison_rows(comparison_result)
    ):
        state = _load_state(table, _incident_key(stage, rule, candidate.scope_value))
        level = _notification_level(rule, candidate)
        kind = _notification_kind(state, candidate, level)
        if kind is not None:
            decisions_by_scope[candidate.scope_value] = (kind, level)
            notification_candidates.append(candidate)

    for candidate in _add_required_breakdowns(
        redshift_client=redshift_client,
        workgroup_name=workgroup_name,
        database_name=database_name,
        ignored_model_keys=ignored_model_keys,
        schema=schema,
        rule=rule,
        candidates=notification_candidates,
    ):
        kind, level = decisions_by_scope[candidate.scope_value]
        _process_breached_candidate(
            table=table,
            sender=sender,
            stage=stage,
            rule=rule,
            candidate=candidate,
            notification_kind=kind,
            notified_level=level,
        )


def _add_required_breakdowns(
    *,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    ignored_model_keys: Sequence[str],
    rule: cost_alerts.CostAlertRule,
    candidates: Sequence[cost_alerts.AlertCandidate],
    schema: str,
) -> tuple[cost_alerts.AlertCandidate, ...]:
    if not candidates:
        return ()
    breakdowns = cost_alerts.parse_breakdown_rows(
        _query_breakdowns(
            redshift_client=redshift_client,
            workgroup_name=workgroup_name,
            database_name=database_name,
            ignored_model_keys=ignored_model_keys,
            schema=schema,
            rule=rule,
            candidates=candidates,
        )
    )
    breakdowns_by_scope: dict[str, list[cost_alerts.AlertBreakdown]] = {}
    for breakdown in breakdowns:
        breakdowns_by_scope.setdefault(breakdown.scope_value, []).append(breakdown)
    return tuple(
        replace(
            candidate,
            breakdowns=tuple(
                breakdowns_by_scope.get(
                    "all" if candidate.scope_value is None else candidate.scope_value,
                    (),
                )
            ),
        )
        for candidate in candidates
    )


def _notification_kind(
    state: AlertState | None,
    candidate: cost_alerts.AlertCandidate,
    notified_level: int,
) -> Literal["active", "worsening"] | None:
    if state is None or candidate.bucket_end_utc > state.last_alerted_bucket:
        return "active"
    if (
        candidate.bucket_end_utc == state.last_alerted_bucket
        and notified_level > state.last_notified_level
    ):
        return "worsening"
    return None


def _process_breached_candidate(
    *,
    table: Any,
    sender: cost_alert_slack.SlackWebhookSender,
    stage: str,
    rule: cost_alerts.CostAlertRule,
    candidate: cost_alerts.AlertCandidate,
    notification_kind: Literal["active", "worsening"],
    notified_level: int,
) -> None:
    if notification_kind == "active":
        sender.post_payload(
            cost_alert_slack.build_alert_payload(stage=stage, candidate=candidate)
        )
    else:
        sender.post_payload(
            cost_alert_slack.build_worsening_payload(
                stage=stage,
                candidate=candidate,
            )
        )
    _write_state(
        table,
        AlertState(
            pk=_incident_key(stage, rule, candidate.scope_value),
            last_alerted_bucket=candidate.bucket_end_utc,
            last_notified_level=notified_level,
        ),
    )


def _query_rules(
    *,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    ignored_model_keys: Sequence[str],
    rules: Sequence[cost_alerts.CostAlertRule],
    schema: str,
) -> Iterator[tuple[cost_alerts.CostAlertRule, dict[str, object]]]:
    for batch_start in range(0, len(rules), _MAX_CONCURRENT_RULE_QUERIES):
        batch = rules[batch_start : batch_start + _MAX_CONCURRENT_RULE_QUERIES]
        pending = {
            index: _start_statement(
                redshift_client=redshift_client,
                workgroup_name=workgroup_name,
                database_name=database_name,
                sql=cost_alerts.build_rule_query(
                    rule,
                    ignored_model_keys=ignored_model_keys,
                    schema=schema,
                ),
                statement_name=f"gateway-cost-alert-{rule.name}",
            )
            for index, rule in enumerate(batch)
        }
        outcomes: dict[int, dict[str, object] | RuntimeError] = {}
        while pending:
            for index, statement_id in tuple(pending.items()):
                terminal_outcome = _terminal_statement_outcome(
                    redshift_client, statement_id
                )
                if terminal_outcome is None:
                    continue
                del pending[index]
                outcomes[index] = (
                    _statement_result(redshift_client, statement_id)
                    if terminal_outcome == "FINISHED"
                    else terminal_outcome
                )
            if pending:
                time.sleep(_QUERY_POLL_INTERVAL_SECONDS)

        for index, rule in enumerate(batch):
            outcome = outcomes[index]
            if isinstance(outcome, Exception):
                raise outcome
            yield rule, outcome


def _query_breakdowns(
    *,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    ignored_model_keys: Sequence[str],
    rule: cost_alerts.CostAlertRule,
    candidates: Sequence[cost_alerts.AlertCandidate],
    schema: str,
) -> dict[str, object]:
    return _query_statement(
        redshift_client=redshift_client,
        workgroup_name=workgroup_name,
        database_name=database_name,
        sql=cost_alerts.build_breakdown_query(
            rule,
            ignored_model_keys=ignored_model_keys,
            scope_values=tuple(
                candidate.scope_value
                for candidate in candidates
                if candidate.scope_value is not None
            ),
            schema=schema,
        ),
        statement_name=f"gateway-cost-alert-breakdown-{rule.name}",
    )


def _start_statement(
    *,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    sql: str,
    statement_name: str,
) -> str:
    return cast(
        str,
        redshift_client.execute_statement(
            WorkgroupName=workgroup_name,
            Database=database_name,
            Sql=sql,
            StatementName=statement_name,
        )["Id"],
    )


def _terminal_statement_outcome(
    redshift_client: Any,
    statement_id: str,
) -> Literal["FINISHED"] | RuntimeError | None:
    response = cast(
        Mapping[str, object],
        redshift_client.describe_statement(Id=statement_id),
    )
    status = cast(str, response["Status"])
    if status == "FINISHED":
        return "FINISHED"
    if status in {"FAILED", "ABORTED"}:
        return RuntimeError(str(response["Error"]))
    return None


def _query_statement(
    *,
    redshift_client: Any,
    workgroup_name: str,
    database_name: str,
    sql: str,
    statement_name: str,
) -> dict[str, object]:
    statement_id = _start_statement(
        redshift_client=redshift_client,
        workgroup_name=workgroup_name,
        database_name=database_name,
        sql=sql,
        statement_name=statement_name,
    )
    while True:
        terminal_outcome = _terminal_statement_outcome(redshift_client, statement_id)
        if terminal_outcome == "FINISHED":
            return _statement_result(redshift_client, statement_id)
        if isinstance(terminal_outcome, RuntimeError):
            raise terminal_outcome
        time.sleep(_QUERY_POLL_INTERVAL_SECONDS)


def _statement_result(
    redshift_client: Any,
    statement_id: str,
) -> dict[str, object]:
    records: list[object] = []
    column_metadata: object = []
    next_token: str | None = None
    first_page = True
    while True:
        kwargs: dict[str, object] = {"Id": statement_id}
        if next_token is not None:
            kwargs["NextToken"] = next_token
        result = cast(
            Mapping[str, object],
            redshift_client.get_statement_result(**kwargs),
        )
        if first_page:
            column_metadata = result["ColumnMetadata"]
            first_page = False
        records.extend(cast(list[object], result["Records"]))
        next_token = cast(str | None, result.get("NextToken"))
        if next_token is None:
            return {"ColumnMetadata": column_metadata, "Records": records}


def _load_state(table: Any, pk: str) -> AlertState | None:
    response = table.get_item(Key={"PK": pk}, ConsistentRead=True)
    if "Item" not in response:
        return None
    item = cast(Mapping[str, object], response["Item"])
    return AlertState(
        pk=pk,
        last_alerted_bucket=datetime.fromisoformat(
            cast(str, item["last_alerted_bucket"])
        ),
        last_notified_level=int(cast(int | Decimal, item["last_notified_level"])),
    )


def _write_state(table: Any, state: AlertState) -> None:
    table.put_item(Item=_state_item(state))


def _state_item(state: AlertState) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "PK": state.pk,
        "last_alerted_bucket": state.last_alerted_bucket.isoformat(),
        "last_notified_level": state.last_notified_level,
        "updated_at": now.isoformat(),
        "ttl": int((now + timedelta(days=_STATE_TTL_DAYS)).timestamp()),
    }


def _notification_level(
    rule: cost_alerts.CostAlertRule,
    candidate: cost_alerts.AlertCandidate,
) -> int:
    ratio_over_threshold = (candidate.total_current_cost_usd / rule.threshold_usd) - 1
    if ratio_over_threshold <= 0:
        return 0
    return int(ratio_over_threshold // _ESCALATION_STEP)


def _incident_key(
    stage: str,
    rule: cost_alerts.CostAlertRule,
    scope_value: str | None,
) -> str:
    if scope_value is None:
        return f"COST_ALERT#{stage}#{rule.name}#{rule.grain}#{rule.group_by}"
    digest = _identity_digest(
        (stage, rule.name, rule.grain, rule.group_by, scope_value)
    )
    return f"COST_ALERT_V2#{digest}"


def _identity_digest(parts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(4, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _parameter_value(name: str) -> str:
    parameter = _ssm_client().get_parameter(Name=name)
    return cast(str, parameter["Parameter"]["Value"])


def _secret_value(name: str) -> str:
    secret = _secrets_client().get_secret_value(SecretId=name)
    value = secret.get("SecretString")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Secret {name} must contain SecretString")
    return value


def _ssm_client() -> Any:
    client_factory = cast(Any, boto3.client)
    return client_factory("ssm")


def _secrets_client() -> Any:
    client_factory = cast(Any, boto3.client)
    return client_factory("secretsmanager")


def _redshift_client() -> Any:
    client_factory = cast(Any, boto3.client)
    return client_factory("redshift-data")


def _state_table(table_name: str) -> Any:
    resource_factory = cast(Any, boto3.resource)
    dynamodb = resource_factory("dynamodb")
    return dynamodb.Table(table_name)


def _slack_sender(webhook_url: str) -> cost_alert_slack.SlackWebhookSender:
    return cost_alert_slack.SlackWebhookSender(webhook_url=webhook_url)
