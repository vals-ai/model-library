# Cost Alerts

Scheduled gateway cost alerts query Redshift and post threshold notifications to
a Slack incoming webhook.

## Configuration

Set these environment variables on the alert process:

| Variable | Purpose |
| --- | --- |
| `GATEWAY_COST_ALERT_CONFIG_PARAMETER_NAME` | SSM parameter containing the JSON rule configuration |
| `GATEWAY_COST_ALERT_STATE_TABLE_NAME` | DynamoDB state used for per-bucket alert deduplication and worsening levels |
| `GATEWAY_COST_ALERT_SLACK_WEBHOOK_SECRET_NAME` | Secrets Manager secret containing the Slack webhook URL as a plain `SecretString` |
| `GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME` | Redshift Serverless workgroup queried through the Data API |
| `GATEWAY_USAGE_REDSHIFT_DATABASE_NAME` | Shared Redshift database name |
| `GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME` | Redshift schema containing the usage aggregate tables |
| `GATEWAY_STAGE` | Stage label included in Slack messages and state keys |

The alert process sends active and worsening messages through a Slack incoming
webhook. Scoped rules keep independent deduplication state per breached
dimension. Messages include Pacific-time windows and populated breakdowns.
Builds that configure a dashboard URL also include a dashboard button. Optional
rule-level images can attach to selected message types. Delivery does not
require a bot token and does not block or reroute model requests.

Example SSM config with sustained and daily horizons:

```json
{
  "rules": [
    {
      "name": "global-1h",
      "grain": "1h",
      "group_by": "global",
      "threshold_usd": "25.00",
      "include_missing": true,
      "image": {
        "url": "https://example.com/severe-hourly.png",
        "alt_text": "severe hourly spend alert",
        "attach_to": ["worsening"],
        "min_threshold_multiple": "2"
      }
    },
    {
      "name": "global-1d",
      "grain": "1d",
      "group_by": "global",
      "threshold_usd": "250.00",
      "include_missing": true
    }
  ]
}
```

## Rule Contract

The root JSON object must contain a non-empty `rules` array. Each entry supports
this contract:

| Field | Required | Values and behavior |
| --- | --- | --- |
| `name` | Yes | Non-empty string. |
| `grain` | Yes | `1h` or `1d`. |
| `group_by` | Yes | `global`, `provider_model`, `benchmark`, or `identity_email`. |
| `threshold_usd` | Yes | Finite numeric value greater than zero. |
| `min_previous_usd` | No | Finite numeric value at least zero. Required when `percent_increase` is set. |
| `percent_increase` | No | Finite numeric value greater than zero. |
| `include_missing` | No | Boolean; defaults to `false`. |
| `image` | No | Object described below; omitted by default. |

Unknown keys at the config root, rule, and image levels are intentionally ignored so harmless forward-compatible fields do not disable alerts. Every recognized field is still validated. Duplicate `(name, grain, group_by)` identities are rejected because they would share deduplication state.

## Slack Messages

The Lambda uses an attachment color rail and Block Kit `blocks`:

| State | Example title | Behavior |
| --- | --- | --- |
| Active | `⏱️ Hourly | Triggered | $36.42 / $25.00 limit` | Yellow alert with scope, over-limit summary, counts, approximate cost/request, Pacific-time window, an optional dashboard button, and up to three contributors per breakdown |
| Worsening | `📅 Daily | Worsening: 2.25x | $56.25 / $25.00 limit` | Orange, red, or purple as threshold multiples increase |

Scope is one of `Global threshold`, `Model: ...`, `Benchmark: ...`, or
`Email: ...`. Model, benchmark, and email breakdowns are filtered to that exact
scope. Empty breakdown sections are omitted.

Optional rule-level images:

- Require an `http://` or `https://` URL no longer than 3,000 characters.
- Require configured `alt_text` to be non-empty and no longer than 2,000
  characters; omitted `alt_text` defaults to `cost alert image`.
- Default omitted `attach_to` to `["worsening"]` and require at least one of
  `active` or `worsening` when configured.
- Require `min_threshold_multiple` to be finite and greater than zero when
  configured.

Rendered Slack sections longer than 3,000 characters are truncated with an
ellipsis.

## Redshift Permissions

- Use a dedicated read-only database role for the configured schema.
- Grant the role schema `USAGE` and `SELECT` on `usage_agg_1h`, then grant role
  membership to the alert process's IAM database user.
- Revoke obsolete grants and retired IAM database-user memberships explicitly;
  replacing deployment configuration does not remove them automatically.

## Rule Behavior

- Primary Redshift comparison statements run eight at a time, then results are
  applied in configuration order. Breakdown queries, state changes, and Slack
  delivery remain sequential.
- The handler polls submitted Data API statements until they finish and surfaces
  Redshift `FAILED` or `ABORTED` outcomes. It does not defer work or cancel
  statements.
- Each rule uses `max(data_through_utc)` from its aggregate table as the
  evaluation watermark.
- Alert scheduling is independent of usage-export scheduling.
- `1h` detects sustained spend.
- `1d` follows the current Pacific-time calendar day, including 23- or 25-hour
  daylight-saving transition days, and aggregates hourly source fragments into
  one daily row per dimension.
- The first eligible breach for a rule and scope in each bucket sends an active
  alert. Another message is sent only after spend crosses a new +25% worsening
  level.
- Below-threshold or absent scopes perform no state work. A breach in a later
  bucket sends a new active alert.
- State is retained for 30 days and written directly after Slack succeeds.
- There is no resolved or healthy-count lifecycle, GSI/open-scope index, or
  state-table scan.

## Delivery and Limitations

- Delivery uses a Slack incoming webhook, not a bot token or channel ID.
- A failure after Slack accepts a message but before state is written can produce
  a duplicate notification on a later invocation.
- There are no custom username/icon overrides, threads, or shared cross-horizon
  incidents.
- Alerts do not block, throttle, or reroute model requests.

Before deployment, validate the configured rules for duplicate identities and
obsolete `image.attach_to: "resolved"` values.
