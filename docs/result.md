# Query Results

`llm.query(...)` returns a `QueryResult`. Import result and metadata types from `model_library.base.output`.

## QueryResult

| Field | Description |
| --- | --- |
| `output_text` | Assistant text, when the provider returned text |
| `reasoning` | Provider reasoning/thinking text, when available |
| `finish_reason` | Normalized stop/length/tool-call/error reason plus raw provider value |
| `metadata` | `QueryResultMetadata`: token usage, cost, duration, and per-query performance telemetry |
| `tool_calls` | Normalized tool calls returned by the model |
| `history` | Provider-ready conversation history for follow-up calls |
| `extras.response_id` | Stable provider response/message/request ID, when available |

## QueryResultMetadata

`QueryResultMetadata` stores token counts, cost, request duration, provider-specific `extra` values, and `performance`.

`QueryResultMetadata` values can be summed for aggregate token, cost, and duration totals. Performance telemetry is per-query and is not aggregated when metadata values are summed.

## Performance metadata

`QueryResultMetadata.performance` stores per-query performance telemetry as a nested timeline. The supported serialized shape is the same shape returned by `result.metadata.model_dump()`.

```python
performance = result.metadata.performance
performance.time_to_first_token_ms.content  # first assistant text token/signal in ms
performance.time_to_first_token_ms.tool_call  # first tool-call token/signal in ms
performance.time_to_first_token_ms.answer  # min(content, tool_call)
performance.tokens_per_second.content  # null unless token attribution is defensible
```

```json
{
  "time_to_first_token_ms": {
    "any": 760,
    "answer": 760,
    "reasoning": null,
    "content": 760,
    "tool_call": 1580
  },
  "tokens_per_second": {
    "reasoning": 7.891,
    "content": 12.346,
    "tool_call": null
  },
  "timeline": [
    {
      "channel": "content",
      "index": 0,
      "start_ms": 700,
      "first_token_ms": 760,
      "ready_ms": null,
      "end_ms": 1300,
      "duration_ms": 600,
      "events": [
        {"type": "content_started", "timestamp_ms": 700},
        {"type": "content_delta", "timestamp_ms": 760},
        {"type": "content_finished", "timestamp_ms": 1300}
      ]
    }
  ]
}
```

Supported channels are `reasoning`, `content`, and `tool_call`. Supported event names are `reasoning_started`, `reasoning_delta`, `reasoning_finished`, `content_started`, `content_delta`, `content_finished`, `tool_call_started`, `tool_call_delta`, `tool_call_ready`, and `tool_call_finished`.

Timeline timestamps and derived segment durations are integer milliseconds from query start. A timeline can contain repeated segments for the same channel; `index` is zero-based and contiguous within each channel. `ready_ms` is only valid for `tool_call` segments.

`time_to_first_token_ms` is derived from timeline segment `first_token_ms` values. `any` is the first generated reasoning/content/tool-call token or signal, and `answer` is the first non-reasoning output: `min(content, tool_call)`. `tokens_per_second` stores aggregate channel speeds only when token attribution is defensible.

The performance schema is nested under `performance`; flat timing/rate fields such as `time_to_first_token_seconds`, `time_to_first_content_token_seconds`, `output_tokens_per_second`, and `reasoning_tokens_per_second` are not part of the public schema. `answer_started` is not a performance event; use `content_*` or `tool_call_*` events.

Timing and rate fields ending in `_seconds` or `_per_second` are rounded to three decimals. Cost precision is handled separately by `QueryResultCost`.
