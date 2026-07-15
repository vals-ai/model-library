# Query Results

`llm.query(...)` returns a `QueryResult`. Import result and metadata types from `model_library.base.output`.

## QueryResult

| Field | Description |
| --- | --- |
| `output_text` | Normalized textual result when the provider returned non-empty text or text-like output, such as OpenAI Code Mode output. Empty provider text is normalized to `None`; `None` means no useful text was observed, such as an empty-text-only or tool-only response. |
| `reasoning` | Provider reasoning/thinking text, when available; for OpenAI Responses reasoning models this is the provider's reasoning summary, not raw hidden reasoning |
| `finish_reason` | Normalized stop/length/tool-call/error reason plus raw provider value |
| `metadata` | `QueryResultMetadata`: token usage, cost, duration, and per-query performance telemetry |
| `tool_calls` | Normalized tool calls returned by the model for local/tool-call execution |
| `history` | Provider-ready conversation history for follow-up calls |
| `extras.response_id` | Legacy compatibility field for provider response/body/message IDs, when available. Prefer `provider_response_id` / `provider_request_id`. |
| `extras.provider_response_id` | Canonical provider response/body/message ID, when available. |
| `extras.provider_request_id` | Provider request/support correlation ID, when available. |

For compatibility, if only one of `extras.response_id` or `extras.provider_response_id` is present, the missing field is hydrated with the same value. If both are present, distinct values are preserved.

## QueryResultMetadata

`QueryResultMetadata` stores token counts, cost, request duration, provider-specific `extra` values, and `performance`.

`QueryResultMetadata` values can be summed for aggregate token, cost, and duration totals. Per-query `extra` values and performance telemetry are not aggregated when metadata values are summed.

## Performance metadata

### Availability and access

`QueryResultMetadata.performance` contains a per-query event timeline when the
provider exposes timing events. It is `None` for non-streaming results without a
timeline and for aggregate metadata.

```python
performance = result.metadata.performance
if performance is not None:
    performance.time_to_first_token_ms.content
    performance.time_to_first_token_ms.tool_call
    performance.time_to_first_token_ms.answer
```

### Shape

| Field | Meaning |
| --- | --- |
| `time_to_first_token_ms` | First observed token/delta timing by output category |
| `timeline` | Ordered timing segments for `reasoning`, `content`, and `tool_call` channels |
| Segment `index` | Zero-based, contiguous index within one channel; repeated channel segments are allowed |
| Segment `start_ms` | First observed start event from query start |
| Segment `first_token_ms` | First observed token/delta event from query start |
| Segment `ready_ms` | Provider readiness signal; valid only for `tool_call` segments |
| Segment `end_ms` | Final observed channel event from query start |
| Segment `duration_ms` | Derived segment duration in integer milliseconds |
| Segment `events` | Raw normalized timing events for that segment |

```json
{
  "time_to_first_token_ms": {
    "any": 760,
    "answer": 760,
    "reasoning": null,
    "content": 760,
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
        {
          "type": "content_delta",
          "timestamp_ms": 760,
          "channel_text_start_char": 0,
          "channel_text_end_char": 42
        },
        {"type": "content_finished", "timestamp_ms": 1300}
      ]
    }
  ]
}
```

### Channels and events

| Channel | Events |
| --- | --- |
| `reasoning` | `reasoning_started`, `reasoning_delta`, `reasoning_finished` |
| `content` | `content_started`, `content_delta`, `content_finished` |
| `tool_call` | `tool_call_started`, `tool_call_delta`, `tool_call_ready`, `tool_call_finished` |

`answer_started` is not a performance event; use `content_*` or
`tool_call_*` events.

Text delta events can include `channel_text_start_char` and
`channel_text_end_char`, an inclusive/exclusive range into the final channel
text:

- `content_delta` ranges index `QueryResult.output_text`.
- `reasoning_delta` ranges index `QueryResult.reasoning`.
- Non-text events omit these fields.
- Providers that rewrite final text during postprocessing omit offsets when the
  original stream chunks no longer map to the final text.

### Timing derivation

`time_to_first_token_ms` is derived from timeline segment `first_token_ms`
values:

| Field | Derivation |
| --- | --- |
| `any` | First generated reasoning, content, or tool-call token delta |
| `answer` | First non-reasoning output: `min(content, tool_call)` |
| `reasoning` | First reasoning token delta |
| `content` | First content token delta |
| `tool_call` | First tool-call argument/token delta |

A tool call's `ready_ms` records the provider signal that the call is available,
but does not count as first-token latency unless argument/token deltas were
observed.

### Interpretation and caveats

- Timeline values are best-effort provider event timings, not a complete
  wall-clock request breakdown.
- They measure provider-emitted reasoning, content, and tool-call segments from
  query start when those events are available.
- They exclude request setup, network latency, provider queueing, prompt
  ingestion/prefill, unstreamed hidden reasoning, and final response assembly.
- `QueryResultMetadata.duration_seconds` remains the request-level duration.
- Timings use raw provider stream events. If postprocessing later moves reasoning
  tags out of content, `content` and `answer` can still describe the original raw
  content delta rather than the first token of final `output_text`.

### Aggregation and precision

`performance` does not include per-query tokens per second and is discarded when
`QueryResultMetadata` values are summed.

- For agentic runs, calculate aggregate TPS as
  `sum(result.metadata.total_output_tokens) / agent_run_wall_clock_seconds`.
- For sequential-only aggregation without a run timer, use
  `sum(total_output_tokens) / sum(duration_seconds)` as a fallback.
- Timing fields ending in `_seconds` are rounded to three decimals.
- `QueryResultCost` handles cost precision separately.
