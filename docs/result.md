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

`QueryResultMetadata` stores token counts, cost, request duration, provider-specific `extra` values, `performance`, and `fallback`.

`fallback` is `None` unless a provider-side fallback model served the response. It is a `FallbackInfo` with `requested_model`, `served_model`, and `hops` (one `FallbackHop` per model attempt: `model`, `served`, `trigger`/`category` for hops that declined, and `usage` as a token-only `QueryResultMetadata`). Top-level token counts and cost are the served hop's, priced at `served_model`. If cost calculation fails (for example `served_model` is not in the registry), the query still succeeds with `cost=None` and a warning is logged.

`QueryResultMetadata` values can be summed for aggregate token, cost, and duration totals. Per-query `extra` values, `fallback`, and performance telemetry are not aggregated when metadata values are summed.

## Performance metadata

### Availability and access

`QueryResultBuilder` preserves a nonempty `performance` value already supplied in metadata
when it records no new timeline. Typical non-streaming or set-only results without supplied
performance remain `None`; summing metadata values discards `performance`. Persisted ledger
details can also omit performance when their oversized-details fallback is used.

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
| `time_to_first_token_ms` | Historical field name for first observed non-empty delta/chunk timing by output category |
| `timeline` | Ordered, adapter-normalized segments for `reasoning`, `content`, and `tool_call` channels |
| Segment `index` | Zero-based, contiguous index within one channel; repeated channel segments are allowed |
| Segment `start_ms` | First normalized start event relative to `QueryResultBuilder` construction |
| Segment `first_token_ms` | Historical field name for the first observed non-empty delta/chunk in the segment |
| Segment `ready_ms` | Provider readiness signal; valid only for `tool_call` segments |
| Segment `end_ms` | Final normalized channel event relative to `QueryResultBuilder` construction |
| Segment `duration_ms` | Derived segment duration in integer milliseconds |
| Segment `events` | Canonical normalized adapter events for that segment |

The typed Python model and the decompressed performance payload have this shape:

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
      "start_ms": 760,
      "first_token_ms": 760,
      "ready_ms": null,
      "end_ms": 1300,
      "duration_ms": 540,
      "events": [
        {"type": "content_started", "timestamp_ms": 760},
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

Content and reasoning segments open when their first non-empty delta is observed,
so their synthetic `*_started` event normally has the same timestamp as
`first_token_ms`. A tool call can start or become ready before its first argument
delta. Adapters can finish segments on provider events or channel transitions;
`QueryResultBuilder.build()` closes any segment still open.

### JSON serialization

`QueryResultMetadata.model_dump(mode="json")` and Model Gateway responses encode a
non-null `performance` value as a lossless gzip/base64 envelope:

```json
{
  "performance": {
    "encoding": "gzip+base64",
    "data": "H4sI..."
  }
}
```

The envelope uses gzip compression level 1. Its decompressed bytes are the
compact JSON representation shown above, including every event, timestamp,
offset, and ordering. This is an application-level field encoding, not HTTP
`Content-Encoding`.

Model Library validation accepts both representations without implicitly
changing either one. Historical uncompressed objects validate as
`QueryResultPerformance`; compressed JSON validates as
`CompressedQueryResultPerformance`. Gateway responses, clients, usage-ledger
storage, and the Redshift performance table retain the compressed envelope when
it is available. Adding `QueryResultMetadata` values discards `performance`
because separate query timelines cannot be merged into one coherent clock.

A consumer must explicitly decompress the envelope when it needs the timeline:

```python
from model_library.base.output import (
    CompressedQueryResultPerformance,
    decompress_query_result_performance,
)

performance = result.metadata.performance
if isinstance(performance, CompressedQueryResultPerformance):
    performance = decompress_query_result_performance(performance)
```

The usage ledger preserves the envelope past its generic 64,000-character field
bound. The complete event remains subject to the ledger's 300,000-byte message
budget, DynamoDB's 400 KiB hard item limit, and the existing oversized-details
fallback.

### Channels and events

The `reasoning` channel contains provider-visible reasoning-like text normalized
by the adapter. Its meaning is provider-specific. For example, OpenAI Responses
records the provider's reasoning summary, not raw hidden reasoning. A reasoning
timeline and separately reported `reasoning_tokens` do not imply each other.

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
| `any` | First observed non-empty reasoning, content, or tool-call delta |
| `answer` | First non-reasoning delta: `min(content, tool_call)` |
| `reasoning` | First observed reasoning delta |
| `content` | First observed content delta |
| `tool_call` | First observed tool-call argument delta |

A tool call's `ready_ms` records the normalized provider signal that the call is
available. It does not populate `first_token_ms` unless an argument delta was
observed.

### Observation boundaries

- Timeline timestamps are local monotonic elapsed times from
  `QueryResultBuilder` construction in the adapter process.
- Work completed before builder construction is outside the timeline. Time before
  the first observed delta can include network transit, provider queueing,
  prompt ingestion/prefill, hidden processing, and buffering; the timeline cannot
  separate those components.
- `QueryResultMetadata.duration_seconds` measures the successful `_query_impl`
  attempt. Its clock starts before builder construction and can end after the
  final timeline event. It excludes earlier failed attempts and retry waits, and
  is not caller, agent, or run wall-clock duration.
- Events are canonical adapter-normalized events, not provider timestamps.
  Provider and SDK buffering remains visible only as local arrival behavior.
- If postprocessing rewrites streamed text, text offsets are omitted when the
  observed chunks no longer map to final `output_text` or `reasoning`.

### Metrics and aggregation

`performance` contains no precomputed throughput metric and is discarded when
`QueryResultMetadata` values are summed. Token accounting uses:

```text
total_output_tokens = out_tokens + reasoning_tokens
```

Use a name and denominator that match the intended measurement. `selected_output_tokens` is the
caller-selected field: use `out_tokens` for non-reasoning output or `total_output_tokens` for
total-generated output, and match the metric name to that selection:


| Metric | Formula | Boundary |
| --- | --- | --- |
| Query-attempt non-reasoning output rate | `out_tokens / duration_seconds` | One successful `_query_impl` attempt |
| Query-attempt total-generated rate | `total_output_tokens / duration_seconds` | One successful `_query_impl` attempt, including reasoning tokens |
| System/workload output throughput | `sum(selected_output_tokens) / (last_response_time - first_request_time)` | One explicitly defined workload window |
| Sequential-only fallback | `sum(selected_output_tokens) / sum(duration_seconds)` | Non-overlapping attempts when no shared wall-clock timer exists |

The sequential fallback is not concurrent system throughput. Timeline deltas are
arbitrary chunks rather than token-aligned events, so their counts and arrival
gaps cannot produce token-level inter-token latency (ITL), time per output token
(TPOT), or model decode TPS.

Timing fields ending in `_seconds` are rounded to three decimals.
`QueryResultCost` handles cost precision separately.
