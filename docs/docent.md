# Docent Integration

Automatic ingestion of LLM transcripts into [Docent](https://docent.transluce.org) for analysis.

## Setup

```bash
uv pip install model-library[docent]
```

Set `DOCENT_API_KEY` in your environment.

## Usage

### Single-turn (LLM.query)

```python
result = await llm.query(
    input_text,
    run_id="my-collection-id",
    question_id="q1",
    docent_ingest=True,
)
```

Creates one Docent AgentRun per question with the input and assistant response.

### Multi-turn (Agent.run)

```python
result = await agent.run(
    [TextInput(text="Find the weather in Tokyo")],
    run_id="my-collection-id",
    question_id="q1",
    docent_ingest=True,
)
```

Creates one Docent AgentRun containing the full conversation: user input, assistant responses, tool calls, tool results, and hook-injected messages.

## What Gets Captured

| Source | Docent Message Type |
|--------|-------------------|
| `SystemInput` | SystemMessage |
| `TextInput` | UserMessage |
| `FileInput` | UserMessage (placeholder: `[image: name (mime)]`) |
| `ToolResult` | ToolMessage |
| `QueryResult` output | AssistantMessage (with reasoning, tool calls) |
| `ErrorTurn` | AssistantMessage (`[error: message]`) |
| `RawResponse` / `RawInput` | Skipped (provider-specific, not parseable) |

### Hook-Injected Messages

`turn_message` and `time_message` hooks append `InputItem`s to history before each query. These are captured in the Docent transcript by extracting them from each turn's `query_result.history`.

```python
config = AgentConfig(
    turn_limit=TurnLimit(
        max_turns=10,
        turn_message=lambda turn, max: TextInput(text=f"[Turn {turn}/{max}]"),
    ),
    time_limit=TimeLimit(
        max_seconds=120,
        time_message=lambda elapsed, max: TextInput(text=f"[{max - elapsed:.0f}s left]"),
    ),
)
```

These appear as UserMessages in the transcript between tool results and assistant responses.

### Known Limitation

History truncation via `before_query` (e.g. `truncate_oldest`) is not reflected in the Docent transcript. Truncated turns still appear even though the model no longer saw them.

## Deduplication

Re-running the same `question_id` within a `run_id` replaces the previous AgentRun. This uses a DQL query on `metadata_json->>'question_id'` to find and delete the existing run before ingesting.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `docent_ingest` | Yes | Set to `True` to enable ingestion |
| `run_id` | Yes | Docent collection ID. All agent runs with the same `run_id` are grouped together. |
| `question_id` | Yes | Unique ID per question within a run. Used for deduplication. |

## Error Handling

Docent failures never block the LLM query or agent run. All ingestion code is wrapped in `try/except` and logs warnings on failure.

## Source Files

| File | Role |
|------|------|
| `docent.py` | Converters (`InputItem` → `ChatMessage`, `QueryResult` → `AssistantMessage`) and ingestion |
| `base/base.py` | `docent_ingest` flag on `LLM.query` |
| `agent/agent.py` | `docent_ingest` flag on `Agent.run` |
| `query_utils.py` | `docent_ingest` flag on `query_with_truncation_retry` (forwarded to inner `LLM.query`) |

## Example

See `examples/agent/basic.py` — `agent_with_hooks_and_docent` for a full example with turn/time hooks and Docent ingestion.
