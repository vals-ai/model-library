# Agent

Tool-augmented conversation loop that queries an LLM and executes tools until a stopping condition is met.

## Quick Start

```python
agent = Agent(
    llm=llm,
    tools=[MyTool()],
    name="finance_agent",
    config=AgentConfig(turn_limit=TurnLimit(max_turns=50), time_limit=None),
)
result = await agent.run(input, question_id="question_1")

result.final_answer        # str
result.success             # True if no error
result.total_turns         # number of turns (including errors)
result.tool_usage          # {"tool_name": count}
result.output_dir          # Path to per-question output directory
```

## AgentResult

Returns lean summaries in memory — full raw data is written to disk per-turn.

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | From `determine_answer` hook, done tool, or LLM text |
| `final_error` | `SerializableException \| None` | Set on max turns/time, unhandled exceptions |
| `final_history` | `list[InputItem]` | Conversation history after the run (excluded from JSON/repr). Use this to continue a conversation across multiple `run()` calls. |
| `turns` | `list[TurnSummary \| ErrorTurn]` | Per-turn summaries |
| `output_dir` | `Path` | Question-level output directory (excluded from JSON) |
| `success` | `bool` | `final_error is None` |
| `total_turns` | `int` | `len(turns)` |
| `error_count` | `int` | ErrorTurns + failed tool calls |
| `tool_calls_count` | `int` | Total tool calls across all turns |
| `tool_usage` | `dict[str, int]` | Tool call counts by name |
| `final_duration_seconds` | `float` | Wall clock for entire run |
| `final_retry_overhead_seconds` | `float` | Sum of retry/backoff time |
| `final_effective_duration_seconds` | `float` | Wall clock minus retry overhead |
| `final_aggregated_metadata` | `QueryResultMetadata` | Sum of query `metadata` across all turns (tool sub-LLM metadata available per `ToolCallSummary.metadata`) |

State is not returned — the caller owns it by reference:

```python
state = {"counter": 0}
result = await agent.run(input, question_id="q1", state=state)
print(state["counter"])  # mutated by tools during execution
```

### TurnSummary

Lean per-turn metadata (content replaced with lengths):

| Field | Type |
|-------|------|
| `output_text_length` | `int` |
| `reasoning_length` | `int` |
| `finish_reason` | `FinishReasonInfo` |
| `metadata` | `QueryResultMetadata` |
| `tool_calls` | `list[ToolCallSummary]` |
| `duration_seconds` | `float` |
| `retry_overhead_seconds` | `float` |
| `effective_duration_seconds` | `float` (computed) |

### ToolCallSummary

| Field | Type |
|-------|------|
| `tool_name` | `str` |
| `tool_call_id` | `str` |
| `args_lengths` | `dict[str, int]` |
| `output_length` | `int` |
| `success` | `bool` |
| `done` | `bool` |
| `error` | `SerializableException \| None` |
| `duration_seconds` | `float` |
| `metadata` | `QueryResultMetadata \| None` |

## Hooks

All hooks receive raw types (`AgentTurn`, `ToolCallRecord`), not summaries. All hook invocations are DEBUG-logged.

| Hook | Signature | Default |
|------|-----------|---------|
| `before_query` | `(history, last_error) → history` | Re-raises errors |
| `should_stop` | `(TurnResult) → bool` | Stop on text-only response |
| `on_tool_result` | `(ToolCallRecord, state) → None` | No-op |
| `determine_answer` | `(state, list[AgentTurn \| ErrorTurn], error) → str` | Done tool output → LLM text → `""` |

Optional per-turn message hooks on limits:

| Hook | On | Signature |
|------|-------|-----------|
| `TurnLimit.turn_message` | Each turn | `(turn_number, max_turns) → InputItem \| None` |
| `TimeLimit.time_message` | Each turn | `(elapsed_seconds, max_seconds) → InputItem \| None` |

## Stopping Conditions

The loop stops when any of these occur:

- A tool returns `done=True`
- `should_stop` hook returns `True` (default: text-only response)
- `turn_limit.max_turns` reached → `MaxTurnsExceeded` error
- `time_limit.max_seconds` exceeded → `MaxTimeExceeded` error
- `before_query` hook re-raises a query error (default behavior)
- Unhandled exception

After the loop, `determine_answer` runs with full raw turns.

## Configuration

| Field | Default | Purpose |
|-------|---------|---------|
| `turn_limit` | required | `TurnLimit(max_turns=N)` or `None` for unlimited |
| `time_limit` | required | `TimeLimit(max_seconds=N)` or `None` for unlimited |
| `max_tool_calls_per_turn` | `None` | Cap on executed tool calls per turn; excess get skip messages |

Time budget uses wall clock minus retry overhead, so retry/backoff time doesn't count against the budget. Set `TimeLimit(include_retries=True)` for strict wall clock.

## Logging & Output

- `name` — required, used for logger name and log directory
- `log_dir` — base directory for file output (default `logs/`)
- `question_id` — required, creates a subdirectory per question
- `logger` — optional custom logger; the agent appends `<name><<model_name>>` as a child

The agent builds a timestamped run directory: `log_dir/<name>/<model_name>/<timestamp>_<uuid>/<question_id>/`

### Directory Structure

```
logs/finance_agent/gpt-4o/2024-01-01_12-00-00_abc123/
├── question_1/
│   ├── agent.log              # full text log (turns, hooks, errors)
│   ├── result.json            # AgentResult with TurnSummary list
│   └── turns/
│       ├── init/
│       │   ├── config.json    # tool definitions, LLM config
│       │   ├── state.json     # initial state
│       │   └── history.bin    # serialized initial input
│       ├── turn_001/
│       │   ├── result.json    # raw AgentTurn (full query result + tool records)
│       │   ├── state.json     # state snapshot after turn
│       │   └── history.bin    # serialized LLM history after turn
│       └── turn_002/
│           └── error.json     # ErrorTurn (failed LLM query)
└── question_2/
    ├── agent.log
    ├── result.json
    └── turns/
```

### Custom Logger

The agent overrides the LLM's logger so all LLM logs become children of the agent logger.

```python
my_logger = logging.getLogger("myapp.cloudwatch")
agent = Agent(name="finance_agent", llm=llm, tools=tools, logger=my_logger)
# logs at: myapp.cloudwatch.finance_agent<gpt-4o>
```

If the logger (or a parent, excluding root) already has a FileHandler, the agent reuses it instead of creating a new one. Output files are still scoped under `<question_id>/`.

## Source Files

| File | Role |
|------|------|
| `agent/agent.py` | `Agent` class and `AgentResult` |
| `agent/config.py` | `AgentConfig`, `TurnLimit`, `TimeLimit`, `truncate_oldest` |
| `agent/hooks.py` | `AgentHooks` dataclass and hook protocols |
| `agent/metadata.py` | `AgentTurn`, `ErrorTurn`, `ToolCallRecord`, `TurnSummary`, `ToolCallSummary` |
| `agent/tool.py` | `Tool` base class, `ToolOutput` |
| `agent/conductor/` | `ConductorAgent` — orchestrates multi-turn conversations between agents |
| `utils.py` | `create_file_logger`, `run_logging` |
