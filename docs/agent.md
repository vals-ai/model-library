# Agent

Tool-augmented conversation loop that queries an LLM and executes tools until a stopping condition is met.

## Quick Start

```python
agent = Agent(
    llm=llm,
    tools=[MyTool()],
    name="finance_agent",
    config=AgentConfig(
        turn_limit=TurnLimit(max_turns=50),
        time_limit=None,
        history_compaction=HistoryCompaction(threshold_tokens=120_000),
    ),
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

| Field                              | Type                             | Description                                                                                                                      |
| ---------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `final_answer`                     | `str`                            | From `determine_answer` hook, done tool, or LLM text                                                                             |
| `final_error`                      | `SerializableException \| None`  | Set on max turns/time, unhandled exceptions                                                                                      |
| `final_history`                    | `list[InputItem]`                | Conversation history after the run (excluded from JSON/repr). Use this to continue a conversation across multiple `run()` calls. |
| `turns`                            | `list[TurnSummary \| ErrorTurn]` | Per-turn summaries                                                                                                               |
| `compactions`                      | `list[CompactionSummary]`        | History compaction attempts with metadata/cost                                                                                   |
| `output_dir`                       | `Path`                           | Question-level output directory (excluded from JSON)                                                                             |
| `success`                          | `bool`                           | `final_error is None`                                                                                                            |
| `total_turns`                      | `int`                            | `len(turns)`                                                                                                                     |
| `error_count`                      | `int`                            | ErrorTurns + failed tool calls                                                                                                   |
| `tool_calls_count`                 | `int`                            | Total tool calls across all turns                                                                                                |
| `tool_usage`                       | `dict[str, int]`                 | Tool call counts by name                                                                                                         |
| `final_duration_seconds`           | `float`                          | Wall clock for entire run                                                                                                        |
| `final_retry_overhead_seconds`     | `float`                          | Sum of retry/backoff time                                                                                                        |
| `final_effective_duration_seconds` | `float`                          | Wall clock minus retry overhead                                                                                                  |
| `final_aggregated_metadata`        | `QueryResultMetadata`            | Sum of query `metadata` across agent turns only — task cost (tool sub-LLM metadata available per `ToolCallSummary.metadata`)     |
| `final_compaction_metadata`        | `QueryResultMetadata`            | Sum of query `metadata` across history compaction calls — overhead. Sum with `final_aggregated_metadata` for the true bill.      |

Query-level metadata, including `QueryResultMetadata.performance`, is documented in [Query Results](result.md).

State is not returned — the caller owns it by reference:

```python
state = {"counter": 0}
result = await agent.run(input, question_id="q1", state=state)
print(state["counter"])  # mutated by tools during execution
```

### TurnSummary

Lean per-turn metadata (content replaced with lengths).

| Field                        | Type                       |
| ---------------------------- | -------------------------- |
| `output_text_length`         | `int`                      |
| `reasoning_length`           | `int`                      |
| `finish_reason`              | `FinishReasonInfo`         |
| `metadata`                   | `QueryResultMetadata`      |
| `tool_calls`                 | `list[ToolCallSummary]`    |
| `duration_seconds`           | `float`                    |
| `retry_overhead_seconds`     | `float`                    |
| `effective_duration_seconds` | `float` (computed)         |

### ToolCallSummary

| Field              | Type                            |
| ------------------ | ------------------------------- |
| `tool_name`        | `str`                           |
| `tool_call_id`     | `str`                           |
| `args_lengths`     | `dict[str, int]`                |
| `output_length`    | `int`                           |
| `success`          | `bool`                          |
| `done`             | `bool`                          |
| `error`            | `SerializableException \| None` |
| `duration_seconds` | `float`                         |
| `metadata`         | `QueryResultMetadata \| None`   |

### CompactionSummary

One record per compaction attempt. Successful compactions write the previous
history as JSON text to `compactions/compaction_NNN/previous_history.bin`
(relative to `AgentResult.output_dir`; the `.bin` suffix is a legacy name);
failed compactions have `artifacts_subdir=None`.
Compaction itself does not increment the agent loop turn counter — `turn_number`
is the turn during which the compaction ran (i.e. the turn whose LLM query
consumed the compacted history).

| Field                  | Type                            | Description                                                           |
| ---------------------- | ------------------------------- | --------------------------------------------------------------------- |
| `turn_number`          | `int \| None`                   | Turn during which compaction fired                                    |
| `input_token_estimate` | `int \| None`                   | Estimated input-token size of the next turn that triggered compaction |
| `threshold_tokens`     | `int \| None`                   | Threshold active when this attempt fired                              |
| `summary`              | `str \| None`                   | Compaction summary returned by the LLM                                |
| `artifacts_subdir`     | `str \| None`                   | Subpath under `output_dir` for JSON-text `previous_history.bin`       |
| `metadata`             | `QueryResultMetadata \| None`   | Token / cost / duration of the compaction LLM call                    |
| `error`                | `SerializableException \| None` | Error if the attempt failed                                           |
| `success`              | `bool`                          | `error is None`                                                       |

## Hooks

All hooks receive raw types (`AgentTurn`, `ToolCallRecord`), not summaries. All hook invocations are DEBUG-logged.

| Hook               | Signature                                            | Default                                                                   |
| ------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `before_query`     | `(history, last_error) → history`                    | Re-raises errors                                                          |
| `should_stop`      | `(TurnResult) → bool`                                | Continue only for local tool calls or a paused provider turn; otherwise stop |
| `on_tool_result`   | `(ToolCallRecord, state) → None`                     | No-op                                                                     |
| `determine_answer` | `(state, list[AgentTurn \| ErrorTurn], error) → str` | Done tool output → LLM text → `""`                                        |

Optional per-turn message hooks on limits:

| Hook                     | On        | Signature                                            |
| ------------------------ | --------- | ---------------------------------------------------- |
| `TurnLimit.turn_message` | Each turn | `(turn_number, max_turns) → InputItem \| None`       |
| `TimeLimit.time_message` | Each turn | `(elapsed_seconds, max_seconds) → InputItem \| None` |

## Stopping Conditions

The loop stops when any of these occur:

- A tool returns `done=True`
- `should_stop` hook returns `True`. By default, the loop continues only when it must execute local tool calls or resume a provider turn with `finish_reason == PAUSED`; every other response is terminal.
- `turn_limit.max_turns` reached → `MaxTurnsExceeded` error
- `time_limit.max_seconds` exceeded → `MaxTimeExceeded` error
- `before_query` hook re-raises a query error (default behavior)
- Unhandled exception

After the loop, `determine_answer` runs with full raw turns.

## Configuration

| Field                     | Default  | Purpose                                                       |
| ------------------------- | -------- | ------------------------------------------------------------- |
| `turn_limit`              | required | `TurnLimit(max_turns=N)` or `None` for unlimited              |
| `time_limit`              | required | `TimeLimit(max_seconds=N)` or `None` for unlimited            |
| `max_tool_calls_per_turn` | `None`   | Cap on executed tool calls per turn; excess get skip messages |
| `history_compaction`      | `None`   | Optional `HistoryCompaction(...)` config for long histories   |

Time budget uses wall clock minus retry overhead, so retry/backoff time doesn't count against the budget. Set `TimeLimit(include_retries=True)` for strict wall clock.

### History Compaction

Before each turn the agent calls `hooks.compaction` with the prior turn's
LLM metadata and tool-call records as signals. The hook decides whether
to actually compact. The default hook (`llm_summary_compactor`)
uses the prior turn's provider-reported tokens plus tokenizer-counted
tool outputs as its input-token estimate — when that crosses
`threshold_tokens` it sends the current history to the same LLM with the
compaction prompt appended as a trailing `TextInput` and `tools=[]`, then
replaces the history with the returned summary. Custom hooks are free to
use a different policy entirely (e.g.
tokenizer-accurate sizing, "compact every N turns", semantic-boundary
detection); see the _Custom compaction strategies_ section below.

For the default hook, the post-compaction history is
`[SystemInputs, TextInput(summary_prefix + summary)]`:

- `SystemInput` items kept in original order.
- Everything else is replaced by the summary. The compaction prompt
  explicitly asks the model to capture the user's task and intent, so
  there's no need to keep the original `TextInput`s around.
- Successive compactions therefore naturally replace prior summaries
  (no need for a prefix-based filter).

If the compaction call itself is too big for the model
(`MaxContextWindowExceededError`), the hook uses `truncate_oldest` to drop
the oldest exchange and retries up to `max_compaction_context_retries`. This
same retry limit applies whether compaction was triggered by the threshold gate
or by `compact_on_max_context`.

The estimate can lag a sudden tool-output spike, so the threshold check
won't always fire in time. As an opt-in safety net, set
`compact_on_max_context=True`. When enabled and a turn's main LLM query
raises `MaxContextWindowExceededError`, the agent compacts once and retries
the query once. If compaction does not succeed, the error becomes an
`ErrorTurn` exposed to the next turn's `before_query` hook as
`last_query_error`.

> **Don't combine `before_query=truncate_oldest` with `compact_on_max_context=True`.**
> They address the same failure mode at different phases —
> compact-on-max-context runs intra-turn; `before_query` runs at the top of
> the next turn after an ErrorTurn. With both enabled a single overflow
> can trigger compact-then-retry, then on the next turn `truncate_oldest`,
> then a threshold-driven compaction, all before another LLM query. Pick
> one. Compaction preserves more semantic info; pure `truncate_oldest` is
> cheaper but cruder.

On failures the agent logs the error, records a `CompactionSummary`
with the error and no artifact directory, and continues with the original
history; after `max_failures` consecutive threshold-driven failures,
threshold-gate compaction stops firing until a successful compaction
resets the counter. If `compact_on_max_context` is enabled, an overflow
still triggers compaction — and a successful one re-enables the threshold
gate.

```python
# Default: compact at 85% of the LLM's input context window
config = AgentConfig(
    turn_limit=TurnLimit(max_turns=50),
    time_limit=None,
    history_compaction=HistoryCompaction(),
)

# Or set an absolute token threshold:
config = AgentConfig(
    turn_limit=TurnLimit(max_turns=50),
    time_limit=None,
    history_compaction=HistoryCompaction(threshold_tokens=80_000),
)
```

`HistoryCompaction` tells the
agent loop when to invoke the compaction hook. Strategy-specific knobs
(LLM prompt, summary prefix, etc.) live on the hook factory itself.

| Field                            | Default                      | Purpose                                                                                                                                         |
| -------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `threshold_tokens`               | `None`                       | Absolute token threshold for compaction                                                                                                         |
| `threshold_percentage`           | `0.85` (when neither is set) | Fraction of the LLM's input context window                                                                                                      |
| `max_failures`                   | `2`                          | Stop attempting threshold-driven compaction after this many consecutive failures                                                                |
| `compact_on_max_context`         | `False`                      | Opt-in safety net: when a turn query raises `MaxContextWindowExceededError`, compact once and retry once before falling through to an ErrorTurn |
| `max_compaction_context_retries` | `2`                          | Maximum truncate-and-retry attempts inside the default hook when a compaction call is still too large, for either trigger                       |

Specify exactly one of `threshold_tokens` or `threshold_percentage`. Both
paths resolve the LLM's input context window from the model registry via
`_registry_key`, so the LLM must be registry-backed (i.e. constructed via
`get_registry_model`). Raw models (e.g. `get_raw_model(...)`) are unsupported
— compaction is disabled with a warning at agent construction.

#### LLM-summary strategy knobs

The default hook (`llm_summary_compactor`) takes its strategy-specific
knobs as factory arguments:

```python
hook = llm_summary_compactor(
    llm,
    cfg,                                  # the same HistoryCompaction
    prompt=DEFAULT_COMPACTION_PROMPT,     # appended as a trailing user message
    summary_prefix=DEFAULT_SUMMARY_PREFIX,  # prepended to the inserted summary
)
```

If you don't pass `hooks.compaction`, the agent constructs this with the
defaults automatically. Override one or both to customize prompt phrasing
or the handoff message.

The default compaction call reuses `llm` directly with `tools=[]` so the
summary request does not accidentally produce a tool call.

The threshold itself is clamped below the input context window. The summary's
output budget comes from the LLM's configured `max_tokens` or the provider
default; the default compactor no longer overrides it for the summary call.

### Custom compaction strategies

The default summarize-with-LLM strategy is `llm_summary_compactor(llm,
cfg, *, prompt=..., summary_prefix=...)`. To swap in a different approach
(sliding window, tokenizer-accurate sizing, a cheaper compaction model,
"compact every N turns", etc.), pass a `CompactionHook` callable on
`AgentHooks.compaction`. The hook owns the entire compaction policy —
when to fire, how to size, what algorithm to use, when to give up. The
agent only forwards signals and consumes the result.

The hook is called every turn (`trigger='each_turn'`) and on
`MaxContextWindowExceededError` (`trigger='max_context'`). Return
`(history, None)` to opt out for a call. Return
`(new_history, CompactionSummary(...))` to record an attempt; success is
determined by `summary.error is None`.

`state: dict[str, Any]` is a hook-private scratchpad that's fresh per
`Agent.run()` call — use it for failure counters, token estimates, or
anything else the hook needs to remember between turns. Per-run isolation
is automatic; concurrent runs don't share this dict.

`previous_turn` is the prior successful `AgentTurn` — read
`query_result.metadata` and `tool_call_records` off it. It is `None` on
the first call, after an error turn, and on `max_context`. Cheap
estimators can use it; tokenizer-accurate ones can ignore it and analyze
`history` directly.

`tools` is the agent's full tool-definition list supplied to the hook. The
default hook intentionally ignores it and calls the compaction LLM with
`tools=[]`; custom hooks may pass, filter, or ignore the definitions.

```python
from model_library.agent import (
    Agent, AgentHooks, AgentConfig, HistoryCompaction, CompactionSummary,
)

async def sliding_window_compactor(
    history, *, state, trigger, turn_number, compaction_number,
    tools, previous_turn,
    output_dir, question_id, run_id, logger,
):
    # Compact every 10 turns OR on max-context-window. State carries the
    # turn-since-last-compaction counter across calls.
    state["since_last"] = state.get("since_last", 0) + 1
    if trigger == "each_turn" and state["since_last"] < 10:
        return history, None
    if len(history) <= 20:
        return history, None
    new_history = history[:1] + history[-19:]
    state["since_last"] = 0
    return new_history, CompactionSummary(
        turn_number=turn_number,
        summary="<sliding-window trim>",
    )

agent = Agent(
    ...,
    config=AgentConfig(
        ...,
        history_compaction=HistoryCompaction(threshold_tokens=80_000),
    ),
    hooks=AgentHooks(compaction=sliding_window_compactor),
)
```

The agent always appends the returned `CompactionSummary` to
`AgentResult.compactions` (success or failure) and reports compaction
cost separately in `final_compaction_metadata`. The hook is otherwise a
black box from the agent's perspective.

## Logging & Output

- `name` — required, used for logger name and log directory
- `log_dir` — base directory for file output (default `logs/`)
- `question_id` — required, creates a subdirectory per question
- `logger` — optional custom logger; the agent appends `<name><<model_name>>` as a child

The agent builds a timestamped run directory: `log_dir/<name>/<model_name>/<timestamp>_<uuid>/<question_id>/`

### Directory Structure

```text
logs/finance_agent/gpt-4o/2024-01-01_12-00-00_abc123/
├── question_1/
│   ├── agent.log              # full text log (turns, hooks, errors)
│   ├── result.json            # AgentResult with TurnSummary list
│   └── turns/
│       ├── init/
│       │   ├── config.json    # tool definitions, LLM config
│       │   ├── state.json     # initial state
│       │   └── history.json    # serialized initial input
│       ├── turn_001/
│       │   ├── result.json    # raw AgentTurn (full query result + tool records)
│       │   ├── state.json     # state snapshot after turn
│       │   └── history.json    # serialized LLM history after turn
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

| File                | Role                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `agent/agent.py`    | `Agent` class and `AgentResult`                                                                   |
| `agent/config.py`   | `AgentConfig`, `TurnLimit`, `TimeLimit`, `HistoryCompaction`, `truncate_oldest`                   |
| `agent/hooks.py`    | `AgentHooks` dataclass and hook protocols                                                         |
| `agent/metadata.py` | `AgentTurn`, `ErrorTurn`, `CompactionSummary`, `ToolCallRecord`, `TurnSummary`, `ToolCallSummary` |
| `agent/tool.py`     | `Tool` base class, `ToolOutput`                                                                   |
| `agent/conductor/`  | `ConductorAgent` — orchestrates multi-turn conversations between agents                           |
| `utils.py`          | `create_file_logger`, `run_logging`                                                               |
