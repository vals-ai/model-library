# Agent

Tool-augmented conversation loop that queries an LLM and executes tools until a stopping condition is met.

## Logging & Output

```python
agent = Agent(llm, tools, name="finance_agent", config=AgentConfig(turn_limit=TurnLimit(max_turns=50), time_limit=None))
result = await agent.run(input, question_id="question_1")
result.output_dir  # Path to question-level output directory
```

- `name` — required, used for logger name and log directory
- `log_dir` — base directory for file output (default `logs/`)
- `question_id` — required, creates a subdirectory per question
- `logger` — optional custom logger; the agent appends `<name><<model_name>>` as a child

The agent creates its own logger at `agent.<name><<model_name>>` and automatically builds a timestamped run directory:
`log_dir/<name>/<model_name>/<timestamp>_<uuid>/<question_id>/`

Console output (INFO level) is provided by the `agent` and `llm` loggers' RichHandler — call `set_logging()` to enable it. LLM query logs are filtered from console output but written to the file log.

### Custom Logger

Pass a custom logger (e.g. CloudWatch) to route all agent and LLM logs through it.

**Note:** The agent overrides the LLM's logger (even a custom one) so that all LLM logs become children of the agent logger. This keeps the full hierarchy under one tree for file logging and console filtering.

```python
import logging

my_logger = logging.getLogger("myapp.cloudwatch")
agent = Agent(name="finance_agent", llm=llm, tools=tools, logger=my_logger)
# logs at: myapp.cloudwatch.finance_agent<gpt-4o>, myapp.cloudwatch.finance_agent<gpt-4o>.<run=...>, etc.
```

Similarly for standalone LLM usage:

```python
from model_library import model

llm = model("openai/gpt-4o", logger=my_logger)
# logs at: myapp.cloudwatch.openai.gpt-4o<run=...>
```

### Case 1: Default

```python
agent = Agent(name="finance_agent", llm=llm, tools=tools)
await agent.run(input, question_id="question_1")
await agent.run(input, question_id="question_2")
```

```
logs/finance_agent/gpt-4o/2024-01-01_12-00-00_abc123/
├── question_1/
│   ├── agent.log
│   ├── result.json
│   └── histories/         # if serialize_histories
│       ├── turn_000.bin
│       └── turn_001.bin
└── question_2/
    ├── agent.log
    ├── result.json
    └── histories/
```

### Case 2: Custom `log_dir`

```python
agent = Agent(name="finance_agent", llm=llm, tools=tools, log_dir=Path("/data/runs"))
await agent.run(input, question_id="question_1")
```

```
/data/runs/finance_agent/gpt-4o/2024-01-01_12-00-00_abc123/
└── question_1/
    ├── agent.log
    ├── result.json
    └── histories/
```

### Case 3: Logger already has a FileHandler

```python
logger = logging.getLogger("agent.finance_agent<gpt-4o>")
logger.addHandler(logging.FileHandler("/custom/path/output/agent.log"))

agent = Agent(name="finance_agent", llm=llm, tools=tools)
await agent.run(input, question_id="question_1")
```

```
/custom/path/output/
├── agent.log
└── question_1/
    ├── result.json
    └── histories/
```

The agent detects an existing FileHandler on the logger or any of its parents (excluding root) and reuses it (no new handler created). Output files are still scoped under `<question_id>/`. This also works when passing a custom `logger` with a FileHandler:

```python
my_logger = logging.getLogger("myapp")
my_logger.addHandler(logging.FileHandler("/custom/path/output/agent.log"))

agent = Agent(name="finance_agent", llm=llm, tools=tools, logger=my_logger)
await agent.run(input, question_id="question_1")
# logs to /custom/path/output/agent.log, output files in /custom/path/output/question_1/
```

## Output Files

- `agent.log` — full agent log (turns, tool calls, errors)
- `result.json` — serialized `AgentResult`
- `histories/turn_NNN.bin` — per-turn LLM input history (only when `serialize_histories=True`)

## Stopping Conditions

The loop stops when any of these occur:

- A tool returns `done=True`
- `should_stop` hook returns `True` (default: text-only response)
- `turn_limit.max_turns` reached (sets `MaxTurnsExceeded` error)
- `time_limit.max_seconds` exceeded (sets `MaxTimeExceeded` error)
- `before_query` hook re-raises a query error (default behavior)
- Unhandled exception

After the loop, `determine_answer` runs with full context. Default falls back to done tool output or last LLM text.

## AgentResult

`AgentResult` includes per-turn data, durations, token/cost metadata, and the output directory:

- `output_dir: Path` — the question-level directory where `result.json`, `agent.log`, and `histories/` are written. Excluded from serialization (`model_dump`/`model_dump_json`).

## Configuration

`AgentConfig` fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `turn_limit` | required | `TurnLimit(max_turns=N)` or `None` for unlimited |
| `time_limit` | required | `TimeLimit(max_seconds=N)` or `None` for unlimited |
| `max_tool_calls_per_turn` | `None` | Cap on executed tool calls per turn |
| `serialize_histories` | `True` | Save per-turn `.bin` histories to disk |

## Source Files

| File | Role |
|------|------|
| `agent/agent.py` | `Agent` class and `AgentResult` |
| `agent/config.py` | `AgentConfig`, `truncate_oldest` helper |
| `agent/hooks.py` | `AgentHooks` dataclass (all hook callbacks) |
| `agent/metadata.py` | `AgentTurn`, `ErrorTurn`, `ToolCallRecord`, `SerializableException` |
| `agent/tool.py` | `Tool` base class, `ToolOutput` |
| `utils.py` | `create_file_logger`, `run_logging` |
