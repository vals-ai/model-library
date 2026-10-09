# ConductorAgent

Orchestrates multi-turn conversations between an auditor Agent and a target Agent for persona-based evaluations.

## Quick Start

```python
from model_library.agent import Agent, AgentConfig, ConductorAgent, ConductorConfig, TurnLimit
from model_library.base.input import SystemInput

auditor = Agent(
    name="auditor",
    llm=auditor_model,
    tools=[done_tool],
    config=AgentConfig(turn_limit=TurnLimit(max_turns=3), time_limit=None),
)
target = Agent(
    name="target",
    llm=target_model,
    tools=[lookup_tool],
    config=AgentConfig(turn_limit=TurnLimit(max_turns=3), time_limit=None),
)

conductor = ConductorAgent(
    auditor=auditor,
    target=target,
    auditor_system_prompt=SystemInput(text="You are role-playing as a customer..."),
    target_system_prompt=SystemInput(text="You are a helpful support agent..."),
    name="eval-run",
    config=ConductorConfig(max_exchanges=5),
)

result = await conductor.run(question_id="q1")

result.messages          # list[ConversationMessage] — flat conversation
result.stop_reason       # ConductorStopReason
result.output_dir        # Path to log directory
```

## How It Works

1. Conductor sends `initial_prompt` (default: "Start the conversation. Send your first message in character.") to the auditor with its system prompt
2. Auditor's response is forwarded to the target with its system prompt
3. Target's response is forwarded back to the auditor
4. Repeat until a stop condition is met
5. History is managed externally via `AgentResult.final_history` — Agent stays stateless

## ConductorResult

| Field | Type | Description |
| --- | --- | --- |
| `messages` | `list[ConversationMessage]` | Flat list of conversation messages |
| `stop_reason` | `ConductorStopReason` | Why the conversation ended |
| `total_duration_seconds` | `float` | Wall clock for the entire run |
| `auditor_aggregated_metadata` | `QueryResultMetadata` | Sum of each auditor result's `final_aggregated_metadata`; excludes compaction metadata |
| `target_aggregated_metadata` | `QueryResultMetadata` | Sum of each target result's `final_aggregated_metadata`; excludes compaction metadata |
| `final_aggregated_metadata` | `QueryResultMetadata` | Auditor plus target aggregate; excludes compaction metadata |
| `output_dir` | `Path` | Log directory, excluded from JSON |

For total billing metadata, add the nested `AgentResult.final_compaction_metadata`
values separately; the conductor aggregates task-query metadata only.

Each `ConversationMessage` has `role` (`"auditor"` or `"target"`) and `result` (full `AgentResult`).

## Stop Conditions

| Reason | When |
| --- | --- |
| `AUDITOR_DONE` | Auditor's agent uses a done tool |
| `MAX_EXCHANGES` | `config.max_exchanges` reached |
| `MAX_TIME` | The pre-exchange time check finds `config.time_limit.max_seconds` already reached |
| `ERROR` | Agent raises an exception or produces an empty response |

The time limit is checked before each exchange, not during auditor or target
execution. An exchange may therefore finish after the budget, and an overrun in
the final configured exchange can still end as `MAX_EXCHANGES`.

Only the auditor can intentionally end the conversation. The conversation can end on an odd number of messages (auditor done without target response).

## Configuration

| Field | Type | Description |
| --- | --- | --- |
| `max_exchanges` | `int` (>= 1) | Maximum auditor-target exchange pairs |
| `time_limit` | `TimeLimit \| None` | Optional pre-exchange wall-clock budget; only `max_seconds` is used |

## Logging

```text
logs/<name>/<auditor_model>_<target_model>/<timestamp>_<uuid>/
├── <question_id>/
│   ├── agent.log
│   ├── result.json         # ConductorResult
│   ├── transcript.json     # [{role, content}, ...]
│   └── exchanges/
│       └── init/
│           ├── config.json  # conductor + agent configs, system prompts
│           └── state.json
```

## Source Files

| File | Role |
|------|------|
| `agent/conductor/conductor.py` | `ConductorAgent` class |
| `agent/conductor/config.py` | `ConductorConfig` |
| `agent/conductor/metadata.py` | `ConversationMessage`, `ConductorResult`, `ConductorStopReason` |

## Explicit opening and durable progress

Pass `first_message` to `ConductorAgent.run` to send an authored opening directly to the target. The auditor starts on exchange two. The transcript retains the opening as an auditor message, with no model call or token charge. Omit this argument to keep the auditor-generated opening.

Each agent stores exchange files under `exchange_NNN/<question_id>/turns`. Earlier exchanges remain available after later exchanges finish. An existing parent file handler remains in use for logs without changing these artifact paths.

Pass `on_progress: Callable[[Path], None]` to `Agent` to update external artifacts from native files. The callback receives the current agent output directory. The agent calls it after initialization, response progress, helper-query capture, and error capture. A response is available before its tools finish.

The callback is synchronous and can run on a worker thread. It must finish before that write returns. Callback exceptions are logged without replacing the conversation outcome. Consumers that require artifacts must reject missing evidence at their release boundary.

Use the existing native `turns/init`, `turns/turn_NNN/result.json`, and `turns/turn_NNN/error.json` files. Helper results remain under `helper_queries/tool_NNN`. Native response metadata omits unset fields, so absent usage differs from a measured zero. A progress callback can observe multiple versions of one turn and must replace that turn instead of counting it again.

Native capture failures appear in `message.result.native_capture_errors`. This list preserves the failure stage and exception type without replacing the conversation result. Artifact collectors must report incomplete evidence when this list is nonempty. The list covers response, state, history, helper, and progress-callback failures.
