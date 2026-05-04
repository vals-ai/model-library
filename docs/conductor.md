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
|-------|------|-------------|
| `messages` | `list[ConversationMessage]` | Flat list of conversation messages |
| `stop_reason` | `ConductorStopReason` | Why the conversation ended |
| `total_duration_seconds` | `float` | Wall clock for entire run |
| `output_dir` | `Path` | Log directory (excluded from JSON) |

Each `ConversationMessage` has `role` (`"auditor"` or `"target"`) and `result` (full `AgentResult`).

## Stop Conditions

| Reason | When |
|--------|------|
| `AUDITOR_DONE` | Auditor's agent uses a done tool |
| `MAX_EXCHANGES` | `config.max_exchanges` reached |
| `MAX_TIME` | `config.time_limit.max_seconds` exceeded |
| `ERROR` | Agent raises an exception or produces empty response |

Only the auditor can intentionally end the conversation. The conversation can end on an odd number of messages (auditor done without target response).

## Configuration

| Field | Type | Description |
|-------|------|-------------|
| `max_exchanges` | `int` (>= 1) | Maximum auditor-target exchange pairs |
| `time_limit` | `TimeLimit \| None` | Optional wall-clock budget (only `max_seconds` is used) |

## Logging

```
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
