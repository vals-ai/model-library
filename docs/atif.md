# ATIF Export

Export agent trajectories in [ATIF v1.7](https://www.harborframework.com/docs/agents/trajectory-format) format.

## Usage

### Via Agent.run

```python
result = await agent.run(input=input, question_id="q1", atif_export=True)
# Writes trajectory_atif.json to the agent's output directory (result.output_dir)
```

### Manual conversion

```python
from model_library.atif import ATIFTrajectory

trajectory = ATIFTrajectory.from_agent_result(
    turns=raw_turns,          # list[AgentTurn | ErrorTurn] from a run
    agent_name="my-agent",
    model_name="openai/gpt-4",
)

data = trajectory.to_json_dict()  # dict, None fields excluded
```

## Field mapping

| ATIF field | model-library source |
| --- | --- |
| `step.message` | `QueryResult.output_text`, coerced to `""` when absent |
| `step.reasoning_content` | `QueryResult.reasoning` |
| `step.tool_calls` | `QueryResult.tool_calls` |
| `step.observation` | `ToolCallRecord.tool_output.output` per call |
| `step.metrics.prompt_tokens` | `QueryResultMetadata.total_input_tokens` |
| `step.metrics.completion_tokens` | `QueryResultMetadata.total_output_tokens` |
| `step.metrics.cached_tokens` | `QueryResultMetadata.cache_read_tokens` |
| `step.metrics.cost_usd` | `QueryResultCost.total` |
| `step.model_name` | Actual response model when the provider reports it; otherwise the requested model |
| `step.reasoning_effort` | Passed to `from_agent_result(reasoning_effort=...)` |
| `step.is_copied_context` | Set manually on `ATIFStep` |
| `step.llm_call_count` | `1` for each `AgentTurn` |
| `step.extra.vals.model_routing` | Requested/resolved model and `fallback_used` when the provider reports a fallback |
| Initial `step.source = "system"` | `SystemInput.text` from the first turn's history |
| Initial `step.source = "user"` | `TextInput.text` from the first turn's history |
| Error `step.source = "system"` | `ErrorTurn`; message is `Error: <message>`, with `error_type` and `duration_seconds` in `step.extra` and no step metrics |
| `agent.tool_definitions` | Passed to `from_agent_result(tool_definitions=...)` |
| `trajectory.notes` | Set manually on `ATIFTrajectory` |
| `trajectory.continued_trajectory_ref` | Set manually on `ATIFTrajectory` |
| `observation.results[].extra.vals.tool_model_metrics` | Complete metadata from model calls made by that tool |
| `final_metrics.extra.helper_metrics` | Observed helper-model totals and metadata-bearing tool execution count |
| `final_metrics.extra.combined_metrics` | Observed task and helper-model totals; count is main responses plus helper metadata records |
| `metrics.prompt_token_ids` | Set manually on `ATIFMetrics` |
| `metrics.extra` | Set manually on `ATIFMetrics` |

## Export boundaries

- Provider-specific `RawResponse` items remain in `QueryResult.history` for
  follow-up model calls.
- ATIF export does not duplicate raw provider responses into `step.extra`.
- `ErrorTurn` entries become system steps so trajectory consumers can distinguish
  failures from agent output by `step.extra.error_type`.

## Compaction and billing

History compactions are not first-party ATIF steps:

| Location | Contents |
| --- | --- |
| `trajectory.extra["compactions"]` | Serialized `CompactionSummary` records |
| `trajectory.extra["compaction_metrics"]` | `total_prompt_tokens`, `total_completion_tokens`, `total_cost_usd`, and `count` |
| `final_metrics` | Task LLM calls inside agent steps only |

`final_metrics.extra.combined_metrics` adds observed task and helper usage;
compaction usage remains separate in `trajectory.extra["compaction_metrics"]`.
These are saved usage totals, not full billing. Each sum covers records that
report that value and is partial when other records omit it. A value stays
unknown only when no record reports it. Summed durations are model-call time,
not wall-clock run time.
