# ATIF Export

Export agent trajectories in [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) format.

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

| ATIF field                            | model-library source                                |
| ------------------------------------- | --------------------------------------------------- |
| `step.message`                        | `QueryResult.output_text`                           |
| `step.reasoning_content`              | `QueryResult.reasoning`                             |
| `step.tool_calls`                     | `QueryResult.tool_calls`                            |
| `step.observation`                    | `ToolCallRecord.tool_output.output` per call        |
| `step.metrics.prompt_tokens`          | `QueryResultMetadata.total_input_tokens`            |
| `step.metrics.completion_tokens`      | `QueryResultMetadata.total_output_tokens`           |
| `step.metrics.cached_tokens`          | `QueryResultMetadata.cache_read_tokens`             |
| `step.metrics.cost_usd`               | `QueryResultCost.total`                             |
| `step.reasoning_effort`               | passed to `from_agent_result(reasoning_effort=...)` |
| `step.is_copied_context`              | set manually on `ATIFStep`                          |
| `step.source = "system"` (initial)    | `SystemInput.text` from first turn's history        |
| `step.source = "user"` (initial)      | `TextInput.text` from first turn's history          |
| `agent.tool_definitions`              | passed to `from_agent_result(tool_definitions=...)` |
| `trajectory.notes`                    | set manually on `ATIFTrajectory`                    |
| `trajectory.continued_trajectory_ref` | set manually on `ATIFTrajectory`                    |
| `metrics.prompt_token_ids`            | set manually on `ATIFMetrics`                       |
| `metrics.extra`                       | set manually on `ATIFMetrics`                       |

Provider-specific raw responses are retained in `QueryResult.history` as
`RawResponse` items for follow-up model calls, but ATIF export does not duplicate
them into `step.extra`.

History compactions aren't first-party in ATIF. When the agent run includes
compactions, they're dumped into `trajectory.extra["compactions"]` as a list
of serialized `CompactionSummary` records, with an aggregate at
`trajectory.extra["compaction_metrics"]` (`total_prompt_tokens`,
`total_completion_tokens`, `total_cost_usd`, `count`). `final_metrics`
reports task cost only (LLM calls inside agent steps); add the compaction
aggregate for the true bill.
