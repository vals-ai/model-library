"""ATIF (Agent Trajectory Interchange Format) v1.6 models and converters.

Converts model-library AgentResult/AgentTurn objects into the standardized
ATIF trajectory JSON format for agent interaction logging.

Spec: https://www.harborframework.com/docs/agents/trajectory-format
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from model_library.agent.metadata import (
    AgentTurn,
    CompactionSummary,
    ErrorTurn,
    ToolCallRecord,
)
from model_library.base.input import RawResponse, SystemInput, TextInput
from model_library.base.output import QueryResultMetadata
from model_library.utils import ValsModel


class ATIFAgent(ValsModel):
    name: str
    version: str
    model_name: str
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFMetrics(ValsModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None
    cost_usd: float | None = None
    logprobs: list[float] | None = None
    completion_token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    extra: dict[str, Any] | None = None


class ATIFFinalMetrics(ValsModel):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int
    extra: dict[str, Any] | None = None


class ATIFObservationResult(ValsModel):
    source_call_id: str
    content: str


class ATIFObservation(ValsModel):
    results: list[ATIFObservationResult]


class ATIFToolCall(ValsModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


class ATIFStep(ValsModel):
    step_id: int
    timestamp: str
    source: str  # "user", "agent", or "system"
    message: str
    model_name: str | None = None
    reasoning_content: str | None = None
    reasoning_effort: str | float | None = None
    is_copied_context: bool | None = None
    tool_calls: list[ATIFToolCall] | None = None
    observation: ATIFObservation | None = None
    metrics: ATIFMetrics | None = None
    extra: dict[str, Any] | None = None


class ATIFTrajectory(ValsModel):
    schema_version: str = "ATIF-v1.6"
    session_id: str
    notes: str | None = None
    continued_trajectory_ref: str | None = None
    agent: ATIFAgent
    steps: list[ATIFStep]
    final_metrics: ATIFFinalMetrics
    extra: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_agent_result(
        cls,
        *,
        turns: Sequence[AgentTurn | ErrorTurn],
        compactions: Sequence[CompactionSummary] = (),
        agent_name: str,
        model_name: str,
        agent_version: str = "1.0",
        session_id: str | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        reasoning_effort: str | float | None = None,
        agent_extra: dict[str, Any] | None = None,
    ) -> "ATIFTrajectory":
        """Convert agent turns into an ATIF trajectory.

        Initial user/system messages are extracted from the first AgentTurn's
        history (all items before the first RawResponse).

        Args:
            turns: The list of AgentTurn/ErrorTurn from the agent run.
            compactions: History compaction attempts from the agent run.
            agent_name: Name of the agent.
            model_name: Model key used for the agent.
            agent_version: Version string for the agent.
            session_id: Optional session ID. Generated if not provided.
            tool_definitions: Optional list of tool definitions in OpenAI function calling format.
            reasoning_effort: Optional reasoning effort value passed to each agent step.
            agent_extra: Optional extra metadata to attach to the agent record.
        """
        steps: list[ATIFStep] = []
        step_counter = 0

        # Extract initial messages from first AgentTurn's history
        if turns and isinstance(turns[0], AgentTurn):
            for item in turns[0].query_result.history:
                if isinstance(item, RawResponse):
                    break
                elif isinstance(item, SystemInput):
                    step_counter += 1
                    steps.append(
                        ATIFStep(
                            step_id=step_counter,
                            timestamp=turns[0].timestamp,
                            source="system",
                            message=item.text,
                        )
                    )
                elif isinstance(item, TextInput):
                    step_counter += 1
                    steps.append(
                        ATIFStep(
                            step_id=step_counter,
                            timestamp=turns[0].timestamp,
                            source="user",
                            message=item.text,
                        )
                    )

        # Aggregate metrics
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0
        has_cached = False
        has_cost = False

        for turn in turns:
            if isinstance(turn, ErrorTurn):
                step_counter += 1
                steps.append(
                    ATIFStep(
                        step_id=step_counter,
                        timestamp=turn.timestamp,
                        source="system",
                        message=f"Error: {turn.error.message}",
                        extra={
                            "error_type": turn.error.type,
                            "duration_seconds": turn.duration_seconds,
                        },
                    )
                )
                continue

            metadata = turn.query_result.metadata
            step_metrics = _make_step_metrics(metadata)
            total_prompt += step_metrics.prompt_tokens
            total_completion += step_metrics.completion_tokens
            if step_metrics.cached_tokens is not None:
                has_cached = True
                total_cached += step_metrics.cached_tokens
            if step_metrics.cost_usd is not None:
                has_cost = True
                total_cost += step_metrics.cost_usd

            step_counter += 1
            steps.append(
                ATIFStep(
                    step_id=step_counter,
                    timestamp=turn.timestamp,
                    source="agent",
                    message=turn.query_result.output_text or "",
                    model_name=model_name,
                    reasoning_content=turn.query_result.reasoning,
                    reasoning_effort=reasoning_effort,
                    tool_calls=_make_tool_calls(turn),
                    observation=_make_observation(turn.tool_call_records),
                    metrics=step_metrics,
                )
            )

        final_metrics = ATIFFinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached if has_cached else None,
            total_cost_usd=total_cost if has_cost else None,
            total_steps=len(steps),
        )

        # Compactions aren't a first-party ATIF concept and their cost is
        # housekeeping overhead, not task cost. Keep them out of
        # final_metrics; expose them in `extra` with a separate aggregate so
        # consumers can compute the true bill as final_metrics + compaction.
        extra: dict[str, Any] | None = None
        if compactions:
            comp_prompt = sum(
                c.metadata.total_input_tokens for c in compactions if c.metadata
            )
            comp_completion = sum(
                c.metadata.total_output_tokens for c in compactions if c.metadata
            )
            # Match final_metrics: distinguish "no cost data" (None) from "zero
            # cost" (0.0). Plain `sum(...) or None` would coerce a real 0.0 to
            # None.
            comp_cost = 0.0
            comp_has_cost = False
            for c in compactions:
                if c.metadata is not None and c.metadata.cost is not None:
                    comp_cost += c.metadata.cost.total
                    comp_has_cost = True
            extra = {
                "compactions": [c.model_dump(exclude_none=True) for c in compactions],
                "compaction_metrics": {
                    "total_prompt_tokens": comp_prompt,
                    "total_completion_tokens": comp_completion,
                    "total_cost_usd": comp_cost if comp_has_cost else None,
                    "count": len(compactions),
                },
            }

        return cls(
            session_id=session_id or str(uuid.uuid4()),
            agent=ATIFAgent(
                name=agent_name,
                version=agent_version,
                model_name=model_name,
                tool_definitions=tool_definitions,
                extra=agent_extra,
            ),
            steps=steps,
            final_metrics=final_metrics,
            extra=extra,
        )


def _make_step_metrics(metadata: QueryResultMetadata) -> ATIFMetrics:
    """Convert QueryResultMetadata to ATIFMetrics."""
    cost_usd = metadata.cost.total if metadata.cost else None
    return ATIFMetrics(
        prompt_tokens=metadata.total_input_tokens,
        completion_tokens=metadata.total_output_tokens,
        cached_tokens=metadata.cache_read_tokens,
        cost_usd=cost_usd,
    )


def _make_observation(records: list[ToolCallRecord]) -> ATIFObservation | None:
    """Convert tool call records to an ATIF observation."""
    if not records:
        return None
    return ATIFObservation(
        results=[
            ATIFObservationResult(
                source_call_id=r.tool_call.id,
                content=r.tool_output.output,
            )
            for r in records
        ]
    )


def _make_tool_calls(turn: AgentTurn) -> list[ATIFToolCall] | None:
    """Convert AgentTurn tool calls to ATIF tool calls."""
    if not turn.query_result.tool_calls:
        return None
    return [
        ATIFToolCall(
            tool_call_id=tc.id,
            function_name=tc.name,
            arguments=tc.parsed_args
            if tc.parsed_args is not None
            else {"raw_arguments": tc.args},
        )
        for tc in turn.query_result.tool_calls
    ]
