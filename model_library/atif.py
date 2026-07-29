"""ATIF (Agent Trajectory Interchange Format) v1.7 models and converters.

Converts model-library AgentResult/AgentTurn objects into the standardized
ATIF trajectory JSON format for agent interaction logging.

Spec: https://www.harborframework.com/docs/agents/trajectory-format
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from model_library.agent.metadata import (
    AgentTurn,
    CompactionSummary,
    ErrorTurn,
)
from model_library.base.input import RawResponse, SystemInput, TextInput
from model_library.base.output import ProviderToolEvent, QueryResultMetadata
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
    extra: dict[str, Any] | None = None


class ATIFObservation(ValsModel):
    results: list[ATIFObservationResult]


class ATIFToolCall(ValsModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class ATIFStep(ValsModel):
    step_id: int
    timestamp: str
    source: str  # "user", "agent", or "system"
    message: str
    model_name: str | None = None
    reasoning_content: str | None = None
    reasoning_effort: str | float | None = None
    is_copied_context: bool | None = None
    llm_call_count: int | None = None
    tool_calls: list[ATIFToolCall] | None = None
    observation: ATIFObservation | None = None
    metrics: ATIFMetrics | None = None
    extra: dict[str, Any] | None = None


class ATIFTrajectory(ValsModel):
    schema_version: str = "ATIF-v1.7"
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
        trajectory_extra: dict[str, Any] | None = None,
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
            trajectory_extra: Optional extra metadata to attach to the trajectory.
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
        total_reasoning = 0
        total_cache_write = 0
        total_duration = 0.0
        has_cached = False
        has_cost = False
        has_reasoning = False
        has_cache_write = False
        has_duration = False

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
            if metadata.reasoning_tokens is not None:
                has_reasoning = True
                total_reasoning += metadata.reasoning_tokens
            if metadata.cache_write_tokens is not None:
                has_cache_write = True
                total_cache_write += metadata.cache_write_tokens
            if metadata.duration_seconds is not None:
                has_duration = True
                total_duration += metadata.duration_seconds

            turn_model_name, turn_extra = _model_routing(metadata, model_name)

            step_counter += 1
            steps.append(
                ATIFStep(
                    step_id=step_counter,
                    timestamp=turn.timestamp,
                    source="agent",
                    message=turn.query_result.output_text or "",
                    model_name=turn_model_name,
                    reasoning_content=turn.query_result.reasoning,
                    reasoning_effort=reasoning_effort,
                    llm_call_count=1,
                    tool_calls=_make_tool_calls(turn),
                    observation=_make_observation(turn),
                    metrics=step_metrics,
                    extra=turn_extra,
                )
            )

        final_extra = {
            key: value
            for key, value in {
                "total_reasoning_tokens": total_reasoning if has_reasoning else None,
                "total_cache_write_tokens": total_cache_write
                if has_cache_write
                else None,
                "total_duration_seconds": total_duration if has_duration else None,
            }.items()
            if value is not None
        }
        final_metrics = ATIFFinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached if has_cached else None,
            total_cost_usd=total_cost if has_cost else None,
            total_steps=len(steps),
            extra=final_extra or None,
        )

        # Compactions aren't a first-party ATIF concept and their cost is
        # housekeeping overhead, not task cost. Keep them out of
        # final_metrics; expose them in `extra` with a separate aggregate so
        # consumers can compute the true bill as final_metrics + compaction.
        extra = dict(trajectory_extra or {}) or None
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
            if extra is None:
                extra = {}
            extra["compactions"] = [
                c.model_dump(exclude_none=True) for c in compactions
            ]
            extra["compaction_metrics"] = {
                "total_prompt_tokens": comp_prompt,
                "total_completion_tokens": comp_completion,
                "total_cost_usd": comp_cost if comp_has_cost else None,
                "count": len(compactions),
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
    extra = {
        key: value
        for key, value in {
            "reasoning_tokens": metadata.reasoning_tokens,
            "cache_write_tokens": metadata.cache_write_tokens,
            "duration_seconds": metadata.duration_seconds,
        }.items()
        if value is not None
    }
    return ATIFMetrics(
        prompt_tokens=metadata.total_input_tokens,
        completion_tokens=metadata.total_output_tokens,
        cached_tokens=metadata.cache_read_tokens,
        cost_usd=cost_usd,
        extra=extra or None,
    )


def _model_routing(
    metadata: QueryResultMetadata, requested_model: str
) -> tuple[str, dict[str, Any] | None]:
    response_model = metadata.extra.get("anthropic_response_model")
    if not isinstance(response_model, str) or not response_model:
        return requested_model, None

    resolved_model = (
        response_model if "/" in response_model else f"anthropic/{response_model}"
    )
    if metadata.extra.get("fallback") is not True:
        return resolved_model, None

    return resolved_model, {
        "vals": {
            "model_routing": {
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "fallback_used": True,
            }
        }
    }


def _provider_call_id(event: ProviderToolEvent, index: int) -> str:
    return f"provider:{index + 1}:{event.id or event.name}"


def _provider_event_extra(event: ProviderToolEvent) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "provider": event.provider,
            "type": event.type,
            "status": event.status,
            "sequence": event.sequence,
            "native_id": event.id,
        }.items()
        if value is not None
    }


def _make_observation(turn: AgentTurn) -> ATIFObservation | None:
    """Convert tool call records to an ATIF observation."""
    results = [
        ATIFObservationResult(
            source_call_id=record.tool_call.id,
            content=record.tool_output.output,
        )
        for record in turn.tool_call_records
    ]
    for index, event in enumerate(turn.query_result.provider_tool_events):
        if event.output is None:
            continue
        content = (
            event.output
            if isinstance(event.output, str)
            else json.dumps(event.output, ensure_ascii=False, sort_keys=True)
        )
        results.append(
            ATIFObservationResult(
                source_call_id=_provider_call_id(event, index),
                content=content,
                extra=_provider_event_extra(event),
            )
        )
    if not results:
        return None
    return ATIFObservation(results=results)


def _make_tool_calls(turn: AgentTurn) -> list[ATIFToolCall] | None:
    """Convert AgentTurn tool calls to ATIF tool calls."""
    calls = [
        ATIFToolCall(
            tool_call_id=tc.id,
            function_name=tc.name,
            arguments=tc.parsed_args
            if tc.parsed_args is not None
            else {"raw_arguments": tc.args},
        )
        for tc in turn.query_result.tool_calls
    ]
    calls.extend(
        ATIFToolCall(
            tool_call_id=_provider_call_id(event, index),
            function_name=event.name,
            arguments=dict(event.input)
            if isinstance(event.input, dict)
            else {"input": event.input},
            extra=_provider_event_extra(event),
        )
        for index, event in enumerate(turn.query_result.provider_tool_events)
    )
    return calls or None
