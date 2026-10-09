"""ATIF (Agent Trajectory Interchange Format) v1.7 models and converters.

Converts model-library AgentResult/AgentTurn objects into the standardized
ATIF trajectory JSON format for agent interaction logging.

Spec: https://www.harborframework.com/docs/agents/trajectory-format
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, NamedTuple

from model_library.agent.metadata import (
    AgentTurn,
    CompactionSummary,
    ErrorTurn,
)
from model_library.base.input import InputItem, RawResponse, SystemInput, TextInput
from model_library.base.output import ProviderToolEvent, QueryResultMetadata
from model_library.utils import ValsModel


class ATIFAgent(ValsModel):
    name: str
    version: str
    model_name: str
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFMetrics(ValsModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    logprobs: list[float] | None = None
    completion_token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    extra: dict[str, Any] | None = None


class _MetadataTotals(NamedTuple):
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    reasoning_tokens: int | None
    cache_write_tokens: int | None
    duration_seconds: float | None
    record_count: int


class ATIFFinalMetrics(ValsModel):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
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
        initial_input: Sequence[InputItem] = (),
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
        history = initial_input
        if not history and turns and isinstance(turns[0], AgentTurn):
            history = turns[0].query_result.history
        if history:
            for item in history:
                if isinstance(item, RawResponse):
                    break
                elif isinstance(item, SystemInput):
                    step_counter += 1
                    steps.append(
                        ATIFStep(
                            step_id=step_counter,
                            timestamp=turns[0].timestamp
                            if turns
                            else datetime.now(timezone.utc).isoformat(),
                            source="system",
                            message=item.text,
                        )
                    )
                elif isinstance(item, TextInput):
                    step_counter += 1
                    steps.append(
                        ATIFStep(
                            step_id=step_counter,
                            timestamp=turns[0].timestamp
                            if turns
                            else datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=item.text,
                        )
                    )

        # Aggregate metrics
        main_metadata: list[QueryResultMetadata] = []
        helper_metadata: list[QueryResultMetadata] = []

        for turn in turns:
            if isinstance(turn, ErrorTurn):
                main_metadata.append(QueryResultMetadata())
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
            main_metadata.append(metadata)
            helper_metadata.extend(
                record.tool_output.metadata
                for record in turn.tool_call_records
                if record.tool_output.metadata is not None
            )
            step_metrics = _make_step_metrics(metadata)

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

        main_totals = _aggregate_metadata(main_metadata)
        final_extra: dict[str, Any] = {
            key: value
            for key, value in {
                "total_reasoning_tokens": main_totals.reasoning_tokens,
                "total_cache_write_tokens": main_totals.cache_write_tokens,
                "total_duration_seconds": main_totals.duration_seconds,
            }.items()
            if value is not None
        }
        if helper_metadata:
            helper_totals = _aggregate_metadata(helper_metadata)
            combined_totals = _aggregate_metadata(main_metadata + helper_metadata)
            final_extra["helper_metrics"] = _extra_metrics(helper_totals)
            final_extra["combined_metrics"] = _extra_metrics(combined_totals)

        final_metrics = ATIFFinalMetrics(
            total_prompt_tokens=main_totals.prompt_tokens,
            total_completion_tokens=main_totals.completion_tokens,
            total_cached_tokens=main_totals.cached_tokens,
            total_cost_usd=main_totals.cost_usd,
            total_steps=len(steps),
            extra=final_extra or None,
        )

        # Compactions aren't a first-party ATIF concept and their cost is
        # housekeeping overhead, not task cost. Keep them out of
        # final_metrics; expose them in `extra` with a separate aggregate so
        # consumers can compute the true bill as final_metrics + compaction.
        extra = dict(trajectory_extra or {}) or None
        if compactions:
            comp_totals = _aggregate_metadata(
                [c.metadata or QueryResultMetadata() for c in compactions]
            )
            if extra is None:
                extra = {}
            compaction_records: list[dict[str, Any]] = []
            for compaction in compactions:
                record = compaction.model_dump(exclude_none=True)
                if compaction.metadata is not None:
                    record["metadata"] = compaction.metadata.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_unset=True,
                        exclude_computed_fields=True,
                    )
                compaction_records.append(record)
            extra["compactions"] = compaction_records
            extra["compaction_metrics"] = {
                "total_prompt_tokens": comp_totals.prompt_tokens,
                "total_completion_tokens": comp_totals.completion_tokens,
                "total_cost_usd": comp_totals.cost_usd,
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


def _input_tokens(metadata: QueryResultMetadata) -> int | None:
    if (
        "in_tokens" not in metadata.model_fields_set
        or metadata.cache_read_tokens is None
        or metadata.cache_write_tokens is None
    ):
        return None
    return metadata.total_input_tokens


def _output_tokens(metadata: QueryResultMetadata) -> int | None:
    if (
        "out_tokens" not in metadata.model_fields_set
        or metadata.reasoning_tokens is None
    ):
        return None
    return metadata.total_output_tokens


def _aggregate_metadata(metadata: Sequence[QueryResultMetadata]) -> _MetadataTotals:
    input_counts = [
        value for item in metadata if (value := _input_tokens(item)) is not None
    ]
    output_counts = [
        value for item in metadata if (value := _output_tokens(item)) is not None
    ]
    costs = [item.cost.total for item in metadata if item.cost is not None]
    cached_tokens = [
        item.cache_read_tokens
        for item in metadata
        if item.cache_read_tokens is not None
    ]
    reasoning_tokens = [
        item.reasoning_tokens for item in metadata if item.reasoning_tokens is not None
    ]
    cache_write_tokens = [
        item.cache_write_tokens
        for item in metadata
        if item.cache_write_tokens is not None
    ]
    durations = [
        item.duration_seconds for item in metadata if item.duration_seconds is not None
    ]
    return _MetadataTotals(
        prompt_tokens=sum(input_counts)
        if input_counts and len(input_counts) == len(metadata)
        else None,
        completion_tokens=sum(output_counts)
        if output_counts and len(output_counts) == len(metadata)
        else None,
        cached_tokens=sum(cached_tokens)
        if cached_tokens and len(cached_tokens) == len(metadata)
        else None,
        cost_usd=sum(costs) if costs and len(costs) == len(metadata) else None,
        reasoning_tokens=sum(reasoning_tokens)
        if reasoning_tokens and len(reasoning_tokens) == len(metadata)
        else None,
        cache_write_tokens=sum(cache_write_tokens)
        if cache_write_tokens and len(cache_write_tokens) == len(metadata)
        else None,
        duration_seconds=sum(durations)
        if durations and len(durations) == len(metadata)
        else None,
        record_count=len(metadata),
    )


def _extra_metrics(totals: _MetadataTotals) -> dict[str, int | float | None]:
    return {
        "total_prompt_tokens": totals.prompt_tokens,
        "total_completion_tokens": totals.completion_tokens,
        "total_cost_usd": totals.cost_usd,
        "total_reasoning_tokens": totals.reasoning_tokens,
        "total_duration_seconds": totals.duration_seconds,
        "count": totals.record_count,
    }


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
        prompt_tokens=_input_tokens(metadata),
        completion_tokens=_output_tokens(metadata),
        cached_tokens=metadata.cache_read_tokens,
        cost_usd=cost_usd,
        extra=extra or None,
    )


def _model_routing(
    metadata: QueryResultMetadata, requested_model: str
) -> tuple[str, dict[str, Any] | None]:
    if metadata.served_model is not None:
        routing: dict[str, Any] = {
            "requested_model": requested_model,
            "resolved_model": metadata.served_model,
            "routed": True,
        }
        return metadata.served_model, {"vals": {"model_routing": routing}}

    if metadata.fallback is None:
        response_model = metadata.extra.get("anthropic_response_model")
        if not isinstance(response_model, str) or not response_model:
            return requested_model, None
        return (
            response_model if "/" in response_model else f"anthropic/{response_model}"
        ), None

    resolved_model = metadata.fallback.served_model
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
            extra=(
                {
                    "vals": {
                        "tool_model_metrics": record.tool_output.metadata.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude_unset=True,
                            exclude_computed_fields=True,
                        )
                    }
                }
                if record.tool_output.metadata is not None
                else None
            ),
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
