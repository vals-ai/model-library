"""Structured performance telemetry for query results."""

import base64
import gzip
from typing import Literal, cast

from pydantic import (
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from model_library.utils import ValsModel

__all__ = [
    "QueryPerformanceChannel",
    "QueryPerformanceEventType",
    "QueryPerformanceEvent",
    "QueryTimeToFirstToken",
    "QueryPerformanceTimelineEntry",
    "QueryResultPerformance",
    "CompressedQueryResultPerformance",
    "compress_query_result_performance",
    "decompress_query_result_performance",
]

QueryPerformanceChannel = Literal["reasoning", "content", "tool_call"]
QueryPerformanceEventKind = Literal["started", "delta", "ready", "finished"]

QueryPerformanceEventType = Literal[
    "reasoning_started",
    "reasoning_delta",
    "reasoning_finished",
    "content_started",
    "content_delta",
    "content_finished",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_ready",
    "tool_call_finished",
]

_QUERY_PERFORMANCE_EVENT_TYPES: dict[
    QueryPerformanceChannel,
    dict[QueryPerformanceEventKind, QueryPerformanceEventType],
] = {
    "reasoning": {
        "started": "reasoning_started",
        "delta": "reasoning_delta",
        "finished": "reasoning_finished",
    },
    "content": {
        "started": "content_started",
        "delta": "content_delta",
        "finished": "content_finished",
    },
    "tool_call": {
        "started": "tool_call_started",
        "delta": "tool_call_delta",
        "ready": "tool_call_ready",
        "finished": "tool_call_finished",
    },
}


def _event_belongs_to_channel(
    event_type: QueryPerformanceEventType, channel: QueryPerformanceChannel
) -> bool:
    return event_type.startswith(f"{channel}_")


class QueryPerformanceEvent(ValsModel):
    """Canonical provider-normalized point-in-time performance event."""

    model_config = ConfigDict(extra="forbid")

    type: QueryPerformanceEventType = Field(
        description="Canonical event type; never contains raw text, tool arguments, or provider payloads."
    )
    timestamp_ms: int = Field(
        ge=0,
        description="Monotonic elapsed milliseconds from query start.",
    )
    channel_text_start_char: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Inclusive character offset into the channel's final text for text delta events.",
    )
    channel_text_end_char: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Exclusive character offset into the channel's final text for text delta events.",
    )

    @model_validator(mode="after")
    def validate_text_offsets(self) -> Self:
        has_start = self.channel_text_start_char is not None
        has_end = self.channel_text_end_char is not None
        if has_start != has_end:
            raise ValueError("text offset fields must be set together")
        if has_start and self.type not in ("content_delta", "reasoning_delta"):
            raise ValueError("text offsets are only valid for text delta events")
        if (
            self.channel_text_start_char is not None
            and self.channel_text_end_char is not None
            and self.channel_text_end_char < self.channel_text_start_char
        ):
            raise ValueError("channel_text_end_char must be >= channel_text_start_char")
        return self


def create_query_performance_event(
    channel: QueryPerformanceChannel,
    kind: QueryPerformanceEventKind,
    timestamp_ms: int,
    *,
    channel_text_start_char: int | None = None,
    channel_text_end_char: int | None = None,
) -> QueryPerformanceEvent:
    """Create a typed performance event for a channel/kind pair."""
    try:
        event_type = _QUERY_PERFORMANCE_EVENT_TYPES[channel][kind]
    except KeyError as exc:
        raise ValueError(
            f"{kind} is not valid for {channel} performance events"
        ) from exc
    return QueryPerformanceEvent(
        type=event_type,
        timestamp_ms=timestamp_ms,
        channel_text_start_char=channel_text_start_char,
        channel_text_end_char=channel_text_end_char,
    )


class QueryTimeToFirstToken(ValsModel):
    """First-token latency rollups in integer milliseconds."""

    model_config = ConfigDict(extra="forbid")

    any: int | None = Field(
        default=None,
        ge=0,
        description="First generated token/delta across reasoning, content, or tool_call.",
    )
    answer: int | None = Field(
        default=None,
        ge=0,
        description="First non-reasoning token/delta: min(content, tool_call).",
    )
    reasoning: int | None = Field(
        default=None,
        ge=0,
        description="First reasoning/thinking token/delta.",
    )
    content: int | None = Field(
        default=None,
        ge=0,
        description="First assistant text token/delta.",
    )
    tool_call: int | None = Field(
        default=None,
        ge=0,
        description="First tool-call token/delta.",
    )


class QueryPerformanceTimelineEntry(ValsModel):
    """Derived timeline segment for one occurrence of a performance channel."""

    model_config = ConfigDict(extra="forbid")

    channel: QueryPerformanceChannel = Field(
        description="Segment channel: reasoning, content, or tool_call."
    )
    index: int = Field(
        ge=0,
        description="Occurrence index within this channel, e.g. content[0], content[1].",
    )
    start_ms: int = Field(
        default=0,
        ge=0,
        description="Derived from events: first *_started timestamp, or first *_delta timestamp when no start event exists.",
    )
    first_token_ms: int | None = Field(
        default=None,
        ge=0,
        description="Derived from events: first *_delta timestamp.",
    )
    ready_ms: int | None = Field(
        default=None,
        ge=0,
        description="Derived from events for tool_call only: first tool_call_ready timestamp.",
    )
    end_ms: int | None = Field(
        default=None,
        ge=0,
        description="Derived from events: last *_finished timestamp, or the last event timestamp when no finish event exists.",
    )
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Derived from events as end_ms - start_ms when end_ms is known.",
    )
    events: list[QueryPerformanceEvent] = Field(
        default_factory=list,
        repr=False,
        description="Canonical raw timing events used as the source of truth for this segment.",
    )

    @staticmethod
    def _raw_event_type(event: object) -> object:
        if isinstance(event, QueryPerformanceEvent):
            return event.type
        if isinstance(event, dict):
            raw_event = cast(dict[str, object], event)
            return raw_event.get("type")
        return getattr(event, "type", None)

    @staticmethod
    def _raw_event_timestamp_ms(event: object) -> object:
        if isinstance(event, QueryPerformanceEvent):
            return event.timestamp_ms
        if isinstance(event, dict):
            raw_event = cast(dict[str, object], event)
            return raw_event.get("timestamp_ms")
        return getattr(event, "timestamp_ms", None)

    @classmethod
    def _derive_timing_from_raw_events(
        cls, channel: object, events: object
    ) -> dict[str, int | None]:
        if channel not in {"reasoning", "content", "tool_call"} or not isinstance(
            events, list
        ):
            return {}

        event_records: list[tuple[object, int]] = []
        raw_events = cast(list[object], events)
        for event in raw_events:
            event_type = cls._raw_event_type(event)
            timestamp_ms = cls._raw_event_timestamp_ms(event)
            if isinstance(timestamp_ms, int):
                event_records.append((event_type, timestamp_ms))
        if not event_records:
            return {}
        previous_timestamp_ms: int | None = None
        for _, timestamp_ms in event_records:
            if (
                previous_timestamp_ms is not None
                and timestamp_ms < previous_timestamp_ms
            ):
                return {}
            previous_timestamp_ms = timestamp_ms

        def first_timestamp(event_type: str) -> int | None:
            return next(
                (
                    timestamp_ms
                    for raw_type, timestamp_ms in event_records
                    if raw_type == event_type
                ),
                None,
            )

        def last_timestamp(event_type: str) -> int | None:
            return next(
                (
                    timestamp_ms
                    for raw_type, timestamp_ms in reversed(event_records)
                    if raw_type == event_type
                ),
                None,
            )

        ready_ms = (
            first_timestamp("tool_call_ready") if channel == "tool_call" else None
        )
        first_token_ms = first_timestamp(f"{channel}_delta")
        started_ms = first_timestamp(f"{channel}_started")
        start_ms = (
            started_ms
            if started_ms is not None
            else first_token_ms
            if first_token_ms is not None
            else event_records[0][1]
        )
        finished_ms = last_timestamp(f"{channel}_finished")
        end_ms = finished_ms if finished_ms is not None else event_records[-1][1]
        return {
            "start_ms": start_ms,
            "first_token_ms": first_token_ms,
            "ready_ms": ready_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
        }

    @model_validator(mode="before")
    @classmethod
    def derive_timing_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_data = cast(dict[str, object], data)
        derived = cls._derive_timing_from_raw_events(
            raw_data.get("channel"), raw_data.get("events")
        )
        if not derived:
            return raw_data
        return {**raw_data, **derived}

    @model_validator(mode="after")
    def validate_timeline_entry(self) -> Self:
        if not self.events:
            raise ValueError("timeline entry must include events")

        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if self.first_token_ms is not None and self.first_token_ms < self.start_ms:
            raise ValueError("first_token_ms must be >= start_ms")
        if (
            self.first_token_ms is not None
            and self.end_ms is not None
            and self.first_token_ms > self.end_ms
        ):
            raise ValueError("first_token_ms must be <= end_ms")
        if self.ready_ms is not None and self.ready_ms < self.start_ms:
            raise ValueError("ready_ms must be >= start_ms")
        if (
            self.ready_ms is not None
            and self.end_ms is not None
            and self.ready_ms > self.end_ms
        ):
            raise ValueError("ready_ms must be <= end_ms")
        if self.ready_ms is not None and self.channel != "tool_call":
            raise ValueError("ready_ms is only valid for tool_call segments")

        previous_timestamp_ms: int | None = None
        for event in self.events:
            if not _event_belongs_to_channel(event.type, self.channel):
                raise ValueError(
                    f"{event.type} does not belong to {self.channel} segment"
                )
            if event.timestamp_ms < self.start_ms:
                raise ValueError("event timestamp_ms must be >= start_ms")
            if self.end_ms is not None and event.timestamp_ms > self.end_ms:
                raise ValueError("event timestamp_ms must be <= end_ms")
            if (
                previous_timestamp_ms is not None
                and event.timestamp_ms < previous_timestamp_ms
            ):
                raise ValueError("timeline events must be ordered by timestamp_ms")
            previous_timestamp_ms = event.timestamp_ms

        return self


class QueryResultPerformance(ValsModel):
    """Structured per-query performance telemetry.

    Token throughput is intentionally not derived per query here. Compute
    aggregate TPS from tokens, durations, and internal data
    """

    model_config = ConfigDict(extra="forbid")

    time_to_first_token_ms: QueryTimeToFirstToken = Field(
        default_factory=QueryTimeToFirstToken,
        description="First-token latency rollups derived from timeline events.",
    )
    timeline: list[QueryPerformanceTimelineEntry] = Field(
        default_factory=list,
        description="Ordered output segments; supports repeated content/reasoning/tool-call phases.",
    )

    @staticmethod
    def _minimum_ms(values: list[int | None]) -> int | None:
        known_values = [value for value in values if value is not None]
        if not known_values:
            return None
        return min(known_values)

    def _compute_time_to_first_token_ms(self) -> QueryTimeToFirstToken:
        reasoning = self._minimum_ms(
            [
                entry.first_token_ms
                for entry in self.timeline
                if entry.channel == "reasoning"
            ]
        )
        content = self._minimum_ms(
            [
                entry.first_token_ms
                for entry in self.timeline
                if entry.channel == "content"
            ]
        )
        tool_call = self._minimum_ms(
            [
                entry.first_token_ms
                for entry in self.timeline
                if entry.channel == "tool_call"
            ]
        )
        return QueryTimeToFirstToken(
            any=self._minimum_ms([reasoning, content, tool_call]),
            answer=self._minimum_ms([content, tool_call]),
            reasoning=reasoning,
            content=content,
            tool_call=tool_call,
        )

    @model_validator(mode="after")
    def validate_timeline_and_derive_time_to_first_token_ms(self) -> Self:
        previous_start_ms: int | None = None
        expected_indexes: dict[QueryPerformanceChannel, int] = {
            "reasoning": 0,
            "content": 0,
            "tool_call": 0,
        }
        for entry in self.timeline:
            if previous_start_ms is not None and entry.start_ms < previous_start_ms:
                raise ValueError("timeline entries must be ordered by start_ms")
            expected_index = expected_indexes[entry.channel]
            if entry.index != expected_index:
                raise ValueError(
                    f"timeline {entry.channel} indexes must be contiguous from 0"
                )
            expected_indexes[entry.channel] = expected_index + 1
            previous_start_ms = entry.start_ms

        if self.timeline:
            self.time_to_first_token_ms = self._compute_time_to_first_token_ms()
        return self


class CompressedQueryResultPerformance(ValsModel):
    """Lossless JSON representation of compressed query performance telemetry."""

    model_config = ConfigDict(extra="forbid")

    encoding: Literal["gzip+base64"] = "gzip+base64"
    data: str


def compress_query_result_performance(
    performance: QueryResultPerformance,
) -> CompressedQueryResultPerformance:
    payload = performance.model_dump_json().encode()
    compressed = gzip.compress(payload, compresslevel=1, mtime=0)
    return CompressedQueryResultPerformance(
        data=base64.b64encode(compressed).decode("ascii")
    )


def decompress_query_result_performance(
    performance: CompressedQueryResultPerformance,
) -> QueryResultPerformance:
    compressed = base64.b64decode(performance.data, validate=True)
    payload = gzip.decompress(compressed)
    return QueryResultPerformance.model_validate_json(payload)
