"""Mutable helpers for assembling query results from provider streams."""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from typing_extensions import Self

from model_library.base.input import InputItem, ToolCall
from model_library.base.output.performance import (
    QueryPerformanceChannel,
    QueryPerformanceEvent,
    QueryPerformanceEventKind,
    QueryPerformanceTimelineEntry,
    QueryResultPerformance,
    create_query_performance_event,
)

if TYPE_CHECKING:
    from model_library.base.output.result import (
        FinishReasonInfo,
        ProviderToolEvent,
        QueryResult,
        QueryResultExtras,
        QueryResultMetadata,
    )

__all__ = ["QueryResultBuilder"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class QueryResultBuilder:
    """Build a ``QueryResult`` while recording provider-normalized timing events."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start_time = clock()
        self._output_text = ""
        self._reasoning = ""
        self._timeline: list[
            tuple[QueryPerformanceChannel, int, list[QueryPerformanceEvent]]
        ] = []
        self._active_channel: QueryPerformanceChannel | None = None
        self._active_events: list[QueryPerformanceEvent] | None = None
        self._tool_call_events_by_key: dict[Hashable, list[QueryPerformanceEvent]] = {}
        self._text_offset_invalid_channels: set[QueryPerformanceChannel] = set()
        self._channel_counts: dict[QueryPerformanceChannel, int] = {
            "reasoning": 0,
            "content": 0,
            "tool_call": 0,
        }
        self._built = False

    def _ensure_not_built(self) -> None:
        if self._built:
            raise RuntimeError("QueryResultBuilder cannot be reused after build()")

    @staticmethod
    def _requires_unbuilt(
        method: Callable[Concatenate[QueryResultBuilder, _P], _R],
    ) -> Callable[Concatenate[QueryResultBuilder, _P], _R]:
        @wraps(method)
        def wrapper(
            self: QueryResultBuilder, /, *args: _P.args, **kwargs: _P.kwargs
        ) -> _R:
            self._ensure_not_built()
            return method(self, *args, **kwargs)

        return wrapper

    @property
    def output_text(self) -> str | None:
        return self._output_text or None

    @property
    def reasoning(self) -> str | None:
        return self._reasoning or None

    @property
    def has_output_text(self) -> bool:
        return self.output_text is not None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None

    def _elapsed_ms(self) -> int:
        return int(round((self._clock() - self._start_time) * 1000))

    def _event(
        self,
        channel: QueryPerformanceChannel,
        kind: QueryPerformanceEventKind,
        timestamp_ms: int,
        *,
        channel_text_start_char: int | None = None,
        channel_text_end_char: int | None = None,
    ) -> QueryPerformanceEvent:
        return create_query_performance_event(
            channel,
            kind,
            timestamp_ms,
            channel_text_start_char=channel_text_start_char,
            channel_text_end_char=channel_text_end_char,
        )

    def _start_segment(
        self, channel: QueryPerformanceChannel, timestamp_ms: int
    ) -> list[QueryPerformanceEvent]:
        index = self._channel_counts[channel]
        self._channel_counts[channel] += 1
        events = [self._event(channel, "started", timestamp_ms)]
        self._timeline.append((channel, index, events))
        self._active_channel = channel
        self._active_events = events
        return events

    def _finish_events(
        self,
        channel: QueryPerformanceChannel,
        events: list[QueryPerformanceEvent],
        timestamp_ms: int,
    ) -> None:
        finish_type = f"{channel}_finished"
        if not events or events[-1].type != finish_type:
            events.append(self._event(channel, "finished", timestamp_ms))

    def _finish_active_segment(self, timestamp_ms: int) -> None:
        if self._active_channel is None or self._active_events is None:
            return
        self._finish_events(self._active_channel, self._active_events, timestamp_ms)
        self._active_channel = None
        self._active_events = None

    def _active_tool_call_is_keyed(self) -> bool:
        return any(
            self._active_events is events
            for events in self._tool_call_events_by_key.values()
        )

    def _active_unkeyed_tool_call_events(
        self,
    ) -> list[QueryPerformanceEvent] | None:
        if (
            self._active_channel != "tool_call"
            or self._active_events is None
            or self._active_tool_call_is_keyed()
        ):
            return None
        return self._active_events

    def _finish_open_tool_call_segments(self, timestamp_ms: int) -> None:
        active_unkeyed_events = self._active_unkeyed_tool_call_events()
        if active_unkeyed_events is not None:
            self._finish_events("tool_call", active_unkeyed_events, timestamp_ms)
        for events in self._tool_call_events_by_key.values():
            self._finish_events("tool_call", events, timestamp_ms)
        self._tool_call_events_by_key.clear()
        if self._active_channel == "tool_call":
            self._active_channel = None
            self._active_events = None

    def _finish_all_open_segments(self, timestamp_ms: int) -> None:
        self._finish_open_tool_call_segments(timestamp_ms)
        self._finish_active_segment(timestamp_ms)

    def _events_for_delta(
        self, channel: QueryPerformanceChannel, timestamp_ms: int
    ) -> list[QueryPerformanceEvent]:
        if channel == "tool_call" and self._active_tool_call_is_keyed():
            self._finish_open_tool_call_segments(timestamp_ms)
            return self._start_segment(channel, timestamp_ms)
        if self._active_channel != channel:
            self._finish_all_open_segments(timestamp_ms)
            return self._start_segment(channel, timestamp_ms)
        if self._active_events is None:
            return self._start_segment(channel, timestamp_ms)
        return self._active_events

    def _start_keyed_tool_call_segment(
        self, key: Hashable, timestamp_ms: int
    ) -> list[QueryPerformanceEvent]:
        if self._active_channel is not None and self._active_channel != "tool_call":
            self._finish_active_segment(timestamp_ms)
        elif self._active_unkeyed_tool_call_events() is not None:
            self._finish_active_segment(timestamp_ms)
        existing_events = self._tool_call_events_by_key.get(key)
        if existing_events is not None:
            self._finish_events("tool_call", existing_events, timestamp_ms)
        events = self._start_segment("tool_call", timestamp_ms)
        self._tool_call_events_by_key[key] = events
        return events

    def _events_for_tool_call_key(
        self, key: Hashable, timestamp_ms: int
    ) -> list[QueryPerformanceEvent]:
        events = self._tool_call_events_by_key.get(key)
        if events is not None:
            return events
        return self._start_keyed_tool_call_segment(key, timestamp_ms)

    def _events_for_tool_call(
        self, key: Hashable | None, timestamp_ms: int
    ) -> list[QueryPerformanceEvent]:
        if key is None:
            return self._events_for_delta("tool_call", timestamp_ms)
        return self._events_for_tool_call_key(key, timestamp_ms)

    def _text_delta_event(
        self,
        channel: QueryPerformanceChannel,
        timestamp_ms: int,
        start_char: int,
        end_char: int,
    ) -> QueryPerformanceEvent:
        if channel in self._text_offset_invalid_channels:
            return self._event(channel, "delta", timestamp_ms)
        return self._event(
            channel,
            "delta",
            timestamp_ms,
            channel_text_start_char=start_char,
            channel_text_end_char=end_char,
        )

    @_requires_unbuilt
    def append_reasoning_delta(self, text: str | None) -> Self:
        if not text:
            return self
        timestamp_ms = self._elapsed_ms()
        events = self._events_for_delta("reasoning", timestamp_ms)
        start_char = len(self._reasoning)
        end_char = start_char + len(text)
        events.append(
            self._text_delta_event("reasoning", timestamp_ms, start_char, end_char)
        )
        self._reasoning += text
        return self

    @_requires_unbuilt
    def append_content_delta(self, text: str | None) -> Self:
        if not text:
            return self
        timestamp_ms = self._elapsed_ms()
        events = self._events_for_delta("content", timestamp_ms)
        start_char = len(self._output_text)
        end_char = start_char + len(text)
        events.append(
            self._text_delta_event("content", timestamp_ms, start_char, end_char)
        )
        self._output_text += text
        return self

    @_requires_unbuilt
    def start_tool_call_segment(self, key: Hashable | None = None) -> Self:
        timestamp_ms = self._elapsed_ms()
        if key is None:
            self._finish_all_open_segments(timestamp_ms)
            self._start_segment("tool_call", timestamp_ms)
            return self
        self._start_keyed_tool_call_segment(key, timestamp_ms)
        return self

    @_requires_unbuilt
    def record_tool_call_delta(self, key: Hashable | None = None) -> Self:
        timestamp_ms = self._elapsed_ms()
        events = self._events_for_tool_call(key, timestamp_ms)
        events.append(self._event("tool_call", "delta", timestamp_ms))
        return self

    @_requires_unbuilt
    def record_tool_call_ready(self, key: Hashable | None = None) -> Self:
        timestamp_ms = self._elapsed_ms()
        events = self._events_for_tool_call(key, timestamp_ms)
        events.append(self._event("tool_call", "ready", timestamp_ms))
        return self

    @_requires_unbuilt
    def record_tool_call_progress(
        self,
        key: Hashable | None = None,
        *,
        ready: bool = False,
        delta: bool = False,
    ) -> bool:
        if not ready and not delta:
            return False
        timestamp_ms = self._elapsed_ms()
        events = self._events_for_tool_call(key, timestamp_ms)
        if ready and not any(event.type == "tool_call_ready" for event in events):
            events.append(self._event("tool_call", "ready", timestamp_ms))
        if delta:
            events.append(self._event("tool_call", "delta", timestamp_ms))
        return True

    @_requires_unbuilt
    def finish_current_segment(
        self, channel: QueryPerformanceChannel | None = None
    ) -> Self:
        timestamp_ms = self._elapsed_ms()
        if channel == "tool_call" or (
            channel is None and self._active_channel == "tool_call"
        ):
            self._finish_open_tool_call_segments(timestamp_ms)
            return self
        if channel is not None and self._active_channel != channel:
            return self
        self._finish_active_segment(timestamp_ms)
        return self

    @_requires_unbuilt
    def finish_tool_call_segment(self, key: Hashable | None) -> Self:
        if key is None:
            return self
        events = self._tool_call_events_by_key.pop(key, None)
        if events is None:
            return self
        self._finish_events("tool_call", events, self._elapsed_ms())
        if self._active_events is events:
            self._active_channel = None
            self._active_events = None
        return self

    def _clear_channel_text_offsets(self, channel: QueryPerformanceChannel) -> bool:
        saw_delta = False
        for entry_channel, _, events in self._timeline:
            if entry_channel != channel:
                continue
            for event in events:
                if event.type == f"{channel}_delta":
                    saw_delta = True
                event.channel_text_start_char = None
                event.channel_text_end_char = None
        return saw_delta

    @_requires_unbuilt
    def set_output_text(self, text: str | None) -> Self:
        output_text = text or ""
        if output_text != self._output_text and self._clear_channel_text_offsets(
            "content"
        ):
            self._text_offset_invalid_channels.add("content")
        self._output_text = output_text
        return self

    @_requires_unbuilt
    def set_reasoning(self, reasoning: str | None) -> Self:
        reasoning_text = reasoning or ""
        if reasoning_text != self._reasoning and self._clear_channel_text_offsets(
            "reasoning"
        ):
            self._text_offset_invalid_channels.add("reasoning")
        self._reasoning = reasoning_text
        return self

    def _performance(self) -> QueryResultPerformance | None:
        if not self._timeline:
            return None
        return QueryResultPerformance(
            timeline=[
                QueryPerformanceTimelineEntry(
                    channel=channel,
                    index=index,
                    events=events,
                )
                for channel, index, events in self._timeline
            ]
        )

    @_requires_unbuilt
    def build(
        self,
        *,
        finish_reason: FinishReasonInfo | None = None,
        metadata: QueryResultMetadata | None = None,
        tool_calls: list[ToolCall] | None = None,
        provider_tool_events: list[ProviderToolEvent] | None = None,
        history: list[InputItem] | None = None,
        extras: QueryResultExtras | None = None,
    ) -> QueryResult:
        from model_library.base.output.result import (
            FinishReason,
            FinishReasonInfo,
            QueryResult,
            QueryResultExtras,
            QueryResultMetadata,
        )

        self._built = True
        end_timestamp_ms = self._elapsed_ms()
        self._finish_all_open_segments(end_timestamp_ms)
        performance = self._performance()
        result_metadata = metadata or QueryResultMetadata()
        update = None
        if performance is not None:
            update = {"performance": performance}
        elif (
            result_metadata.performance is not None
            and not result_metadata.performance.timeline
        ):
            update = {"performance": None}
        result_metadata = result_metadata.model_copy(deep=True, update=update)
        result = QueryResult(
            output_text=self.output_text,
            reasoning=self.reasoning,
            finish_reason=finish_reason
            or FinishReasonInfo(reason=FinishReason.UNKNOWN, raw=None),
            metadata=result_metadata,
            tool_calls=tool_calls or [],
            provider_tool_events=provider_tool_events or [],
            history=history or [],
            extras=extras or QueryResultExtras(),
        )
        return result
