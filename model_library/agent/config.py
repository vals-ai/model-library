from typing import Protocol

from pydantic import ConfigDict, SkipValidation, field_validator

from model_library.base.input import InputItem, RawResponse, ToolDefinition
from model_library.utils import PrettyModel


def truncate_oldest(history: list[InputItem]) -> list[InputItem]:
    """Remove the oldest exchange (response + following inputs) from history

    Preserves SystemInput and the initial user input (everything before the
    first RawResponse). Raises ``ValueError`` if there are no exchanges to
    truncate.

    Use with before_query hook for context window management:

        def before_query(history, last_error):
            if isinstance(last_error, MaxContextWindowExceededError):
                return truncate_oldest(history)
            if last_error:
                raise last_error
            return history
    """
    # preserve everything before the first RawResponse (system prompt + user input)
    preamble: list[InputItem] = []
    i = 0
    while i < len(history) and not isinstance(history[i], RawResponse):
        preamble.append(history[i])
        i += 1

    rest = history[i:]
    if not rest:
        raise ValueError("No prior exchange found to truncate.")

    # skip RawResponse items (the oldest model response block)
    i = 0
    while i < len(rest) and isinstance(rest[i], RawResponse):
        i += 1

    # skip InputItems (ToolResults etc.) until next RawResponse or end
    while i < len(rest) and not isinstance(rest[i], RawResponse):
        i += 1

    return preamble + list(rest[i:])


class TurnMessageHook(Protocol):
    """Called before each query to optionally inject a message into the history

    Receives the current turn number (1-indexed) and the max turns.
    Return an InputItem to append to history, or None to skip.
    """

    def __call__(self, turn_number: int, max_turns: int) -> InputItem | None: ...


class ToolFilterHook(Protocol):
    """Called before each query to filter which tools are available for that turn

    Receives the current turn number (1-indexed), max turns, and all tool definitions.
    Return the subset of tools to make available for this turn.
    Excluded tools are not sent to the LLM (it won't see them).
    """

    def __call__(
        self, turn_number: int, max_turns: int, tools: list[ToolDefinition]
    ) -> list[ToolDefinition]: ...


class TurnLimit(PrettyModel):
    """Turn limit for agent execution

    - max_turns: maximum loop iterations (includes ErrorTurns)
    - turn_message: optional hook to inject a message each turn (e.g. "turn 3/10")
    - tool_filter: optional hook to filter tools per turn (e.g. last turn only allows submit)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_turns: int
    turn_message: SkipValidation[TurnMessageHook | None] = None
    tool_filter: SkipValidation[ToolFilterHook | None] = None

    @field_validator("max_turns", mode="before")
    @classmethod
    def _validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_turns must be >= 1, got {v}")
        return v


class TimeMessageHook(Protocol):
    """Called before each query to optionally inject a message into the history

    Receives elapsed seconds and the max seconds.
    Return an InputItem to append to history, or None to skip.
    """

    def __call__(
        self, elapsed_seconds: float, max_seconds: float
    ) -> InputItem | None: ...


class TimeLimit(PrettyModel):
    """Wall-clock time limit for agent execution

    - max_seconds: deadline in seconds
    - time_message: optional hook to inject a message each turn (e.g. "2min remaining")

    Time budget uses wall clock minus retry overhead (query_wall_time - query_duration
    per turn), so retry/backoff time is excluded from the budget.

    - include_retries: override to count retry/backoff time against the budget (strict wall clock)

    TODO: for real-time cancellation of in-flight queries, the retry layer
    should expose cumulative wait time (e.g. retrier.total_wait_seconds) or
    report wait intervals via a callback, so the agent can adjust the deadline
    dynamically without wall-clock drift.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_seconds: float
    include_retries: bool = False
    time_message: SkipValidation[TimeMessageHook | None] = None

    @field_validator("max_seconds", mode="before")
    @classmethod
    def _validate_max_seconds(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_seconds must be > 0, got {v}")
        return v


class AgentConfig(PrettyModel):
    """Configuration for agent execution

    - turn_limit: turn limit config, None = unlimited
    - time_limit: wall-clock deadline config, None = unlimited
    - max_tool_calls_per_turn: cap on executed tool calls per turn, None = unlimited.
      Calls beyond the limit get a skip message as their result (not executed).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_limit: TurnLimit | None
    time_limit: TimeLimit | None
    max_tool_calls_per_turn: int | None = None

    @field_validator("max_tool_calls_per_turn", mode="before")
    @classmethod
    def _validate_max_tool_calls(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_tool_calls_per_turn must be >= 1, got {v}")
        return v
