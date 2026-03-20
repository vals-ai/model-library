from dataclasses import dataclass
from typing import Protocol

from pydantic import ConfigDict, SkipValidation, field_validator

from model_library.base.input import InputItem, RawResponse
from model_library.utils import PrettyModel


def truncate_oldest(history: list[InputItem]) -> list[InputItem]:
    """Remove the oldest model response and associated inputs after it

    Always preserves the first message (initial prompt).
    Use with before_query hook for context window management:

        def before_query(history, last_error):
            if isinstance(last_error, MaxContextWindowExceededError):
                return truncate_oldest(history)
            if last_error:
                raise last_error
            return history
    """
    if len(history) <= 1:
        return history

    result = [history[0]]

    # skip RawResponse items (the first model response block)
    i = 1
    while i < len(history) and isinstance(history[i], RawResponse):
        i += 1

    # skip InputItems (ToolResults etc.) until next RawResponse or end
    while i < len(history) and not isinstance(history[i], RawResponse):
        i += 1

    # keep the rest
    result.extend(history[i:])
    return result


class TurnMessageHook(Protocol):
    """Called before each query to optionally inject a message into the history

    Receives the current turn number (1-indexed) and the max turns.
    Return an InputItem to append to history, or None to skip.
    """

    def __call__(self, turn_number: int, max_turns: int) -> InputItem | None: ...


@dataclass
class TurnLimit:
    """Turn limit for agent execution

    - max_turns: maximum loop iterations (includes ErrorTurns)
    - turn_message: optional hook to inject a message each turn (e.g. "turn 3/10")
    """

    max_turns: int
    turn_message: SkipValidation[TurnMessageHook | None] = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")


class TimeMessageHook(Protocol):
    """Called before each query to optionally inject a message into the history

    Receives elapsed seconds and the max seconds.
    Return an InputItem to append to history, or None to skip.
    """

    def __call__(
        self, elapsed_seconds: float, max_seconds: float
    ) -> InputItem | None: ...


@dataclass
class TimeLimit:
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

    max_seconds: float
    include_retries: bool = False
    time_message: SkipValidation[TimeMessageHook | None] = None

    def __post_init__(self) -> None:
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be > 0, got {self.max_seconds}")


class AgentConfig(PrettyModel):
    """Configuration for agent execution

    - turn_limit: turn limit config, None = unlimited
    - time_limit: wall-clock deadline config, None = unlimited
    - max_tool_calls_per_turn: cap on executed tool calls per turn, None = unlimited.
      Calls beyond the limit get a skip message as their result (not executed).
    - serialize_histories: save per-turn histories to disk via FileHandler path
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_limit: TurnLimit | None
    time_limit: TimeLimit | None
    max_tool_calls_per_turn: int | None = None
    serialize_histories: bool = True

    @field_validator("max_tool_calls_per_turn", mode="before")
    @classmethod
    def _validate_max_tool_calls(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_tool_calls_per_turn must be >= 1, got {v}")
        return v
