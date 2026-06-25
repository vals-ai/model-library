from typing import Protocol

from pydantic import ConfigDict, SkipValidation, field_validator, model_validator

from model_library.base.input import InputItem, RawResponse, ToolDefinition
from model_library.utils import ValsModel


DEFAULT_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""

DEFAULT_SUMMARY_PREFIX = "Another LLM agent started to solve this task and produced this summary of its progress. Use the information here to continue the work without duplicating effort.\n\n"


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


class TurnLimit(ValsModel):
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


class TimeLimit(ValsModel):
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


class HistoryCompaction(ValsModel):
    """Compaction *gating* config — strategy-agnostic.

    Tells the agent loop *when* to call the compaction hook and how many
    failures to tolerate. Strategy-specific knobs (the LLM prompt, summary
    prefix, etc.) belong on the hook factory itself
    (e.g. ``llm_summary_compactor(llm, prompt=..., summary_prefix=...)``)
    so custom strategies don't carry irrelevant fields.

    Specify exactly one of `threshold_tokens` or `threshold_percentage`. The
    default is 85% of the model's input context window.
    """

    threshold_tokens: int | None = None
    threshold_percentage: float | None = None
    max_failures: int = 2
    compact_on_max_context: bool = False
    max_compaction_context_retries: int = 2

    @field_validator("threshold_tokens", mode="before")
    @classmethod
    def _validate_threshold_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"threshold_tokens must be >= 1, got {v}")
        return v

    @field_validator("threshold_percentage", mode="before")
    @classmethod
    def _validate_threshold_percentage(cls, v: float | None) -> float | None:
        if v is not None and not 0 < v <= 0.90:
            raise ValueError(f"threshold_percentage must be in (0, 0.90], got {v}")
        return v

    @field_validator("max_failures", mode="before")
    @classmethod
    def _validate_max_failures(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_failures must be >= 1, got {v}")
        return v

    @field_validator("max_compaction_context_retries", mode="before")
    @classmethod
    def _validate_max_compaction_context_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_compaction_context_retries must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_one_threshold(self) -> "HistoryCompaction":
        if self.threshold_tokens is not None and self.threshold_percentage is not None:
            raise ValueError(
                "Specify exactly one of threshold_tokens or threshold_percentage."
            )
        if self.threshold_tokens is None and self.threshold_percentage is None:
            self.threshold_percentage = 0.85
        return self


class AgentConfig(ValsModel):
    """Configuration for agent execution

    - turn_limit: turn limit config, None = unlimited
    - time_limit: wall-clock deadline config, None = unlimited
    - max_tool_calls_per_turn: cap on executed tool calls per turn, None = unlimited.
      Calls beyond the limit get a skip message as their result (not executed).
    - history_compaction: optional summarization config for long agent histories.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_limit: TurnLimit | None
    time_limit: TimeLimit | None
    max_tool_calls_per_turn: int | None = None
    history_compaction: HistoryCompaction | None = None

    @field_validator("max_tool_calls_per_turn", mode="before")
    @classmethod
    def _validate_max_tool_calls(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_tool_calls_per_turn must be >= 1, got {v}")
        return v
