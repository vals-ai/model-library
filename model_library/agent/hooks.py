from dataclasses import dataclass
from typing import Any, Protocol

from model_library.agent.metadata import (
    AgentTurn,
    ErrorTurn,
    SerializableException,
    ToolCallRecord,
)
from model_library.base.input import InputItem, ToolCall


@dataclass
class TurnResult:
    """Context passed to hooks after a turn completes

    turn_number is 1-indexed (first turn = 1).
    """

    turn_number: int
    turn: AgentTurn
    state: dict[str, Any]
    elapsed_seconds: float

    @property
    def response_text(self) -> str | None:
        return self.turn.query_result.output_text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.turn.query_result.tool_calls


class BeforeQueryHook(Protocol):
    """Called before each LLM query (skipped on first turn)

    Receives the last query error if the previous query failed.
    Must return the (possibly modified) history.
    Default: re-raises errors, passes history through unchanged.

    When handling errors, check the error type and only handle what you expect.
    Re-raise unhandled errors to stop the agent. Example:

        def before_query(history, last_error):
            if isinstance(last_error, MaxContextWindowExceededError):
                return truncate_oldest(history)
            if last_error:
                raise last_error
            return history
    """

    def __call__(
        self, history: list[InputItem], last_error: Exception | None
    ) -> list[InputItem]: ...


def default_before_query(
    history: list[InputItem], last_error: Exception | None
) -> list[InputItem]:
    """Re-raise query errors, otherwise pass history through unchanged"""

    if last_error:
        raise last_error
    return history


class ShouldStopHook(Protocol):
    """Called after each turn (both tool-call and text-only turns)

    Return True to stop the agent loop.
    Default: stops on text-only responses (no tool calls).
    """

    def __call__(self, turn_result: TurnResult) -> bool: ...


def default_should_stop(turn_result: TurnResult) -> bool:
    """Stop when the LLM responds with text only (no tool calls)"""

    return not turn_result.tool_calls


class OnToolResultHook(Protocol):
    """Called after each tool execution

    Use for logging, state updates, or side effects. Cannot control flow.
    """

    def __call__(self, record: ToolCallRecord, state: dict[str, Any]) -> None: ...


def default_on_tool_result(record: ToolCallRecord, state: dict[str, Any]) -> None:
    """No-op"""


class DetermineAnswerHook(Protocol):
    """Called after the loop ends (see Agent.run() for stop conditions)

    Return the final answer string.
    Default: done tool output, or LLM text, or empty string on error.
    """

    def __call__(
        self,
        state: dict[str, Any],
        turns: list[AgentTurn | ErrorTurn],
        final_error: SerializableException | None,
    ) -> str: ...


def default_determine_answer(
    state: dict[str, Any],
    turns: list[AgentTurn | ErrorTurn],
    final_error: SerializableException | None,
) -> str:
    """Done tool output → LLM text → empty string"""

    if final_error:
        return ""

    last_turn = turns[-1]
    if not turns or isinstance(last_turn, ErrorTurn):
        return ""

    # done tool output
    done_record = next(
        (r for r in last_turn.tool_call_records if r.tool_output.done), None
    )
    if done_record:
        return done_record.tool_output.output

    # LLM text
    return last_turn.query_result.output_text or ""


@dataclass
class AgentHooks:
    """Lifecycle hooks for customizing agent behavior"""

    before_query: BeforeQueryHook = default_before_query
    should_stop: ShouldStopHook = default_should_stop
    on_tool_result: OnToolResultHook = default_on_tool_result
    determine_answer: DetermineAnswerHook = default_determine_answer
