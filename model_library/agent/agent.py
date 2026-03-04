import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import computed_field, field_validator

from model_library.agent.config import AgentConfig
from model_library.agent.hooks import AgentHooks, TurnResult
from model_library.agent.metadata import (
    AgentTurn,
    ErrorTurn,
    SerializableException,
    ToolCallRecord,
)
from model_library.agent.tool import Tool, ToolOutput
from model_library.base.base import LLM
from model_library.base.input import InputItem, ToolCall, ToolResult
from model_library.base.output import QueryResultMetadata
from model_library.utils import PrettyModel, setup_history_dir


class AgentResult(PrettyModel):
    """Result of an agent run

    - final_answer: from determine_answer hook, done tool output, or LLM text
    - final_error: set on max turns/time, unhandled exceptions, or no answer
    - turns: AgentTurn for successful queries, ErrorTurn for failed ones
    - state: mutable dict shared across the run, modified by tools and hooks
    - error_count: ErrorTurns + failed tool calls

    Durations at three levels:
    - final_duration_seconds: wall-clock time of the entire run
    - final_aggregated_metadata.duration_seconds: sum of LLM query durations
    - turns[i].tool_call_records[j].duration_seconds: individual tool call duration
    """

    final_answer: str
    final_error: SerializableException | None = None
    turns: list[AgentTurn | ErrorTurn]
    final_duration_seconds: float  # rounded to ms
    state: dict[str, Any]

    @field_validator("final_duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def success(self) -> bool:
        return self.final_error is None

    @computed_field
    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @computed_field
    @property
    def error_count(self) -> int:
        count = 0
        for turn in self.turns:
            if isinstance(turn, ErrorTurn):
                count += 1
            else:
                count += sum(1 for tc in turn.tool_call_records if not tc.success)
        return count

    @computed_field
    @property
    def tool_calls_count(self) -> int:
        return sum(
            len(turn.tool_call_records)
            for turn in self.turns
            if isinstance(turn, AgentTurn)
        )

    @computed_field
    @property
    def tool_usage(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for turn in self.turns:
            if isinstance(turn, AgentTurn):
                for tc in turn.tool_call_records:
                    usage[tc.tool_call.name] = usage.get(tc.tool_call.name, 0) + 1
        return usage

    @computed_field
    @property
    def final_aggregated_metadata(self) -> QueryResultMetadata:
        """Aggregated token/cost/duration metadata across all turns"""
        result = QueryResultMetadata()
        for turn in self.turns:
            if isinstance(turn, AgentTurn):
                result = result + turn.combined_metadata
        return result


class Agent:
    """
    Composable agent that runs a tool-augmented conversation loop
    Returns AgentResult with per-turn data and final results
    """

    def __init__(
        self,
        llm: LLM,
        tools: Sequence[Tool],
        *,
        logger: logging.Logger,
        config: AgentConfig | None = None,
        hooks: AgentHooks | None = None,
    ):
        self._llm = llm
        self._llm.logger = logger.getChild(llm.logger.name)
        self._tools = {tool.name: tool for tool in tools}
        self._tool_defs = [tool.definition for tool in tools]

        self._logger = logger
        self._config = config or AgentConfig()
        self._hooks = hooks or AgentHooks()

    async def run(
        self,
        input: Sequence[InputItem],
        *,
        state: dict[str, Any] | None = None,
        question_id: str | None = None,
    ) -> AgentResult:
        """Run the agent loop

        The loop stops when any of these occur:
        - A tool returns done=True
        - should_stop hook returns True (default: text-only response)
        - max_turns reached (sets MaxTurnsExceeded error)
        - max_time_seconds exceeded (sets MaxTimeExceeded error)
        - before_query hook re-raises a query error (default behavior)
        - Unhandled exception (sets error)

        After the loop, determine_answer hook runs with full context.
        Default returns None, falling back to done tool output or LLM text.
        """
        # Setup history serialization before any logging (may redirect FileHandler)
        histories_dir: Path | None = None
        if self._config.serialize_histories:
            histories_dir = setup_history_dir(self._logger)

        custom_hooks = {
            name: getattr(self._hooks, name).__qualname__
            for name, field in AgentHooks.__dataclass_fields__.items()
            if getattr(self._hooks, name) is not field.default
        }
        tools_str = "\n".join(f"  {tool_def}" for tool_def in self._tool_defs)
        self._logger.info(
            "Agent starting:\n"
            f"--- LLM\n{self._llm}\n"
            f"--- Tools ({len(self._tool_defs)}):\n{tools_str}\n"
            f"--- Config: {self._config}\n"
            f"--- Custom hooks: {custom_hooks or 'none'}\n"
        )

        if state is None:
            state = {}

        # track history so we can modify it
        history = list(input)

        # track turns
        turns: list[AgentTurn | ErrorTurn] = []

        start_time = time.monotonic()

        final_error: SerializableException | None = None
        last_query_error: Exception | None = None

        try:
            for turn_number in range(self._config.max_turns):
                # check if we have exceeded the max time
                elapsed = time.monotonic() - start_time
                if elapsed >= self._config.max_time_seconds:
                    final_error = SerializableException(
                        type="MaxTimeExceeded",
                        message="Max time reached",
                        context={
                            "elapsed_seconds": elapsed,
                            "max_time_seconds": self._config.max_time_seconds,
                        },
                    )
                    self._logger.warning(str(final_error))
                    break

                # hook: before_query (skip first turn, nothing to transform)
                if turn_number > 0:
                    history = self._hooks.before_query(history, last_query_error)
                    last_query_error = None

                # query LLM
                try:
                    response = await self._llm.query(
                        input=history, tools=self._tool_defs, question_id=question_id
                    )
                except Exception as query_error:
                    self._logger.warning(f"Query failed: {query_error}", exc_info=True)
                    turns.append(
                        ErrorTurn(
                            error=SerializableException.from_exception(query_error)
                        )
                    )
                    last_query_error = query_error
                    continue

                history = list(response.history)

                self._logger.info(
                    f"Turn {turn_number}: {len(response.tool_calls)} tool calls, "
                    f"response={'yes' if response.output_text else 'no'}, "
                    f"tokens={response.metadata.total_input_tokens}in/{response.metadata.total_output_tokens}out"
                )

                # process tool calls
                tool_call_records = await self._execute_tool_calls(
                    response.tool_calls, state, history
                )

                # Serialize turn history before clearing
                if histories_dir is not None:
                    try:
                        path = histories_dir / f"turn_{turn_number:03d}.bin"
                        path.write_bytes(LLM.serialize_input(history))
                    except Exception:
                        self._logger.exception(
                            f"Failed to serialize history for turn {turn_number}"
                        )

                # Clear history and raw from the stored response to avoid
                # keeping redundant copies (the agent maintains its own history)
                response.history = []
                response.raw = {}

                turn = AgentTurn(
                    query_result=response,
                    tool_call_records=tool_call_records,
                )
                turns.append(turn)

                done = any(r.tool_output.done for r in tool_call_records)
                if done:
                    self._logger.info("Stop: tool signaled done")
                    break

                # hook: should_stop
                elapsed = time.monotonic() - start_time
                turn_result = TurnResult(
                    turn_number=turn_number,
                    turn=turn,
                    state=state,
                    elapsed_seconds=elapsed,
                )
                if self._hooks.should_stop(turn_result):
                    self._logger.info("Stop: should_stop hook returned True")
                    break
            else:
                final_error = SerializableException(
                    type="MaxTurnsExceeded",
                    message=f"Max turns ({self._config.max_turns}) reached",
                    context={"max_turns": self._config.max_turns},
                )
                self._logger.warning(str(final_error))
        except Exception as e:
            final_error = SerializableException.from_exception(e)
            self._logger.error(f"Agent loop failed: {final_error}", exc_info=True)

        # determine final answer
        elapsed = time.monotonic() - start_time
        answer = self._hooks.determine_answer(state, turns, final_error)

        result = AgentResult(
            final_answer=answer,
            final_error=final_error,
            turns=turns,
            final_duration_seconds=elapsed,
            state=state,
        )
        self._logger.info(f"Run complete: {result!r}")

        if histories_dir is not None:
            try:
                result_path = histories_dir.parent / "result.json"
                result_path.write_text(result.model_dump_json(indent=2))
            except Exception:
                self._logger.exception("Failed to serialize result")

        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        state: dict[str, Any],
        history: list[InputItem],
    ) -> list[ToolCallRecord]:
        """Execute tool calls, appending results to history

        Short-circuits on done — remaining tool calls in the batch are skipped.
        """
        records: list[ToolCallRecord] = []
        for tool_call in tool_calls:
            record = await self._execute_tool(tool_call, state)
            records.append(record)
            history.append(
                ToolResult(tool_call=record.tool_call, result=record.tool_output.output)
            )

            self._hooks.on_tool_result(record, state)

            if record.tool_output.done:
                break

        return records

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        state: dict[str, Any],
    ) -> ToolCallRecord:
        """Execute a single tool call, return the record"""

        error: SerializableException | None = None
        start = time.monotonic()

        self._logger.info(f"Tool call: {tool_call.name}({tool_call.parsed_args})")

        try:
            tool = self._tools.get(tool_call.name)
            if not tool:
                raise ValueError(
                    f"Tool '{tool_call.name}' not found. Available: {list(self._tools)}"
                )

            if tool_call.parsed_args is None:
                raise ValueError(f"Unparseable tool call args: {tool_call.args!r}")

            missing = [r for r in tool.required if r not in tool_call.parsed_args]
            if missing:
                raise ValueError(
                    f"Missing required parameters for '{tool_call.name}': {missing}"
                )

            tool_output = await tool.execute(tool_call.parsed_args, state, self._logger)
        except Exception as e:
            self._logger.error(f"Tool '{tool_call.name}' raised: {e}", exc_info=True)
            tool_output = ToolOutput(output=str(e), error=str(e))
            error = SerializableException.from_exception(e, tool_name=tool_call.name)
        else:
            if tool_output.error:
                self._logger.warning(
                    f"Tool '{tool_call.name}' failed: {tool_output.error}"
                )
                error = SerializableException(
                    type="ToolFailed",
                    message=tool_output.error,
                    context={"tool_name": tool_call.name},
                )

        duration = time.monotonic() - start
        record = ToolCallRecord(
            tool_call=tool_call,
            tool_output=tool_output,
            duration_seconds=duration,
            error=error,
        )

        self._logger.info(f"Tool result: {record!r}")
        return record
