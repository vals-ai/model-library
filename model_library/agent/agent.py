import json
import logging
import time
import uuid
from collections.abc import Generator, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, computed_field, field_validator
from rich.pretty import pretty_repr

from model_library.agent.config import AgentConfig
from model_library.agent.hooks import AgentHooks, TurnResult
from model_library.agent.metadata import (
    AgentTurn,
    ErrorTurn,
    SerializableException,
    ToolCallRecord,
    TurnSummary,
)
from model_library.agent.tool import Tool, ToolOutput
from model_library.base.base import LLM
from model_library.base.input import InputItem, ToolCall, ToolResult
from model_library.base.output import QueryResultMetadata
from model_library.utils import PrettyModel, run_logging


class AgentStopReason(StrEnum):
    DONE_TOOL = "done_tool"
    SHOULD_STOP = "should_stop"
    MAX_TURNS = "max_turns"
    MAX_TIME = "max_time"
    ERROR = "error"


class AgentResult(PrettyModel):
    """Result of an agent run

    - final_answer: from determine_answer hook, done tool output, or LLM text
    - final_error: set on max turns/time, unhandled exceptions, or no answer
    - turns: TurnSummary for successful queries, ErrorTurn for failed ones
    - error_count: ErrorTurns + failed tool calls

    Durations (all wall clock, all rounded to ms):
    - final_duration_seconds: total run time
    - final_retry_overhead_seconds: sum of retry overhead across turns
    - final_effective_duration_seconds: wall clock minus retry overhead
    - final_aggregated_metadata.duration_seconds: sum of LLM query durations (excludes retries)

    The time budget (TimeLimit.max_seconds) is checked using wall clock minus retry overhead,
    so retry/backoff time does not count against the budget.
    """

    final_answer: str
    final_error: SerializableException | None = None
    turns: list[TurnSummary | ErrorTurn]
    final_duration_seconds: float  # wall clock, rounded to ms
    output_dir: Path = Field(exclude=True)

    @field_validator("final_duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def stop_reason(self) -> AgentStopReason:
        if self.final_error is not None:
            if self.final_error.type == "MaxTurnsExceeded":
                return AgentStopReason.MAX_TURNS
            if self.final_error.type == "MaxTimeExceeded":
                return AgentStopReason.MAX_TIME
            return AgentStopReason.ERROR
        if self.turns:
            last = self.turns[-1]
            if isinstance(last, TurnSummary) and any(tc.done for tc in last.tool_calls):
                return AgentStopReason.DONE_TOOL
        return AgentStopReason.SHOULD_STOP

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
                count += sum(1 for tc in turn.tool_calls if not tc.success)
        return count

    @computed_field
    @property
    def tool_calls_count(self) -> int:
        return sum(
            len(turn.tool_calls) for turn in self.turns if isinstance(turn, TurnSummary)
        )

    @computed_field
    @property
    def tool_usage(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for turn in self.turns:
            if isinstance(turn, TurnSummary):
                for tc in turn.tool_calls:
                    usage[tc.tool_name] = usage.get(tc.tool_name, 0) + 1
        return usage

    @computed_field
    @property
    def final_retry_overhead_seconds(self) -> float:
        return round(
            sum(
                turn.retry_overhead_seconds
                for turn in self.turns
                if isinstance(turn, TurnSummary)
            ),
            3,
        )

    @computed_field
    @property
    def final_effective_duration_seconds(self) -> float:
        """Wall clock minus retry overhead (final_duration_seconds - final_retry_overhead_seconds)"""
        return round(self.final_duration_seconds - self.final_retry_overhead_seconds, 3)

    @computed_field
    @property
    def final_aggregated_metadata(self) -> QueryResultMetadata:
        """Aggregated LLM query metadata across all turns (excludes tool sub-LLM calls)"""
        result = QueryResultMetadata()
        for turn in self.turns:
            if isinstance(turn, TurnSummary):
                result = result + turn.metadata
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
        name: str,
        log_dir: Path = Path("logs"),
        config: AgentConfig,
        hooks: AgentHooks | None = None,
    ):
        self._name = name
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._tool_defs = [tool.definition for tool in tools]

        self._log_dir = self._build_log_dir(log_dir, name, llm.model_name)
        self._config = config
        self._hooks = hooks or AgentHooks()

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        yield "name", self._name
        yield "llm", self._llm
        yield "tools", list(self._tools)
        yield "config", self._config
        yield "hooks", self._hooks

    def __repr__(self) -> str:
        return pretty_repr(self)

    __str__ = __repr__

    @staticmethod
    def _build_log_dir(base: Path, name: str, model_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = uuid.uuid4().hex[:6]
        return base / name / model_name.replace("/", "_") / f"{timestamp}_{short_id}"

    async def run(
        self,
        input: Sequence[InputItem],
        *,
        question_id: str,
        run_id: str | None = None,
        state: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> AgentResult:
        """Run the agent loop

        Each turn executes in this order:
        1. Check time limit and turn limit
        2. before_query hook (skipped on first turn)
        3. turn_limit.turn_message (appended to history)
        4. time_limit.time_message (appended to history)
        5. LLM query
        6. Tool execution
        7. should_stop hook

        The loop stops when any of these occur:
        - A tool returns done=True
        - should_stop hook returns True (default: text-only response)
        - turn_limit.max_turns reached (sets MaxTurnsExceeded error)
        - time_limit exceeded (sets MaxTimeExceeded error)
        - before_query hook re-raises a query error (default behavior)
        - Unhandled exception (sets error)

        After the loop, determine_answer hook runs with full context.
        Default returns None, falling back to done tool output or LLM text.
        """
        question_logger = (
            (logger or logging.getLogger("agent"))
            .getChild(f"{self._name}<{self._llm.model_name}>")
            .getChild(f"<question={question_id}>")
        )
        question_logger.setLevel(logging.DEBUG)
        with run_logging(question_logger, self._log_dir, question_id) as output_dir:
            question_logger.debug(repr(self))

            # run the loop
            return await self._run(
                input,
                state=state,
                question_id=question_id,
                run_id=run_id,
                output_dir=output_dir,
                logger=question_logger,
            )

    def _write_init_dir(
        self,
        output_dir: Path,
        input: Sequence[InputItem],
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        """Write init/ directory with config, initial state, and input."""
        init_dir = output_dir / "turns" / "init"
        init_dir.mkdir(parents=True, exist_ok=True)
        try:
            (init_dir / "config.json").write_text(
                json.dumps(
                    {
                        "llm": dict(self._llm.__rich_repr__()),
                        "agent_config": self._config.model_dump(),
                        "tools": [td.model_dump() for td in self._tool_defs],
                    },
                    indent=2,
                    default=str,
                )
            )
            (init_dir / "state.json").write_text(
                json.dumps(state, indent=2, default=str)
            )
            (init_dir / "history.bin").write_bytes(LLM.serialize_input(input))
        except Exception:
            logger.exception("Failed to write init directory")

    def _write_turn_dir(
        self,
        output_dir: Path,
        turn_number: int,
        turn: AgentTurn,
        state: dict[str, Any],
        history: list[InputItem],
        logger: logging.Logger,
    ) -> None:
        """Write turn directory with raw result, state snapshot, and history."""
        turn_dir = output_dir / "turns" / f"turn_{turn_number:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Exclude history and raw from JSON — history is saved as .bin,
            # raw contains non-serializable provider objects
            turn_data = turn.model_dump(exclude={"query_result": {"history", "raw"}})
            (turn_dir / "result.json").write_text(
                json.dumps(turn_data, indent=2, default=str)
            )
            (turn_dir / "state.json").write_text(
                json.dumps(state, indent=2, default=str)
            )
            (turn_dir / "history.bin").write_bytes(LLM.serialize_input(history))
        except Exception:
            logger.exception(f"Failed to write turn {turn_number} directory")

    def _write_error_turn_dir(
        self,
        output_dir: Path,
        turn_number: int,
        error_turn: ErrorTurn,
        logger: logging.Logger,
    ) -> None:
        """Write error turn directory with just the error."""
        turn_dir = output_dir / "turns" / f"turn_{turn_number:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        try:
            (turn_dir / "error.json").write_text(error_turn.model_dump_json(indent=2))
        except Exception:
            logger.exception(f"Failed to write error turn {turn_number} directory")

    async def _run(
        self,
        input: Sequence[InputItem],
        *,
        question_id: str,
        run_id: str | None = None,
        state: dict[str, Any] | None = None,
        output_dir: Path,
        logger: logging.Logger,
    ) -> AgentResult:
        if state is None:
            state = {}

        # track history so we can modify it
        history = list(input)

        # write init/ directory
        self._write_init_dir(output_dir, input, state, logger)

        # track turns (summaries for the result, raw turns for hooks)
        turns: list[TurnSummary | ErrorTurn] = []
        raw_turns: list[AgentTurn | ErrorTurn] = []

        start_time = time.monotonic()

        final_error: SerializableException | None = None
        last_query_error: Exception | None = None

        turn_limit = self._config.turn_limit
        time_limit = self._config.time_limit
        turn_number = 0
        retry_overhead = 0.0

        try:
            while turn_limit is None or turn_number < turn_limit.max_turns:
                turn_start = time.monotonic()
                turn_number += 1

                # check time limit
                elapsed = time.monotonic() - start_time
                effective_elapsed = (
                    elapsed
                    if (time_limit is not None and time_limit.include_retries)
                    else elapsed - retry_overhead
                )
                if (
                    time_limit is not None
                    and effective_elapsed >= time_limit.max_seconds
                ):
                    final_error = SerializableException(
                        type="MaxTimeExceeded",
                        message="Max time reached",
                        context={
                            "elapsed_seconds": elapsed,
                            "effective_elapsed_seconds": effective_elapsed,
                            "retry_overhead_seconds": retry_overhead,
                            "max_seconds": time_limit.max_seconds,
                        },
                    )
                    logger.warning(str(final_error))
                    break

                logger.info(
                    f"Turn {turn_number}/{turn_limit.max_turns if turn_limit else '?'} starting"
                )

                # hook: before_query (skip first turn, nothing to transform)
                if turn_number > 1:
                    if last_query_error is not None:
                        logger.debug(
                            f"before_query: handling error {type(last_query_error).__name__}: {last_query_error}"
                        )
                    new_history = self._hooks.before_query(history, last_query_error)
                    last_query_error = None
                    if new_history != history:
                        logger.debug(
                            f"before_query modified history: {len(history)} → {len(new_history)} items"
                        )
                    history = new_history

                # hooks: optional per-turn message injection
                if turn_limit is not None and turn_limit.turn_message is not None:
                    msg = turn_limit.turn_message(turn_number, turn_limit.max_turns)
                    if msg is not None:
                        logger.debug(
                            f"Hook turn_message({turn_number}/{turn_limit.max_turns}): {msg}"
                        )
                        history.append(msg)
                if time_limit is not None and time_limit.time_message is not None:
                    msg = time_limit.time_message(
                        effective_elapsed, time_limit.max_seconds
                    )
                    if msg is not None:
                        logger.debug(
                            f"Hook time_message({effective_elapsed:.3f}s/{time_limit.max_seconds}s): {msg}"
                        )
                        history.append(msg)

                # filter tools for this turn
                if turn_limit is not None and turn_limit.tool_filter is not None:
                    tools_for_turn = turn_limit.tool_filter(
                        turn_number, turn_limit.max_turns, self._tool_defs
                    )
                else:
                    tools_for_turn = self._tool_defs

                # query LLM
                try:
                    query_start = time.monotonic()
                    response = await self._llm.query(
                        input=history,
                        tools=tools_for_turn,
                        question_id=question_id,
                        run_id=run_id,
                        logger=logger,
                        in_agent=True,
                    )
                    query_wall = time.monotonic() - query_start
                    turn_retry_overhead = max(
                        0.0, query_wall - response.metadata.default_duration_seconds
                    )
                    retry_overhead += turn_retry_overhead
                except Exception as query_error:
                    logger.warning(f"Query failed: {query_error}", exc_info=True)
                    turn_duration = time.monotonic() - turn_start
                    error_turn = ErrorTurn(
                        error=SerializableException.from_exception(query_error),
                        duration_seconds=turn_duration,
                    )
                    turns.append(error_turn)
                    raw_turns.append(error_turn)

                    # Write error turn directory
                    self._write_error_turn_dir(
                        output_dir, turn_number, error_turn, logger
                    )

                    last_query_error = query_error
                    continue

                history: list[InputItem] = list(response.history)

                # process tool calls
                tool_call_records = await self._execute_tool_calls(
                    response.tool_calls, state, history, logger
                )

                turn_duration = time.monotonic() - turn_start

                logger.info(
                    f"Turn {turn_number}/{turn_limit.max_turns if turn_limit else '?'} | {len(tool_call_records)} tool calls"
                    + f" | in: {response.metadata.total_input_tokens}, out: {response.metadata.total_output_tokens}"
                    + f", cost:{response.metadata.cost.total if response.metadata.cost else '?'}"
                )
                for r in tool_call_records:
                    icon = "✓" if r.tool_output.error is None else "✗"
                    error_str = f" {r.tool_output.error}" if r.tool_output.error else ""
                    logger.info(
                        f"  {icon} {r.tool_call.name} ({r.duration_seconds}s){error_str}"
                    )

                turn = AgentTurn(
                    query_result=response,
                    tool_call_records=tool_call_records,
                    duration_seconds=turn_duration,
                    retry_overhead_seconds=turn_retry_overhead,
                )

                # Write full raw turn to disk, then convert to summary
                self._write_turn_dir(
                    output_dir, turn_number, turn, state, history, logger
                )
                turns.append(turn.to_summary())
                raw_turns.append(turn)

                done = any(r.tool_output.done for r in tool_call_records)
                if done:
                    logger.info("Stop: tool signaled done")
                    break

                # hook: should_stop (receives raw turn, not summary)
                elapsed = time.monotonic() - start_time
                turn_result = TurnResult(
                    turn_number=turn_number,
                    turn=turn,
                    state=state,
                    elapsed_seconds=elapsed,
                )
                should_stop = self._hooks.should_stop(turn_result)
                if should_stop:
                    logger.info("Stop: should_stop hook returned True")
                    break
            else:
                max_turns = turn_limit.max_turns if turn_limit else None
                final_error = SerializableException(
                    type="MaxTurnsExceeded",
                    message=f"Max turns ({max_turns}) reached",
                    context={"max_turns": max_turns},
                )
                logger.warning(str(final_error))
        except Exception as e:
            final_error = SerializableException.from_exception(e)
            logger.error(f"Agent loop failed: {final_error}", exc_info=True)

        # determine final answer (hooks get raw turns)
        elapsed = time.monotonic() - start_time
        answer = self._hooks.determine_answer(state, raw_turns, final_error)

        result = AgentResult(
            final_answer=answer,
            final_error=final_error,
            turns=turns,
            final_duration_seconds=elapsed,
            output_dir=output_dir,
        )
        logger.debug(f"Run complete: {result!r}")

        try:
            result_path = output_dir / "result.json"
            result_path.write_text(result.model_dump_json(indent=2))
        except Exception:
            logger.exception("Failed to serialize result")

        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        state: dict[str, Any],
        history: list[InputItem],
        logger: logging.Logger,
    ) -> list[ToolCallRecord]:
        """Execute tool calls, appending results to history

        Short-circuits on done — remaining tool calls in the batch are skipped.
        If max_tool_calls_per_turn is set, calls beyond the limit are not executed
        but still get a ToolResult appended (providers require results for all calls).
        """
        cap = self._config.max_tool_calls_per_turn
        records: list[ToolCallRecord] = []
        for i, tool_call in enumerate(tool_calls):
            if cap is not None and i >= cap:
                output = ToolOutput(output="Skipped: tool call limit exceeded")
                records.append(
                    ToolCallRecord(
                        tool_call=tool_call, tool_output=output, duration_seconds=0.0
                    )
                )
                history.append(ToolResult(tool_call=tool_call, result=output.output))
                continue

            record = await self._execute_tool(tool_call, state, logger)
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
        logger: logging.Logger,
    ) -> ToolCallRecord:
        """Execute a single tool call, return the record"""

        error: SerializableException | None = None
        start = time.monotonic()

        logger.debug(f"Tool call: {tool_call.name}({tool_call.parsed_args})")

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

            tool_output = await tool.execute(tool_call.parsed_args, state, logger)
        except Exception as e:
            logger.error(f"Tool '{tool_call.name}' raised: {e}", exc_info=True)
            tool_output = ToolOutput(output=str(e), error=str(e))
            error = SerializableException.from_exception(e, tool_name=tool_call.name)
        else:
            if tool_output.error:
                logger.warning(f"Tool '{tool_call.name}' failed: {tool_output.error}")
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

        logger.debug(f"Tool result: {record!r}")
        return record
