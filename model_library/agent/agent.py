import logging
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
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
from model_library.utils import PrettyModel, run_logging


class AgentResult(PrettyModel):
    """Result of an agent run

    - final_answer: from determine_answer hook, done tool output, or LLM text
    - final_error: set on max turns/time, unhandled exceptions, or no answer
    - turns: AgentTurn for successful queries, ErrorTurn for failed ones
    - state: mutable dict shared across the run, modified by tools and hooks
    - error_count: ErrorTurns + failed tool calls

    Durations (all wall clock, all rounded to ms):
    - final_duration_seconds: total run time (includes between-turn overhead)
    - final_turns_duration_seconds: sum of turn durations (derived from turns)
    - final_retry_overhead_seconds: sum of retry overhead across turns (derived from turns)
    - final_effective_duration_seconds: wall clock minus retry overhead (derived)
    - final_aggregated_metadata.duration_seconds: sum of LLM query durations (excludes retries)

    The time budget (TimeLimit.max_seconds) is checked using wall clock minus retry overhead,
    so retry/backoff time does not count against the budget.
    """

    final_answer: str
    final_error: SerializableException | None = None
    turns: list[AgentTurn | ErrorTurn]
    final_duration_seconds: float  # wall clock, rounded to ms
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
    def final_turns_duration_seconds(self) -> float:
        return round(sum(turn.duration_seconds for turn in self.turns), 3)

    @computed_field
    @property
    def final_retry_overhead_seconds(self) -> float:
        return round(
            sum(
                turn.retry_overhead_seconds
                for turn in self.turns
                if isinstance(turn, AgentTurn)
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
        name: str,
        log_dir: Path = Path("logs"),
        config: AgentConfig,
        hooks: AgentHooks | None = None,
        logger: logging.Logger | None = None,
    ):
        self._logger = (logger or logging.getLogger("agent")).getChild(
            f"{name}<{llm.model_name}>"
        )
        self._logger.setLevel(logging.DEBUG)

        self._llm = llm
        self._llm.logger = self._logger.getChild(f"<run={llm.run_id}>")
        self._tools = {tool.name: tool for tool in tools}
        self._tool_defs = [tool.definition for tool in tools]

        self._log_dir = self._build_log_dir(log_dir, name, llm.model_name)
        self._config = config
        self._hooks = hooks or AgentHooks()

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
        state: dict[str, Any] | None = None,
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
        with run_logging(self._logger, self._log_dir, question_id) as output_dir:
            # setup history serialization if needed
            histories_dir: Path | None = None
            if self._config.serialize_histories:
                if output_dir is not None:
                    histories_dir = output_dir / "histories"
                    histories_dir.mkdir(exist_ok=True)
                else:
                    self._logger.warning(
                        "serialize_histories enabled but no log_dir set, skipping"
                    )

            # log startup state
            custom_hooks = {
                name: getattr(self._hooks, name).__qualname__
                for name, field in AgentHooks.__dataclass_fields__.items()
                if getattr(self._hooks, name) is not field.default
            }
            tools_str = "\n".join(f"  {tool_def}" for tool_def in self._tool_defs)
            self._logger.debug(
                "Agent starting:\n"
                f"--- LLM\n{self._llm}\n"
                f"--- Tools ({len(self._tool_defs)}):\n{tools_str}\n"
                f"--- Config: {self._config}\n"
                f"--- Custom hooks: {custom_hooks or 'none'}\n"
            )

            # run the loop
            return await self._run(
                input,
                state=state,
                question_id=question_id,
                output_dir=output_dir,
                histories_dir=histories_dir,
            )

    async def _run(
        self,
        input: Sequence[InputItem],
        *,
        question_id: str,
        state: dict[str, Any] | None = None,
        output_dir: Path | None = None,
        histories_dir: Path | None = None,
    ) -> AgentResult:
        if state is None:
            state = {}

        # track history so we can modify it
        history = list(input)

        # track turns
        turns: list[AgentTurn | ErrorTurn] = []

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
                    self._logger.warning(str(final_error))
                    break

                # hook: before_query (skip first turn, nothing to transform)
                if turn_number > 1:
                    history = self._hooks.before_query(history, last_query_error)
                    last_query_error = None

                # hooks: optional per-turn message injection
                if turn_limit is not None and turn_limit.turn_message is not None:
                    msg = turn_limit.turn_message(turn_number, turn_limit.max_turns)
                    if msg is not None:
                        history.append(msg)
                if time_limit is not None and time_limit.time_message is not None:
                    msg = time_limit.time_message(
                        effective_elapsed, time_limit.max_seconds
                    )
                    if msg is not None:
                        history.append(msg)

                # query LLM
                try:
                    query_start = time.monotonic()
                    response = await self._llm.query(
                        input=history,
                        tools=self._tool_defs,
                        question_id=question_id,
                    )
                    query_wall = time.monotonic() - query_start
                    turn_retry_overhead = max(
                        0.0, query_wall - response.metadata.default_duration_seconds
                    )
                    retry_overhead += turn_retry_overhead
                except Exception as query_error:
                    self._logger.warning(f"Query failed: {query_error}", exc_info=True)
                    turn_duration = time.monotonic() - turn_start
                    turns.append(
                        ErrorTurn(
                            error=SerializableException.from_exception(query_error),
                            duration_seconds=turn_duration,
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

                turn_duration = time.monotonic() - turn_start
                turn = AgentTurn(
                    query_result=response,
                    tool_call_records=tool_call_records,
                    duration_seconds=turn_duration,
                    retry_overhead_seconds=turn_retry_overhead,
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
                max_turns = turn_limit.max_turns if turn_limit else None
                final_error = SerializableException(
                    type="MaxTurnsExceeded",
                    message=f"Max turns ({max_turns}) reached",
                    context={"max_turns": max_turns},
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

        if output_dir is not None:
            try:
                result_path = output_dir / "result.json"
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

        self._logger.debug(f"Tool call: {tool_call.name}({tool_call.parsed_args})")

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

        self._logger.debug(f"Tool result: {record!r}")
        return record
