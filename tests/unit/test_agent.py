"""Unit tests for model_library.agent"""

import logging
import time as time_module
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base.input import InputItem, RawResponse, TextInput, ToolCall
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentResult,
    AgentTurn,
    ErrorTurn,
    SerializableException,
    Tool,
    ToolCallRecord,
    ToolOutput,
    TurnResult,
    truncate_oldest,
)


_logger = logging.getLogger("test_agent")


# --- Helpers ---


def make_metadata(in_tokens: int = 10, out_tokens: int = 5, cost_total: float = 0.01) -> QueryResultMetadata:
    return QueryResultMetadata(
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost=QueryResultCost(input=cost_total / 2, output=cost_total / 2),
    )


def make_text_response(text: str, metadata: QueryResultMetadata | None = None) -> QueryResult:
    """LLM response with text output and no tool calls"""
    return QueryResult(
        output_text=text,
        metadata=metadata or make_metadata(),
        tool_calls=[],
        history=[TextInput(text="prompt")],
    )


def make_tool_response(
    tool_calls: list[ToolCall],
    metadata: QueryResultMetadata | None = None,
    output_text: str | None = None,
) -> QueryResult:
    """LLM response with tool calls"""
    return QueryResult(
        output_text=output_text,
        metadata=metadata or make_metadata(),
        tool_calls=tool_calls,
        history=[TextInput(text="prompt")],
    )


def make_tool_call(name: str = "echo", args: dict[str, Any] | str | None = None) -> ToolCall:
    return ToolCall(id="tc_1", name=name, args=args or {"text": "hello"})


def mock_llm(*responses: QueryResult | Exception) -> MagicMock:
    """Create a mock LLM that returns the given responses in sequence"""
    llm = MagicMock()
    llm.query = AsyncMock(side_effect=list(responses))
    llm.logger = logging.getLogger("mock_llm")
    return llm


# --- Tool implementations ---


class EchoTool(Tool):
    name = "echo"
    description = "Echo the input"
    parameters = {"text": {"type": "string"}}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        return ToolOutput(output=args.get("text", ""))


class StateTool(Tool):
    name = "set_state"
    description = "Set a state value"
    parameters = {"key": {"type": "string"}, "value": {"type": "string"}}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        state[args["key"]] = args["value"]
        return ToolOutput(output="ok")


class DoneTool(Tool):
    name = "submit"
    description = "Submit final answer"
    parameters = {"answer": {"type": "string"}}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        return ToolOutput(output=args["answer"], done=True)


class FailingTool(Tool):
    name = "fail"
    description = "Always fails"
    parameters: dict[str, Any] = {}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        raise RuntimeError("tool broke")


class ErrorReturnTool(Tool):
    """Tool that returns an error via ToolOutput (doesn't raise)"""

    name = "soft_fail"
    description = "Returns error"
    parameters: dict[str, Any] = {}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        return ToolOutput(output="error details", error="something went wrong")


class LLMCallingTool(Tool):
    name = "retrieve"
    description = "Retrieve info"
    parameters = {"q": {"type": "string"}}

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        sub_metadata = QueryResultMetadata(
            in_tokens=50, out_tokens=20, cost=QueryResultCost(input=0.005, output=0.003)
        )
        return ToolOutput(output="retrieved info", metadata=sub_metadata)


# --- Agent loop ---


class TestAgentBasicLoop:
    async def test_single_turn_text_response(self):
        llm = mock_llm(make_text_response("hello world"))
        agent = Agent(logger=_logger, llm=llm, tools=[])

        result = await agent.run([TextInput(text="say hello")])

        assert result.final_answer == "hello world"
        assert result.success
        assert result.final_error is None
        assert result.total_turns == 1
        assert result.final_aggregated_metadata.in_tokens == 10
        assert result.final_aggregated_metadata.out_tokens == 5

    async def test_tool_call_then_text_response(self):
        tc = make_tool_call("echo", {"text": "ping"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("got: ping"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="echo test")])

        assert result.final_answer == "got: ping"
        assert result.success
        assert llm.query.call_count == 2
        assert result.total_turns == 2
        assert result.tool_calls_count == 1
        assert result.tool_usage == {"echo": 1}

    async def test_done_tool_stops_loop(self):
        tc = make_tool_call("submit", {"answer": "42"})
        llm = mock_llm(make_tool_response([tc]))
        agent = Agent(logger=_logger, llm=llm, tools=[DoneTool()])

        result = await agent.run([TextInput(text="answer?")])

        assert result.final_answer == "42"
        assert result.success
        assert llm.query.call_count == 1

    async def test_max_turns_sets_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()], config=AgentConfig(max_turns=3))

        result = await agent.run([TextInput(text="go")])

        assert not result.success
        assert result.final_error is not None
        assert result.final_error.type == "MaxTurnsExceeded"
        assert result.final_answer == ""
        assert llm.query.call_count == 3
        assert result.total_turns == 3

    async def test_max_time_sets_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)

        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            # 0 for first iteration, 100 for second iteration's time check
            return 0.0 if call_count[0] < 10 else 100.0

        with patch.object(time_module, "monotonic", side_effect=fake_monotonic):
            agent = Agent(
                logger=_logger, llm=llm, tools=[EchoTool()], config=AgentConfig(max_time_seconds=50)
            )
            result = await agent.run([TextInput(text="go")])

        assert not result.success
        assert result.final_error is not None
        assert result.final_error.type == "MaxTimeExceeded"

    async def test_text_only_auto_stops_without_should_stop(self):
        llm = mock_llm(make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[], config=AgentConfig(max_turns=10))

        result = await agent.run([TextInput(text="go")])

        assert result.final_answer == "done"
        assert llm.query.call_count == 1

    async def test_input_passed_to_llm(self):
        llm = mock_llm(make_text_response("ok"))
        agent = Agent(logger=_logger, llm=llm, tools=[])

        await agent.run([TextInput(text="a"), TextInput(text="b")])

        messages = llm.query.call_args.kwargs["input"]
        assert len(messages) == 2

    async def test_state_defaults_to_empty_dict(self):
        llm = mock_llm(make_text_response("ok"))
        agent = Agent(logger=_logger, llm=llm, tools=[])

        result = await agent.run([TextInput(text="go")])

        assert result.state == {}

    async def test_initial_state_preserved(self):
        llm = mock_llm(make_text_response("ok"))
        agent = Agent(logger=_logger, llm=llm, tools=[])
        state = {"initial": True}

        result = await agent.run([TextInput(text="go")], state=state)

        assert result.state == state
        assert result.state["initial"] is True


# --- Tool execution ---


class TestAgentToolExecution:
    async def test_unknown_tool_records_error(self):
        tc = make_tool_call("nonexistent", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("ok"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="try unknown")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.error is not None
        assert "not found" in record.error.message

    async def test_tool_exception_caught_and_recorded(self):
        tc = make_tool_call("fail", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = Agent(logger=_logger, llm=llm, tools=[FailingTool()])

        result = await agent.run([TextInput(text="try failing")])

        assert result.final_answer == "recovered"
        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.error is not None
        assert "tool broke" in record.error.message
        assert record.error.traceback is not None

    async def test_tool_error_return_recorded(self):
        """Tool returning ToolOutput with error set (doesn't raise)"""
        tc = make_tool_call("soft_fail", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("ok"))
        agent = Agent(logger=_logger, llm=llm, tools=[ErrorReturnTool()])

        result = await agent.run([TextInput(text="try soft fail")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.error is not None
        assert record.error.type == "ToolFailed"
        assert record.error.message == "something went wrong"
        assert record.tool_output.output == "error details"

    async def test_string_args_parsed_as_json(self):
        tc = ToolCall(id="tc_1", name="echo", args='{"text": "from json"}')
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        assert result.turns[0].tool_call_records[0].tool_call.parsed_args == {"text": "from json"}

    async def test_invalid_json_args_errors(self):
        tc = ToolCall(id="tc_1", name="echo", args="not json")
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.tool_call.parsed_args is None
        assert "Unparseable" in record.tool_output.output

    async def test_json_args_non_dict_errors(self):
        """JSON that parses to non-dict (e.g. list) is unparseable"""
        tc = ToolCall(id="tc_1", name="echo", args="[1, 2, 3]")
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.tool_call.parsed_args is None
        assert "Unparseable" in record.tool_output.output

    async def test_missing_required_param_records_error(self):
        """Tool with required params called without them → error record"""
        tc = ToolCall(id="tc_1", name="echo", args={})  # missing "text"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert record.error is not None
        assert "Missing required parameters" in record.error.message
        assert "text" in record.error.message

    async def test_missing_multiple_required_params(self):
        tc = ToolCall(id="tc_1", name="set_state", args={})  # missing "key" and "value"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = Agent(logger=_logger, llm=llm, tools=[StateTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert "key" in record.error.message
        assert "value" in record.error.message

    async def test_partial_required_params_reports_missing(self):
        tc = make_tool_call("set_state", {"key": "k"})  # has "key", missing "value"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = Agent(logger=_logger, llm=llm, tools=[StateTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert not record.success
        assert "value" in record.error.message
        assert "key" not in record.error.message  # "key" was provided

    async def test_optional_params_not_required(self):
        """Tool with explicit required subset — extra params are optional"""

        class OptionalTool(Tool):
            name = "opt"
            description = "Has optional params"
            parameters = {"needed": {"type": "string"}, "optional": {"type": "string"}}
            required = ["needed"]

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output=args["needed"])

        tc = make_tool_call("opt", {"needed": "yes"})  # "optional" omitted
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[OptionalTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert record.success
        assert record.tool_output.output == "yes"

    async def test_no_params_tool_accepts_empty_args(self):
        tc = ToolCall(id="tc_1", name="fail", args={})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        # FailingTool has parameters={}, required=[] — should pass validation
        # (it will fail in execute, but that's a different error)
        agent = Agent(logger=_logger, llm=llm, tools=[FailingTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        # Fails from execute(), not from validation
        assert "tool broke" in record.error.message

    async def test_state_shared_between_tools(self):
        set_call = make_tool_call("set_state", {"key": "found", "value": "yes"})
        echo_call = make_tool_call("echo", {"text": "check"})
        llm = mock_llm(make_tool_response([set_call]), make_tool_response([echo_call]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[StateTool(), EchoTool()])
        state: dict[str, Any] = {}

        result = await agent.run([TextInput(text="test")], state=state)

        assert state["found"] == "yes"
        assert result.state["found"] == "yes"

    async def test_tool_call_duration_tracked(self):
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], AgentTurn)
        record = result.turns[0].tool_call_records[0]
        assert isinstance(record.duration_seconds, float)
        assert record.duration_seconds >= 0

    async def test_done_tool_skips_remaining_tool_calls(self):
        done_call = make_tool_call("submit", {"answer": "42"})
        echo_call = make_tool_call("echo", {"text": "should not run"})
        llm = mock_llm(make_tool_response([done_call, echo_call]))
        agent = Agent(logger=_logger, llm=llm, tools=[DoneTool(), EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert result.final_answer == "42"
        assert isinstance(result.turns[0], AgentTurn)
        assert len(result.turns[0].tool_call_records) == 1
        assert result.turns[0].tool_call_records[0].tool_call.name == "submit"


# --- Tool metadata ---


class TestAgentToolMetadata:
    async def test_tool_metadata_aggregated_in_turn(self):
        tc = make_tool_call("retrieve", {"q": "test"})
        llm = mock_llm(
            make_tool_response([tc], metadata=make_metadata(in_tokens=100, out_tokens=50)),
            make_text_response("done"),
        )
        agent = Agent(logger=_logger, llm=llm, tools=[LLMCallingTool()])

        result = await agent.run([TextInput(text="retrieve")])

        assert isinstance(result.turns[0], AgentTurn)
        turn = result.turns[0]
        # Query: 100 in + 50 out, Tool sub-LLM: 50 in + 20 out
        assert turn.combined_metadata.in_tokens == 150
        assert turn.combined_metadata.out_tokens == 70

    async def test_metadata_aggregates_across_turns(self):
        tc = make_tool_call("echo", {"text": "hi"})
        meta = make_metadata(in_tokens=100, out_tokens=50, cost_total=0.02)
        llm = mock_llm(make_tool_response([tc], metadata=meta), make_text_response("done", metadata=meta))
        agent = Agent(logger=_logger, llm=llm, tools=[EchoTool()])

        result = await agent.run([TextInput(text="test")])

        assert result.final_aggregated_metadata.in_tokens == 200
        assert result.final_aggregated_metadata.out_tokens == 100
        assert result.total_turns == 2


# --- Hooks ---


class TestAgentHooks:
    async def test_should_stop_controls_loop(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(10)]
        llm = mock_llm(*responses)

        def stop_at_3(turn_result: TurnResult) -> bool:
            return turn_result.turn_number >= 2

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[EchoTool()],
            hooks=AgentHooks(should_stop=stop_at_3),
            config=AgentConfig(max_turns=10),
        )

        result = await agent.run([TextInput(text="go")])

        assert llm.query.call_count == 3
        assert result.total_turns == 3

    async def test_should_stop_on_text_only_turn(self):
        """With should_stop, text-only responses don't auto-stop; hook decides"""
        responses = [
            make_text_response("not done yet"),
            make_text_response("still going"),
            make_text_response("EXIT"),
        ]
        llm = mock_llm(*responses)

        def stop_on_exit(turn_result: TurnResult) -> bool:
            return turn_result.response_text is not None and "EXIT" in turn_result.response_text

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[],
            hooks=AgentHooks(should_stop=stop_on_exit),
            config=AgentConfig(max_turns=10),
        )

        result = await agent.run([TextInput(text="go")])

        assert llm.query.call_count == 3
        assert result.final_answer == "EXIT"

    async def test_determine_answer_overrides_default(self):
        llm = mock_llm(make_text_response("raw"))

        def custom_answer(
            state: dict[str, Any],
            turns: list[AgentTurn | ErrorTurn],
            final_error: SerializableException | None,
        ) -> str:
            return "overridden"

        agent = Agent(logger=_logger, llm=llm, tools=[], hooks=AgentHooks(determine_answer=custom_answer))

        result = await agent.run([TextInput(text="test")])

        assert result.final_answer == "overridden"

    async def test_determine_answer_uses_state(self):
        tc = make_tool_call("set_state", {"key": "score", "value": "100"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        def answer_from_state(
            state: dict[str, Any],
            turns: list[AgentTurn | ErrorTurn],
            final_error: SerializableException | None,
        ) -> str:
            return f"score={state.get('score', 'unknown')}"

        agent = Agent(
            logger=_logger, llm=llm, tools=[StateTool()], hooks=AgentHooks(determine_answer=answer_from_state)
        )

        result = await agent.run([TextInput(text="test")])

        assert result.final_answer == "score=100"

    async def test_determine_answer_called_on_error(self):
        """determine_answer receives final_error and can salvage an answer"""
        tc = make_tool_call("set_state", {"key": "partial", "value": "yes"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)

        def salvage(
            state: dict[str, Any],
            turns: list[AgentTurn | ErrorTurn],
            final_error: SerializableException | None,
        ) -> str:
            if state.get("partial"):
                return "salvaged"
            return ""

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[StateTool()],
            config=AgentConfig(max_turns=3),
            hooks=AgentHooks(determine_answer=salvage),
        )

        result = await agent.run([TextInput(text="go")])

        assert result.final_answer == "salvaged"
        assert result.final_error is not None
        assert result.final_error.type == "MaxTurnsExceeded"
        assert not result.success

    async def test_default_determine_answer_returns_text(self):
        llm = mock_llm(make_text_response("raw"))
        agent = Agent(logger=_logger, llm=llm, tools=[])

        result = await agent.run([TextInput(text="test")])

        assert result.success
        assert result.final_answer == "raw"

    async def test_default_determine_answer_returns_empty_on_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[EchoTool()],
            config=AgentConfig(max_turns=2),
        )

        result = await agent.run([TextInput(text="go")])

        assert result.final_error is not None
        assert result.final_error.type == "MaxTurnsExceeded"
        assert result.final_answer == ""

    async def test_on_tool_result_called_for_each_tool(self):
        tc1 = make_tool_call("echo", {"text": "a"})
        tc2 = make_tool_call("echo", {"text": "b"})
        llm = mock_llm(make_tool_response([tc1, tc2]), make_text_response("done"))

        records: list[ToolCallRecord] = []

        def on_result(record: ToolCallRecord, state: dict) -> None:
            records.append(record)

        agent = Agent(
            logger=_logger, llm=llm, tools=[EchoTool()], hooks=AgentHooks(on_tool_result=on_result)
        )

        await agent.run([TextInput(text="test")])

        assert len(records) == 2
        assert records[0].tool_call.name == "echo"
        assert records[0].tool_output.output == "a"
        assert records[1].tool_output.output == "b"

    async def test_before_query_handles_error(self):
        """before_query receives last_query_error and can recover"""
        errors_seen: list[Exception | None] = []

        def handle_error(history: list[InputItem], error: Exception | None) -> list[InputItem]:
            errors_seen.append(error)
            return history

        llm = mock_llm(RuntimeError("query failed"), make_text_response("recovered"))

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[],
            config=AgentConfig(max_turns=5),
            hooks=AgentHooks(before_query=handle_error),
        )

        result = await agent.run([TextInput(text="test")])

        assert result.final_answer == "recovered"
        assert result.success
        assert len(result.turns) == 2
        assert isinstance(result.turns[0], ErrorTurn)
        assert isinstance(result.turns[1], AgentTurn)
        assert len(errors_seen) == 1
        assert isinstance(errors_seen[0], RuntimeError)

    async def test_before_query_absent_with_error_stops(self):
        """Without before_query, query error is re-raised and sets final_error"""
        llm = mock_llm(RuntimeError("query failed"), make_text_response("unreachable"))

        agent = Agent(logger=_logger, llm=llm, tools=[], config=AgentConfig(max_turns=5))

        result = await agent.run([TextInput(text="test")])

        assert not result.success
        assert result.final_error is not None
        assert "query failed" in result.final_error.message
        assert result.final_answer == ""
        assert len(result.turns) == 1
        assert isinstance(result.turns[0], ErrorTurn)


# --- Error handling ---


class TestErrorHandling:
    async def test_query_exception_creates_error_turn(self):
        llm = mock_llm(ValueError("bad input"), make_text_response("unreachable"))

        agent = Agent(logger=_logger, llm=llm, tools=[], config=AgentConfig(max_turns=5))

        result = await agent.run([TextInput(text="test")])

        assert isinstance(result.turns[0], ErrorTurn)
        assert result.turns[0].error.type == "ValueError"
        assert result.turns[0].error.message == "bad input"

    async def test_error_count_includes_error_turns_and_failed_tools(self):
        tc = make_tool_call("fail", {})
        llm = mock_llm(RuntimeError("query failed"), make_tool_response([tc]), make_text_response("done"))

        def handle_error(history: list[InputItem], error: Exception | None) -> list[InputItem]:
            return history

        agent = Agent(
            logger=_logger,
            llm=llm,
            tools=[FailingTool()],
            config=AgentConfig(max_turns=5),
            hooks=AgentHooks(before_query=handle_error),
        )

        result = await agent.run([TextInput(text="test")])

        # 1 ErrorTurn + 1 failed tool call = 2
        assert result.error_count == 2


# --- Computed fields ---


class TestAgentResultComputedFields:
    def _make_result(self, **kwargs: Any) -> AgentResult:
        defaults: dict[str, Any] = {
            "final_answer": "answer",
            "final_error": None,
            "turns": [],
            "final_duration_seconds": 1.0,
            "state": {},
        }
        defaults.update(kwargs)
        return AgentResult(**defaults)

    def _make_qr(self, in_tokens: int = 10, out_tokens: int = 5) -> QueryResult:
        return QueryResult(
            output_text="",
            metadata=QueryResultMetadata(in_tokens=in_tokens, out_tokens=out_tokens),
            tool_calls=[],
            history=[],
        )

    def test_success_true_when_no_error(self):
        assert self._make_result(final_error=None).success is True

    def test_success_false_when_error(self):
        result = self._make_result(final_error=SerializableException(type="Err", message="bad"))
        assert result.success is False

    def test_total_turns(self):
        result = self._make_result(
            turns=[
                AgentTurn(query_result=self._make_qr()),
                ErrorTurn(error=SerializableException(type="E", message="e")),
                AgentTurn(query_result=self._make_qr()),
            ]
        )
        assert result.total_turns == 3

    def test_tool_usage(self):
        result = self._make_result(
            turns=[
                AgentTurn(
                    query_result=self._make_qr(),
                    tool_call_records=[
                        ToolCallRecord(tool_call=ToolCall(id="1", name="search", args={}), tool_output=ToolOutput(output="r"), duration_seconds=0.1),
                        ToolCallRecord(tool_call=ToolCall(id="2", name="search", args={}), tool_output=ToolOutput(output="r"), duration_seconds=0.1),
                        ToolCallRecord(tool_call=ToolCall(id="3", name="submit", args={}), tool_output=ToolOutput(output="r"), duration_seconds=0.1),
                    ],
                ),
            ]
        )
        assert result.tool_usage == {"search": 2, "submit": 1}
        assert result.tool_calls_count == 3

    def test_error_count(self):
        err = SerializableException(type="E", message="e")
        result = self._make_result(
            turns=[
                ErrorTurn(error=err),
                AgentTurn(
                    query_result=self._make_qr(),
                    tool_call_records=[
                        ToolCallRecord(tool_call=ToolCall(id="1", name="a", args={}), tool_output=ToolOutput(output="r"), duration_seconds=0.1, error=err),
                        ToolCallRecord(tool_call=ToolCall(id="2", name="b", args={}), tool_output=ToolOutput(output="r"), duration_seconds=0.1),
                    ],
                ),
            ]
        )
        # 1 ErrorTurn + 1 failed tool = 2
        assert result.error_count == 2

    def test_final_aggregated_metadata(self):
        result = self._make_result(
            turns=[
                AgentTurn(query_result=self._make_qr(in_tokens=100, out_tokens=50)),
                ErrorTurn(error=SerializableException(type="E", message="e")),
                AgentTurn(query_result=self._make_qr(in_tokens=200, out_tokens=100)),
            ]
        )
        agg = result.final_aggregated_metadata
        assert agg.in_tokens == 300
        assert agg.out_tokens == 150

    def test_final_aggregated_metadata_skips_error_turns(self):
        result = self._make_result(
            turns=[ErrorTurn(error=SerializableException(type="E", message="e"))]
        )
        assert result.final_aggregated_metadata == QueryResultMetadata()


# --- Models ---


class TestModels:
    def test_tool_call_record_success_computed_from_error(self):
        tc = ToolCall(id="1", name="t", args={})
        ok = ToolCallRecord(tool_call=tc, tool_output=ToolOutput(output="r"), duration_seconds=0.1)
        assert ok.success is True

        err = ToolCallRecord(
            tool_call=tc,
            tool_output=ToolOutput(output="r"),
            duration_seconds=0.1,
            error=SerializableException(type="E", message="e"),
        )
        assert err.success is False

    def test_tool_output_success_computed_from_error(self):
        assert ToolOutput(output="r").success is True
        assert ToolOutput(output="r", error="bad").success is False

    def test_serialized_exception_from_exception(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            exc = SerializableException.from_exception(e, tool_name="mytool")

        assert exc.type == "ValueError"
        assert exc.message == "test error"
        assert exc.traceback is not None
        assert "ValueError" in exc.traceback
        assert exc.context == {"tool_name": "mytool"}

    def test_serialized_exception_repr(self):
        exc = SerializableException(type="E", message="msg")
        r = repr(exc)
        assert "SerializableException" in r
        assert "msg" in r

    def test_serialized_exception_context_defaults_to_empty(self):
        exc = SerializableException(type="E", message="m")
        assert exc.context == {}

    def test_turn_result_properties(self):
        tc = ToolCall(id="tc_1", name="echo", args={"text": "hi"})
        qr = QueryResult(output_text="hello", metadata=QueryResultMetadata(), tool_calls=[tc], history=[])
        turn = AgentTurn(query_result=qr)
        tr = TurnResult(turn_number=1, turn=turn, state={}, elapsed_seconds=0.5)

        assert tr.response_text == "hello"
        assert tr.tool_calls == [tc]

    def test_agent_result_repr(self):
        result = AgentResult(
            final_answer="answer", final_error=None, turns=[], final_duration_seconds=1.0, state={}
        )
        r = repr(result)
        assert "AgentResult" in r
        assert "success" in r


# --- truncate_oldest ---


class TestTruncateOldest:
    def test_preserves_single_message(self):
        msgs: list[InputItem] = [TextInput(text="prompt")]
        assert truncate_oldest(msgs) == msgs

    def test_preserves_empty(self):
        assert truncate_oldest([]) == []

    def test_removes_first_response_block(self):
        prompt = TextInput(text="prompt")
        resp1 = RawResponse(response="model reply 1")
        tool_result1 = TextInput(text="tool output 1")
        resp2 = RawResponse(response="model reply 2")
        tool_result2 = TextInput(text="tool output 2")

        msgs: list[InputItem] = [prompt, resp1, tool_result1, resp2, tool_result2]
        result = truncate_oldest(msgs)

        assert len(result) == 3
        assert result[0] is prompt
        assert result[1] is resp2
        assert result[2] is tool_result2


# --- Tool definition ---


class TestToolDefinition:
    def test_definition_property(self):
        tool = EchoTool()
        defn = tool.definition

        assert defn.name == "echo"
        assert defn.body.name == "echo"
        assert defn.body.description == "Echo the input"
        assert "text" in defn.body.properties
        assert defn.body.required == ["text"]

    def test_required_defaults_to_all_params(self):
        class MultiParamTool(Tool):
            name = "multi"
            description = "test"
            parameters = {"a": {"type": "string"}, "b": {"type": "integer"}}

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output="ok")

        tool = MultiParamTool()
        assert set(tool.required) == {"a", "b"}

    def test_missing_name_raises(self):
        with pytest.raises(TypeError, match="must define class attribute 'name'"):

            class NoNameTool(Tool):
                description = "test"
                parameters: dict[str, Any] = {}

                async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                    return ToolOutput(output="ok")

    def test_missing_description_raises(self):
        with pytest.raises(TypeError, match="must define class attribute 'description'"):

            class NoDescTool(Tool):
                name = "test"
                parameters: dict[str, Any] = {}

                async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                    return ToolOutput(output="ok")

    def test_missing_parameters_raises(self):
        with pytest.raises(TypeError, match="must define class attribute 'parameters'"):

            class NoParamsTool(Tool):
                name = "test"
                description = "test"

                async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                    return ToolOutput(output="ok")

    def test_required_auto_derived_from_parameters(self):
        class AutoRequiredTool(Tool):
            name = "auto"
            description = "test"
            parameters = {"x": {"type": "string"}, "y": {"type": "integer"}}

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output="ok")

        assert set(AutoRequiredTool.required) == {"x", "y"}

    def test_explicit_required_not_overridden(self):
        class PartialRequiredTool(Tool):
            name = "partial"
            description = "test"
            parameters = {"a": {"type": "string"}, "b": {"type": "string"}}
            required = ["a"]

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output="ok")

        assert PartialRequiredTool.required == ["a"]

    def test_init_override(self):
        class OverridableTool(Tool):
            name = "base"
            description = "base desc"
            parameters = {"a": {"type": "string"}}

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output="ok")

        tool = OverridableTool(name="custom", description="custom desc")
        assert tool.name == "custom"
        assert tool.description == "custom desc"
        assert tool.parameters == {"a": {"type": "string"}}

    def test_class_attrs_work_without_super_init(self):
        class NoSuperTool(Tool):
            name = "nosup"
            description = "no super init"
            parameters = {"q": {"type": "string"}}

            def __init__(self):
                pass  # deliberately skip super().__init__()

            async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
                return ToolOutput(output="ok")

        tool = NoSuperTool()
        assert tool.name == "nosup"
        assert tool.description == "no super init"
        assert tool.parameters == {"q": {"type": "string"}}
        assert tool.required == ["q"]
