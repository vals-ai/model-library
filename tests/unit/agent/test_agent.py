"""Unit tests for model_library.agent"""

import asyncio
import logging
import time as time_module
from pathlib import Path
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
    TimeLimit,
    Tool,
    ToolCallRecord,
    ToolCallSummary,
    ToolOutput,
    TurnLimit,
    TurnResult,
    TurnSummary,
    truncate_oldest,
)


_cfg = AgentConfig(turn_limit=None, time_limit=None)


def make_agent(llm: MagicMock, tools: list[Tool] | None = None, **kwargs: Any) -> Agent:
    kwargs.setdefault("name", "test")
    kwargs.setdefault("config", _cfg)
    return Agent(llm=llm, tools=tools or [], **kwargs)


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
    duration: float | None = None,
) -> QueryResult:
    """LLM response with tool calls"""
    meta = metadata or make_metadata()
    if duration is not None:
        meta.duration_seconds = duration
    return QueryResult(
        output_text=output_text,
        metadata=meta,
        tool_calls=tool_calls,
        history=[TextInput(text="prompt")],
    )


def make_tool_call(name: str = "echo", args: dict[str, Any] | str | None = None) -> ToolCall:
    return ToolCall(id="tc_1", name=name, args=args or {"text": "hello"})


def mock_llm(*responses: QueryResult | Exception) -> MagicMock:
    """Create a mock LLM that returns the given responses in sequence"""

    class _MockLLM(MagicMock):
        def __rich_repr__(self):
            yield "model_name", self.model_name
            yield "temperature", self.temperature
            yield "max_tokens", self.max_tokens

    llm = _MockLLM()
    llm.model_name = "mock-model"
    llm.query = AsyncMock(side_effect=list(responses))
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
        agent = make_agent(llm)

        result = await agent.run([TextInput(text="say hello")], question_id="q1")

        assert result.final_answer == "hello world"
        assert result.success
        assert result.final_error is None
        assert result.total_turns == 1
        assert result.final_aggregated_metadata.in_tokens == 10
        assert result.final_aggregated_metadata.out_tokens == 5

    async def test_tool_call_then_text_response(self):
        tc = make_tool_call("echo", {"text": "ping"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("got: ping"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="echo test")], question_id="q1")

        assert result.final_answer == "got: ping"
        assert result.success
        assert llm.query.call_count == 2
        assert result.total_turns == 2
        assert result.tool_calls_count == 1
        assert result.tool_usage == {"echo": 1}

    async def test_done_tool_stops_loop(self):
        tc = make_tool_call("submit", {"answer": "42"})
        llm = mock_llm(make_tool_response([tc]))
        agent = make_agent(llm, [DoneTool()])

        result = await agent.run([TextInput(text="answer?")], question_id="q1")

        assert result.final_answer == "42"
        assert result.success
        assert llm.query.call_count == 1

    async def test_max_turns_sets_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)
        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=3), time_limit=None))

        result = await agent.run([TextInput(text="go")], question_id="q1")

        assert not result.success
        assert result.final_error is not None
        assert result.final_error.type == "MaxTurnsExceeded"
        assert result.final_answer == ""
        assert llm.query.call_count == 3
        assert result.total_turns == 3

    async def test_turn_message_appended_to_history(self):
        """turn_message hook injects an InputItem into history before each query"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        def budget_message(turn_number: int, max_turns: int) -> InputItem | None:
            return TextInput(text=f"Turn {turn_number}/{max_turns}")

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=5, turn_message=budget_message), time_limit=None))
        await agent.run([TextInput(text="go")], question_id="q1")

        # second query should have the turn message in its input
        second_call_input = llm.query.call_args_list[1].kwargs["input"]
        injected = [m for m in second_call_input if isinstance(m, TextInput) and "Turn " in m.text]
        assert len(injected) == 1
        assert injected[0].text == "Turn 2/5"

    async def test_turn_message_none_skipped(self):
        """turn_message returning None does not append anything"""
        llm = mock_llm(make_text_response("done"))

        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5, turn_message=lambda t, m: None), time_limit=None))
        await agent.run([TextInput(text="go")], question_id="q1")

        first_call_input = llm.query.call_args_list[0].kwargs["input"]
        assert len(first_call_input) == 1  # just the original prompt

    async def test_max_time_sets_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc], duration=30.0) for _ in range(5)]
        llm = mock_llm(*responses)

        # wall clock advances 30s per monotonic() call; query duration matches
        # so retry_overhead stays 0 and effective_elapsed tracks wall time
        call_count = [0]

        def fake_monotonic() -> float:
            call_count[0] += 1
            return call_count[0] * 5.0  # 5s per call, ~30s per turn

        with patch.object(time_module, "monotonic", side_effect=fake_monotonic):
            agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=None, time_limit=TimeLimit(max_seconds=50)))
            result = await agent.run([TextInput(text="go")], question_id="q1")

        assert not result.success
        assert result.final_error is not None
        assert result.final_error.type == "MaxTimeExceeded"

    async def test_time_message_appended_to_history(self):
        """time_message hook injects an InputItem into history before each query"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        def remaining_message(elapsed_seconds: float, max_seconds: float) -> InputItem | None:
            remaining = max_seconds - elapsed_seconds
            return TextInput(text=f"{remaining:.0f}s remaining")

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=None, time_limit=TimeLimit(max_seconds=300, time_message=remaining_message)))
        await agent.run([TextInput(text="go")], question_id="q1")

        # first query should have the time message
        first_call_input = llm.query.call_args_list[0].kwargs["input"]
        injected = [m for m in first_call_input if isinstance(m, TextInput) and "remaining" in m.text]
        assert len(injected) == 1


    async def test_text_only_auto_stops_without_should_stop(self):
        llm = mock_llm(make_text_response("done"))
        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=10), time_limit=None))

        result = await agent.run([TextInput(text="go")], question_id="q1")

        assert result.final_answer == "done"
        assert llm.query.call_count == 1

    async def test_input_passed_to_llm(self):
        llm = mock_llm(make_text_response("ok"))
        agent = make_agent(llm)

        await agent.run([TextInput(text="a"), TextInput(text="b")], question_id="q1")

        messages = llm.query.call_args.kwargs["input"]
        assert len(messages) == 2

    async def test_state_defaults_to_empty_dict(self):
        """State is not on the result; caller owns it by reference"""
        llm = mock_llm(make_text_response("ok"))
        agent = make_agent(llm)
        state: dict[str, Any] = {}

        await agent.run([TextInput(text="go")], question_id="q1", state=state)

        assert state == {}

    async def test_initial_state_preserved(self):
        llm = mock_llm(make_text_response("ok"))
        agent = make_agent(llm)
        state = {"initial": True}

        await agent.run([TextInput(text="go")], question_id="q1", state=state)

        assert state["initial"] is True


# --- File output ---


class TestAgentFileOutput:
    async def test_init_dir_written(self):
        """init/ directory contains config.json, state.json, history.bin"""
        llm = mock_llm(make_text_response("done"))
        agent = make_agent(llm)
        state = {"key": "value"}

        result = await agent.run([TextInput(text="hello")], question_id="q1", state=state)

        init_dir = result.output_dir / "turns" / "init"
        assert init_dir.exists()
        assert (init_dir / "config.json").exists()
        assert (init_dir / "state.json").exists()
        assert (init_dir / "history.bin").exists()

        import json
        config = json.loads((init_dir / "config.json").read_text())
        assert "tools" in config
        assert "llm" in config
        assert "model_name" in config["llm"]
        assert "temperature" in config["llm"]
        assert "max_tokens" in config["llm"]
        assert "agent_config" in config

        init_state = json.loads((init_dir / "state.json").read_text())
        assert init_state == {"key": "value"}

    async def test_turn_dirs_written(self):
        """Each successful turn gets a turn_NNN/ directory with result.json, state.json, history.bin"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="go")], question_id="q1")

        turn_dir = result.output_dir / "turns" / "turn_001"
        assert turn_dir.exists()
        assert (turn_dir / "result.json").exists()
        assert (turn_dir / "state.json").exists()
        assert (turn_dir / "history.bin").exists()

        turn_dir2 = result.output_dir / "turns" / "turn_002"
        assert turn_dir2.exists()
        assert (turn_dir2 / "result.json").exists()

    async def test_error_turn_dir_written(self):
        """Error turns get a turn_NNN/ directory with just error.json"""
        llm = mock_llm(RuntimeError("boom"), make_text_response("recovered"))
        hooks = AgentHooks(before_query=lambda h, e: h)
        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None), hooks=hooks)

        result = await agent.run([TextInput(text="go")], question_id="q1")

        error_dir = result.output_dir / "turns" / "turn_001"
        assert error_dir.exists()
        assert (error_dir / "error.json").exists()
        assert not (error_dir / "result.json").exists()
        assert not (error_dir / "state.json").exists()
        assert not (error_dir / "history.bin").exists()

    async def test_result_json_written(self):
        """result.json is written at the output_dir root"""
        llm = mock_llm(make_text_response("done"))
        agent = make_agent(llm)

        result = await agent.run([TextInput(text="go")], question_id="q1")

        result_path = result.output_dir / "result.json"
        assert result_path.exists()

        import json
        data = json.loads(result_path.read_text())
        assert data["final_answer"] == "done"
        assert data["success"] is True


# --- Tool execution ---


class TestAgentToolExecution:
    async def test_unknown_tool_records_error(self):
        tc = make_tool_call("nonexistent", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("ok"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="try unknown")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert "not found" in tc_summary.error.message

    async def test_tool_exception_caught_and_recorded(self):
        tc = make_tool_call("fail", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = make_agent(llm, [FailingTool()])

        result = await agent.run([TextInput(text="try failing")], question_id="q1")

        assert result.final_answer == "recovered"
        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert "tool broke" in tc_summary.error.message
        assert tc_summary.error.traceback is not None

    async def test_tool_error_return_recorded(self):
        """Tool returning ToolOutput with error set (doesn't raise)"""
        tc = make_tool_call("soft_fail", {})
        llm = mock_llm(make_tool_response([tc]), make_text_response("ok"))
        agent = make_agent(llm, [ErrorReturnTool()])

        result = await agent.run([TextInput(text="try soft fail")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert tc_summary.error.type == "ToolFailed"
        assert tc_summary.error.message == "something went wrong"
        assert tc_summary.output_length == len("error details")

    async def test_string_args_parsed_as_json(self):
        tc = ToolCall(id="tc_1", name="echo", args='{"text": "from json"}')
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        assert result.turns[0].tool_calls[0].args_lengths == {"text": len("from json")}

    async def test_invalid_json_args_errors(self):
        tc = ToolCall(id="tc_1", name="echo", args="not json")
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert "Unparseable" in tc_summary.error.message

    async def test_json_args_non_dict_errors(self):
        """JSON that parses to non-dict (e.g. list) is unparseable"""
        tc = ToolCall(id="tc_1", name="echo", args="[1, 2, 3]")
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert "Unparseable" in tc_summary.error.message

    async def test_missing_required_param_records_error(self):
        """Tool with required params called without them → error record"""
        tc = ToolCall(id="tc_1", name="echo", args={})  # missing "text"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert tc_summary.error is not None
        assert "Missing required parameters" in tc_summary.error.message
        assert "text" in tc_summary.error.message

    async def test_missing_multiple_required_params(self):
        tc = ToolCall(id="tc_1", name="set_state", args={})  # missing "key" and "value"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = make_agent(llm, [StateTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert "key" in tc_summary.error.message
        assert "value" in tc_summary.error.message

    async def test_partial_required_params_reports_missing(self):
        tc = make_tool_call("set_state", {"key": "k"})  # has "key", missing "value"
        llm = mock_llm(make_tool_response([tc]), make_text_response("recovered"))
        agent = make_agent(llm, [StateTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert not tc_summary.success
        assert "value" in tc_summary.error.message
        assert "key" not in tc_summary.error.message  # "key" was provided

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
        agent = make_agent(llm, [OptionalTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert tc_summary.success
        assert tc_summary.output_length == len("yes")

    async def test_no_params_tool_accepts_empty_args(self):
        tc = ToolCall(id="tc_1", name="fail", args={})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        # FailingTool has parameters={}, required=[] — should pass validation
        # (it will fail in execute, but that's a different error)
        agent = make_agent(llm, [FailingTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        # Fails from execute(), not from validation
        assert "tool broke" in tc_summary.error.message

    async def test_state_shared_between_tools(self):
        set_call = make_tool_call("set_state", {"key": "found", "value": "yes"})
        echo_call = make_tool_call("echo", {"text": "check"})
        llm = mock_llm(make_tool_response([set_call]), make_tool_response([echo_call]), make_text_response("done"))
        agent = make_agent(llm, [StateTool(), EchoTool()])
        state: dict[str, Any] = {}

        await agent.run([TextInput(text="test")], question_id="q1", state=state)

        assert state["found"] == "yes"

    async def test_tool_call_duration_tracked(self):
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tc_summary = result.turns[0].tool_calls[0]
        assert isinstance(tc_summary.duration_seconds, float)
        assert tc_summary.duration_seconds >= 0

    async def test_done_tool_skips_remaining_tool_calls(self):
        done_call = make_tool_call("submit", {"answer": "42"})
        echo_call = make_tool_call("echo", {"text": "should not run"})
        llm = mock_llm(make_tool_response([done_call, echo_call]))
        agent = make_agent(llm, [DoneTool(), EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert result.final_answer == "42"
        assert isinstance(result.turns[0], TurnSummary)
        assert len(result.turns[0].tool_calls) == 1
        assert result.turns[0].tool_calls[0].tool_name == "submit"


# --- Tool metadata ---


class TestAgentToolMetadata:
    async def test_tool_metadata_aggregated_in_turn(self):
        tc = make_tool_call("retrieve", {"q": "test"})
        llm = mock_llm(
            make_tool_response([tc], metadata=make_metadata(in_tokens=100, out_tokens=50)),
            make_text_response("done"),
        )
        agent = make_agent(llm, [LLMCallingTool()])

        result = await agent.run([TextInput(text="retrieve")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        turn = result.turns[0]
        # Query metadata: 100 in + 50 out
        assert turn.metadata.in_tokens == 100
        assert turn.metadata.out_tokens == 50
        # Tool sub-LLM metadata available per tool call
        assert turn.tool_calls[0].metadata is not None
        assert turn.tool_calls[0].metadata.in_tokens == 50
        assert turn.tool_calls[0].metadata.out_tokens == 20

    async def test_metadata_aggregates_across_turns(self):
        tc = make_tool_call("echo", {"text": "hi"})
        meta = make_metadata(in_tokens=100, out_tokens=50, cost_total=0.02)
        llm = mock_llm(make_tool_response([tc], metadata=meta), make_text_response("done", metadata=meta))
        agent = make_agent(llm, [EchoTool()])

        result = await agent.run([TextInput(text="test")], question_id="q1")

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
            return turn_result.turn_number >= 3

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=10), time_limit=None), hooks=AgentHooks(should_stop=stop_at_3))

        result = await agent.run([TextInput(text="go")], question_id="q1")

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

        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=10), time_limit=None), hooks=AgentHooks(should_stop=stop_on_exit))

        result = await agent.run([TextInput(text="go")], question_id="q1")

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

        agent = make_agent(llm, hooks=AgentHooks(determine_answer=custom_answer))

        result = await agent.run([TextInput(text="test")], question_id="q1")

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

        agent = make_agent(llm, [StateTool()], hooks=AgentHooks(determine_answer=answer_from_state))

        result = await agent.run([TextInput(text="test")], question_id="q1")

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

        agent = make_agent(llm, [StateTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=3), time_limit=None), hooks=AgentHooks(determine_answer=salvage))

        result = await agent.run([TextInput(text="go")], question_id="q1")

        assert result.final_answer == "salvaged"
        assert result.final_error is not None
        assert result.final_error.type == "MaxTurnsExceeded"
        assert not result.success

    async def test_default_determine_answer_returns_text(self):
        llm = mock_llm(make_text_response("raw"))
        agent = make_agent(llm)

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert result.success
        assert result.final_answer == "raw"

    async def test_default_determine_answer_returns_empty_on_error(self):
        tc = make_tool_call("echo", {"text": "hi"})
        responses = [make_tool_response([tc]) for _ in range(5)]
        llm = mock_llm(*responses)

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=2), time_limit=None))

        result = await agent.run([TextInput(text="go")], question_id="q1")

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

        agent = make_agent(llm, [EchoTool()], hooks=AgentHooks(on_tool_result=on_result))

        await agent.run([TextInput(text="test")], question_id="q1")

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

        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None), hooks=AgentHooks(before_query=handle_error))

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert result.final_answer == "recovered"
        assert result.success
        assert len(result.turns) == 2
        assert isinstance(result.turns[0], ErrorTurn)
        assert isinstance(result.turns[1], TurnSummary)
        assert len(errors_seen) == 1
        assert isinstance(errors_seen[0], RuntimeError)

    async def test_before_query_absent_with_error_stops(self):
        """Without before_query, query error is re-raised and sets final_error"""
        llm = mock_llm(RuntimeError("query failed"), make_text_response("unreachable"))

        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None))

        result = await agent.run([TextInput(text="test")], question_id="q1")

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

        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None))

        result = await agent.run([TextInput(text="test")], question_id="q1")

        assert isinstance(result.turns[0], ErrorTurn)
        assert result.turns[0].error.type == "ValueError"
        assert result.turns[0].error.message == "bad input"

    async def test_error_count_includes_error_turns_and_failed_tools(self):
        tc = make_tool_call("fail", {})
        llm = mock_llm(RuntimeError("query failed"), make_tool_response([tc]), make_text_response("done"))

        def handle_error(history: list[InputItem], error: Exception | None) -> list[InputItem]:
            return history

        agent = make_agent(llm, [FailingTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None), hooks=AgentHooks(before_query=handle_error))

        result = await agent.run([TextInput(text="test")], question_id="q1")

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
            "output_dir": Path("/tmp/test"),
        }
        defaults.update(kwargs)
        return AgentResult(**defaults)

    def _make_turn_summary(self, in_tokens: int = 10, out_tokens: int = 5, **kwargs: Any) -> TurnSummary:
        defaults: dict[str, Any] = {
            "output_text_length": 0,
            "reasoning_length": 0,
            "finish_reason": {"reason": "stop", "raw": "stop"},
            "metadata": QueryResultMetadata(in_tokens=in_tokens, out_tokens=out_tokens),
            "duration_seconds": kwargs.pop("duration_seconds", 0.0),
            "retry_overhead_seconds": kwargs.pop("retry_overhead_seconds", 0.0),
        }
        defaults.update(kwargs)
        return TurnSummary(**defaults)

    def test_success_true_when_no_error(self):
        assert self._make_result(final_error=None).success is True

    def test_success_false_when_error(self):
        result = self._make_result(final_error=SerializableException(type="Err", message="bad"))
        assert result.success is False

    def test_total_turns(self):
        result = self._make_result(
            turns=[
                self._make_turn_summary(),
                ErrorTurn(error=SerializableException(type="E", message="e"), duration_seconds=0.0),
                self._make_turn_summary(),
            ]
        )
        assert result.total_turns == 3

    def test_tool_usage(self):
        result = self._make_result(
            turns=[
                self._make_turn_summary(tool_calls=[
                    ToolCallSummary(tool_name="search", tool_call_id="1", args_lengths={}, output_length=1, success=True, done=False, duration_seconds=0.1),
                    ToolCallSummary(tool_name="search", tool_call_id="2", args_lengths={}, output_length=1, success=True, done=False, duration_seconds=0.1),
                    ToolCallSummary(tool_name="submit", tool_call_id="3", args_lengths={}, output_length=1, success=True, done=False, duration_seconds=0.1),
                ]),
            ]
        )
        assert result.tool_usage == {"search": 2, "submit": 1}
        assert result.tool_calls_count == 3

    def test_error_count(self):
        err = SerializableException(type="E", message="e")
        result = self._make_result(
            turns=[
                ErrorTurn(error=err, duration_seconds=0.0),
                self._make_turn_summary(tool_calls=[
                    ToolCallSummary(tool_name="a", tool_call_id="1", args_lengths={}, output_length=1, success=False, done=False, duration_seconds=0.1, error=err),
                    ToolCallSummary(tool_name="b", tool_call_id="2", args_lengths={}, output_length=1, success=True, done=False, duration_seconds=0.1),
                ]),
            ]
        )
        # 1 ErrorTurn + 1 failed tool = 2
        assert result.error_count == 2

    def test_final_aggregated_metadata(self):
        result = self._make_result(
            turns=[
                self._make_turn_summary(in_tokens=100, out_tokens=50),
                ErrorTurn(error=SerializableException(type="E", message="e"), duration_seconds=0.0),
                self._make_turn_summary(in_tokens=200, out_tokens=100),
            ]
        )
        agg = result.final_aggregated_metadata
        assert agg.in_tokens == 300
        assert agg.out_tokens == 150

    def test_final_aggregated_metadata_skips_error_turns(self):
        result = self._make_result(
            turns=[ErrorTurn(error=SerializableException(type="E", message="e"), duration_seconds=0.0)]
        )
        assert result.final_aggregated_metadata == QueryResultMetadata()

    def test_duration_computed_fields(self):
        result = self._make_result(
            final_duration_seconds=10.0,
            turns=[
                self._make_turn_summary(duration_seconds=4.0, retry_overhead_seconds=1.0),
                ErrorTurn(error=SerializableException(type="E", message="e"), duration_seconds=0.3),
                self._make_turn_summary(duration_seconds=5.0, retry_overhead_seconds=0.5),
            ],
        )
        assert result.final_retry_overhead_seconds == 1.5
        assert result.final_effective_duration_seconds == 8.5  # 10.0 - 1.5


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
        turn = AgentTurn(duration_seconds=0.0, query_result=qr)
        tr = TurnResult(turn_number=1, turn=turn, state={}, elapsed_seconds=0.5)

        assert tr.response_text == "hello"
        assert tr.tool_calls == [tc]

    def test_agent_result_repr(self):
        result = AgentResult(
            final_answer="answer", final_error=None, turns=[], final_duration_seconds=1.0, state={},
            output_dir=Path("/tmp/test"),
        )
        r = repr(result)
        assert "AgentResult" in r
        assert "success" in r

    # --- ToolCallSummary ---

    def test_tool_call_summary_from_record(self):
        tc = ToolCall(id="1", name="search", args={"query": "hello world", "limit": "10"})
        output = ToolOutput(output="found 3 results", error=None, done=False)
        record = ToolCallRecord(tool_call=tc, tool_output=output, duration_seconds=1.5)

        summary = record.to_summary()

        assert summary.tool_name == "search"
        assert summary.tool_call_id == "1"
        assert summary.args_lengths == {"query": 11, "limit": 2}
        assert summary.output_length == 15
        assert summary.success is True
        assert summary.done is False
        assert summary.error is None
        assert summary.duration_seconds == 1.5
        assert summary.metadata is None

    def test_tool_call_summary_with_error(self):
        tc = ToolCall(id="2", name="fail", args={"x": "y"})
        output = ToolOutput(output="boom", error="something broke", done=False)
        error = SerializableException(type="ToolFailed", message="something broke")
        record = ToolCallRecord(tool_call=tc, tool_output=output, duration_seconds=0.5, error=error)

        summary = record.to_summary()

        assert summary.success is False
        assert summary.error is not None
        assert summary.error.type == "ToolFailed"
        assert summary.output_length == 4

    def test_tool_call_summary_with_metadata(self):
        tc = ToolCall(id="3", name="llm_tool", args={})
        meta = QueryResultMetadata(in_tokens=100, out_tokens=50)
        output = ToolOutput(output="result", metadata=meta, done=True)
        record = ToolCallRecord(tool_call=tc, tool_output=output, duration_seconds=2.0)

        summary = record.to_summary()

        assert summary.done is True
        assert summary.metadata is not None
        assert summary.metadata.in_tokens == 100

    # --- TurnSummary ---

    def test_turn_summary_from_agent_turn(self):
        meta = QueryResultMetadata(in_tokens=100, out_tokens=50, duration_seconds=1.0)
        qr = QueryResult(
            output_text="hello world",
            reasoning="because reasons",
            metadata=meta,
            tool_calls=[],
            history=[],
        )
        turn = AgentTurn(
            query_result=qr,
            tool_call_records=[],
            duration_seconds=2.0,
            retry_overhead_seconds=0.5,
        )

        summary = turn.to_summary()

        assert summary.output_text_length == 11
        assert summary.reasoning_length == 15
        assert summary.metadata.in_tokens == 100
        assert summary.metadata.out_tokens == 50
        assert summary.tool_calls == []
        assert summary.duration_seconds == 2.0
        assert summary.retry_overhead_seconds == 0.5
        assert summary.finish_reason == qr.finish_reason

    def test_turn_summary_with_tool_calls(self):
        meta = QueryResultMetadata(in_tokens=10, out_tokens=5)
        tc = ToolCall(id="1", name="echo", args={"text": "hi"})
        qr = QueryResult(metadata=meta, tool_calls=[tc], history=[])
        tool_meta = QueryResultMetadata(in_tokens=50, out_tokens=25)
        record = ToolCallRecord(
            tool_call=tc,
            tool_output=ToolOutput(output="hi", metadata=tool_meta),
            duration_seconds=0.1,
        )
        turn = AgentTurn(
            query_result=qr,
            tool_call_records=[record],
            duration_seconds=1.0,
        )

        summary = turn.to_summary()

        assert len(summary.tool_calls) == 1
        assert summary.tool_calls[0].tool_name == "echo"
        assert summary.metadata.in_tokens == 10
        assert summary.tool_calls[0].metadata.in_tokens == 50

    def test_turn_summary_none_text_and_reasoning(self):
        qr = QueryResult(output_text=None, reasoning=None, metadata=QueryResultMetadata(), history=[])
        turn = AgentTurn(query_result=qr, duration_seconds=0.5)

        summary = turn.to_summary()

        assert summary.output_text_length == 0
        assert summary.reasoning_length == 0


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


# --- max_tool_calls_per_turn ---


class TestMaxToolCallsPerTurn:
    async def test_cap_skips_excess_tool_calls(self):
        """Tool calls beyond the cap get skip messages, not executed"""
        tc1 = make_tool_call("echo", {"text": "first"})
        tc2 = make_tool_call("echo", {"text": "second"})
        tc3 = make_tool_call("echo", {"text": "third"})
        llm = mock_llm(make_tool_response([tc1, tc2, tc3]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=None, time_limit=None, max_tool_calls_per_turn=1))

        result = await agent.run([TextInput(text="go")], question_id="q1")

        assert isinstance(result.turns[0], TurnSummary)
        tcs = result.turns[0].tool_calls
        assert len(tcs) == 3
        assert tcs[0].output_length == len("first")
        assert tcs[0].success
        assert tcs[1].output_length == len("Skipped: tool call limit exceeded")
        assert tcs[2].output_length == len("Skipped: tool call limit exceeded")

    async def test_cap_skipped_calls_appended_to_history(self):
        """Skipped tool calls still get ToolResult in history for provider compatibility"""
        tc1 = make_tool_call("echo", {"text": "ok"})
        tc2 = make_tool_call("echo", {"text": "skip me"})
        llm = mock_llm(make_tool_response([tc1, tc2]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=None, time_limit=None, max_tool_calls_per_turn=1))

        await agent.run([TextInput(text="go")], question_id="q1")

        # second LLM call should have both tool results in its input
        second_call_input = llm.query.call_args_list[1].kwargs["input"]
        from model_library.base.input import ToolResult
        tool_results = [m for m in second_call_input if isinstance(m, ToolResult)]
        assert len(tool_results) == 2


# --- Logging ---


class TestAgentLogging:
    async def test_before_query_logs_error(self, caplog: pytest.LogCaptureFixture):
        """before_query logs when it receives a non-None error"""
        llm = mock_llm(RuntimeError("boom"), make_text_response("done"))
        agent = make_agent(llm, config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None), hooks=AgentHooks(before_query=lambda h, e: h))

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert any("before_query: handling error RuntimeError: boom" in r.message for r in caplog.records)

    async def test_before_query_no_error_no_log(self, caplog: pytest.LogCaptureFixture):
        """before_query does not log when there is no error"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert not any("before_query: handling error" in r.message for r in caplog.records)

    async def test_before_query_logs_history_modification(self, caplog: pytest.LogCaptureFixture):
        """before_query logs when the hook returns a modified history"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        def shrink(history: list[InputItem], error: Exception | None) -> list[InputItem]:
            return history[:1]

        agent = make_agent(llm, [EchoTool()], hooks=AgentHooks(before_query=shrink))

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert any("before_query modified history" in r.message for r in caplog.records)

    async def test_before_query_unchanged_history_no_modification_log(self, caplog: pytest.LogCaptureFixture):
        """before_query does not log 'modified history' when history is unchanged"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))
        agent = make_agent(llm, [EchoTool()])

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert not any("before_query modified history" in r.message for r in caplog.records)

    async def test_turn_message_logs_only_when_injected(self, caplog: pytest.LogCaptureFixture):
        """turn_message only logs when the hook returns a non-None message"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        def message_on_turn_2(turn_number: int, max_turns: int) -> InputItem | None:
            return TextInput(text=f"turn {turn_number}") if turn_number == 2 else None

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=5, turn_message=message_on_turn_2), time_limit=None))

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        turn_msg_logs = [r for r in caplog.records if "Hook turn_message(" in r.message]
        assert len(turn_msg_logs) == 1
        assert "2/5" in turn_msg_logs[0].message

    async def test_time_message_logs_only_when_injected(self, caplog: pytest.LogCaptureFixture):
        """time_message only logs when the hook returns a non-None message"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_text_response("done"))

        call_count = [0]

        def message_once(elapsed: float, max_seconds: float) -> InputItem | None:
            call_count[0] += 1
            return TextInput(text="hurry") if call_count[0] == 1 else None

        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=None, time_limit=TimeLimit(max_seconds=300, time_message=message_once)))

        with caplog.at_level(logging.DEBUG):
            await agent.run([TextInput(text="go")], question_id="q1")

        time_msg_logs = [r for r in caplog.records if "Hook time_message(" in r.message]
        assert len(time_msg_logs) == 1
        assert "hurry" in time_msg_logs[0].message

    async def test_should_stop_no_info_log_when_continues(self, caplog: pytest.LogCaptureFixture):
        """should_stop does not log the stop message when it returns False"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]), make_tool_response([tc]))
        # always returns False; max_turns stops the loop instead
        agent = make_agent(llm, [EchoTool()], config=AgentConfig(turn_limit=TurnLimit(max_turns=1), time_limit=None), hooks=AgentHooks(should_stop=lambda _: False))

        with caplog.at_level(logging.INFO):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert not any("should_stop hook returned True" in r.message for r in caplog.records)

    async def test_should_stop_info_log_when_stops(self, caplog: pytest.LogCaptureFixture):
        """should_stop produces an INFO log when it returns True"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm = mock_llm(make_tool_response([tc]))
        agent = make_agent(llm, [EchoTool()], hooks=AgentHooks(should_stop=lambda _: True))

        with caplog.at_level(logging.INFO):
            await agent.run([TextInput(text="go")], question_id="q1")

        assert any("should_stop hook returned True" in r.message for r in caplog.records)


# --- Retry time budget ---


class TestRetryTimeBudget:
    """Tests for retry overhead time tracking and budget behavior.

    Uses a fake clock that only advances during LLM queries (simulating retry delays).
    Other monotonic() calls (time checks, tool execution) see the same clock value,
    so only query wall time contributes to elapsed/overhead calculations.
    """

    @staticmethod
    def _make_timed_llm(*responses: QueryResult, wall_per_query: float = 20.0):
        """Mock LLM where each query advances a fake clock by wall_per_query seconds"""
        clock = [0.0]

        def fake_monotonic() -> float:
            return clock[0]

        response_list = list(responses)
        idx = [0]

        async def query_side_effect(**kwargs: Any) -> QueryResult:
            resp = response_list[idx[0]]
            idx[0] += 1
            clock[0] += wall_per_query
            return resp

        llm = MagicMock()
        llm.query = AsyncMock(side_effect=query_side_effect)
        llm.instance_logger = logging.getLogger("mock_llm")
        return llm, fake_monotonic

    async def test_retry_overhead_excluded_from_budget(self):
        """Retry time doesn't count: wall=80s exceeds budget=30, but effective=6s doesn't"""
        tc = make_tool_call("echo", {"text": "hi"})
        # 3 tool turns (reported 2s each) + 1 text turn (reported 0s)
        # wall: 4 × 20 = 80s, overhead: 3×(20-2) + (20-0) = 74, effective: 80-74 = 6
        llm, fake_mono = self._make_timed_llm(
            make_tool_response([tc], duration=2.0),
            make_tool_response([tc], duration=2.0),
            make_tool_response([tc], duration=2.0),
            make_text_response("done"),
            wall_per_query=20.0,
        )

        with patch.object(time_module, "monotonic", new=fake_mono):
            agent = make_agent(llm, [EchoTool()], config=AgentConfig(
                turn_limit=None,
                time_limit=TimeLimit(max_seconds=30),
            ))
            result = await agent.run([TextInput(text="go")], question_id="q1")

        assert result.success
        assert result.final_duration_seconds == 80
        assert result.final_retry_overhead_seconds == 74
        assert result.final_effective_duration_seconds == 6

    async def test_include_retries_uses_wall_clock(self):
        """With include_retries=True, wall clock counts — same scenario hits budget"""
        tc = make_tool_call("echo", {"text": "hi"})
        llm, fake_mono = self._make_timed_llm(
            make_tool_response([tc], duration=2.0),
            make_tool_response([tc], duration=2.0),
            make_tool_response([tc], duration=2.0),
            make_text_response("done"),
            wall_per_query=20.0,
        )

        with patch.object(time_module, "monotonic", new=fake_mono):
            agent = make_agent(llm, [EchoTool()], config=AgentConfig(
                turn_limit=None,
                time_limit=TimeLimit(max_seconds=30, include_retries=True),
            ))
            result = await agent.run([TextInput(text="go")], question_id="q1")

        assert not result.success
        assert result.final_error is not None
        assert result.final_error.type == "MaxTimeExceeded"

    async def test_retry_overhead_tracked_on_turns(self):
        """Each AgentTurn records its per-turn retry overhead"""
        tc = make_tool_call("echo", {"text": "hi"})
        # tool turn: wall=20, reported=2 → overhead=18
        # text turn: wall=20, reported=0 → overhead=20
        llm, fake_mono = self._make_timed_llm(
            make_tool_response([tc], duration=2.0),
            make_text_response("done"),
            wall_per_query=20.0,
        )

        with patch.object(time_module, "monotonic", new=fake_mono):
            agent = make_agent(llm, [EchoTool()], config=AgentConfig(
                turn_limit=None,
                time_limit=TimeLimit(max_seconds=300),
            ))
            result = await agent.run([TextInput(text="go")], question_id="q1")

        assert result.success
        assert isinstance(result.turns[0], TurnSummary)
        assert result.turns[0].retry_overhead_seconds == 18.0
        assert isinstance(result.turns[1], TurnSummary)
        assert result.turns[1].retry_overhead_seconds == 20.0
        assert result.final_retry_overhead_seconds == 38.0
        assert result.final_effective_duration_seconds < result.final_duration_seconds


class TestConcurrentRuns:
    async def test_output_dirs_are_siblings(self):
        """Concurrent runs produce sibling output dirs, not nested ones."""
        llm = mock_llm(
            make_text_response("a"),
            make_text_response("b"),
            make_text_response("c"),
        )
        agent = make_agent(llm)

        results = await asyncio.gather(
            agent.run([TextInput(text="q1")], question_id="q001"),
            agent.run([TextInput(text="q2")], question_id="q002"),
            agent.run([TextInput(text="q3")], question_id="q003"),
        )

        dirs = [r.output_dir for r in results]

        # All output dirs share the same parent
        assert len({d.parent for d in dirs}) == 1

        # No output dir is nested inside another
        for i, d1 in enumerate(dirs):
            for j, d2 in enumerate(dirs):
                if i != j:
                    assert not d2.is_relative_to(d1), f"{d2} is nested inside {d1}"

    async def test_each_run_has_own_log_file(self):
        """Each concurrent run gets its own agent.log."""
        llm = mock_llm(make_text_response("a"), make_text_response("b"))
        agent = make_agent(llm)

        results = await asyncio.gather(
            agent.run([TextInput(text="q1")], question_id="q001"),
            agent.run([TextInput(text="q2")], question_id="q002"),
        )

        for result in results:
            assert (result.output_dir / "agent.log").exists()

    async def test_llm_receives_question_scoped_logger(self):
        """The logger passed to llm.query() is unique and scoped per question."""
        captured: list[logging.Logger | None] = []

        async def capturing_query(*args: object, **kwargs: object) -> QueryResult:
            captured.append(kwargs.get("logger"))  # type: ignore[arg-type]
            return make_text_response("done")

        llm = MagicMock()
        llm.query = AsyncMock(side_effect=capturing_query)
        agent = make_agent(llm)

        await asyncio.gather(
            agent.run([TextInput(text="q1")], question_id="q001"),
            agent.run([TextInput(text="q2")], question_id="q002"),
        )

        assert len(captured) == 2
        assert captured[0] is not captured[1]
        names = {lg.name for lg in captured if lg is not None}
        assert any("q001" in name for name in names)
        assert any("q002" in name for name in names)
