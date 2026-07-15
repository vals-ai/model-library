# tests/unit/test_atif.py
import json
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from model_library.agent.agent import Agent
from model_library.agent.config import AgentConfig, TurnLimit
from model_library.agent.metadata import (
    AgentTurn,
    ErrorTurn,
    SerializableException,
    ToolCallRecord,
)
from model_library.agent.tool import Tool, ToolOutput
from model_library.atif import (
    ATIFAgent,
    ATIFMetrics,
    ATIFFinalMetrics,
    ATIFObservationResult,
    ATIFObservation,
    ATIFToolCall,
    ATIFStep,
    ATIFTrajectory,
)
from model_library.base.gateway import GatewayLLM
from model_library.base.input import (
    InputItem,
    RawResponse,
    SystemInput,
    TextInput,
    ToolCall,
    ToolResult,
)
from model_library.base.output import (
    FinishReason,
    FinishReasonInfo,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
)


@pytest.mark.unit
class TestATIFModels:
    def test_trajectory_construction(self):
        trajectory = ATIFTrajectory(
            session_id="test-session-123",
            agent=ATIFAgent(name="test-agent", version="1.0", model_name="gpt-4"),
            steps=[],
            final_metrics=ATIFFinalMetrics(
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_steps=0,
            ),
        )
        assert trajectory.schema_version == "ATIF-v1.6"
        assert trajectory.session_id == "test-session-123"
        assert trajectory.steps == []

    def test_step_with_tool_calls_and_observation(self):
        step = ATIFStep(
            step_id=1,
            timestamp="2026-04-16T12:00:00Z",
            source="agent",
            message="Let me search for that.",
            tool_calls=[
                ATIFToolCall(
                    tool_call_id="call_1",
                    function_name="web_search",
                    arguments={"query": "weather"},
                )
            ],
            observation=ATIFObservation(
                results=[
                    ATIFObservationResult(
                        source_call_id="call_1",
                        content="Sunny, 72F",
                    )
                ]
            ),
            metrics=ATIFMetrics(
                prompt_tokens=100, completion_tokens=20, cost_usd=0.001
            ),
        )
        assert step.step_id == 1
        assert step.source == "agent"
        assert step.tool_calls is not None
        assert len(step.tool_calls) == 1
        assert step.observation is not None
        assert step.observation.results[0].content == "Sunny, 72F"

    def test_trajectory_serialization_excludes_none(self):
        trajectory = ATIFTrajectory(
            session_id="s1",
            agent=ATIFAgent(name="a", version="1", model_name="m"),
            steps=[
                ATIFStep(
                    step_id=1,
                    timestamp="2026-04-16T12:00:00Z",
                    source="user",
                    message="Hello",
                )
            ],
            final_metrics=ATIFFinalMetrics(
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_steps=1,
            ),
        )
        d = trajectory.to_json_dict()
        # None fields should be excluded
        assert "tool_calls" not in d["steps"][0]
        assert "observation" not in d["steps"][0]
        assert "metrics" not in d["steps"][0]
        assert "reasoning_content" not in d["steps"][0]
        assert "extra" not in d  # no extra set


@pytest.mark.unit
class TestAgentResultToATIF:
    """Tests for the agent_result_to_atif converter.

    These tests build AgentTurn/ErrorTurn lists that mirror the history
    sequences produced by Agent._run, same approach as test_docent.py.
    """

    def _make_turn(
        self,
        output_text: str | None,
        history: Sequence[InputItem] | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_records: list[ToolCallRecord] | None = None,
        reasoning: str | None = None,
        metadata: QueryResultMetadata | None = None,
    ) -> AgentTurn:
        return AgentTurn(
            timestamp="2026-04-16T12:00:00Z",
            query_result=QueryResult(
                output_text=output_text,
                reasoning=reasoning,
                tool_calls=tool_calls or [],
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
                history=list(history) if history else [],
                metadata=metadata or QueryResultMetadata(),
            ),
            tool_call_records=tool_records or [],
            duration_seconds=1.0,
        )

    def _make_tool_record(self, call_id: str, name: str, output: str) -> ToolCallRecord:
        return ToolCallRecord(
            tool_call=ToolCall(id=call_id, name=name, args={}),
            tool_output=ToolOutput(output=output),
            duration_seconds=0.5,
        )

    def test_single_turn_no_tools(self):
        """User question + agent text response = 2 steps."""
        question = TextInput(text="What is 2+2?")
        turn_history: list[InputItem] = [question, RawResponse(response={})]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn(
                "The answer is 4.",
                history=turn_history,
                metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
            )
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="test-agent",
            model_name="openai/gpt-4",
        )

        assert trajectory.schema_version == "ATIF-v1.6"
        assert trajectory.agent.name == "test-agent"
        assert trajectory.agent.model_name == "openai/gpt-4"

        # step 1: user, step 2: agent
        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].source == "user"
        assert trajectory.steps[0].message == "What is 2+2?"
        assert trajectory.steps[1].source == "agent"
        assert trajectory.steps[1].message == "The answer is 4."
        assert trajectory.steps[1].metrics is not None
        assert trajectory.steps[1].metrics.prompt_tokens == 10
        assert trajectory.steps[1].metrics.completion_tokens == 5

        assert trajectory.final_metrics.total_prompt_tokens == 10
        assert trajectory.final_metrics.total_completion_tokens == 5
        assert trajectory.final_metrics.total_steps == 2

    def test_agent_step_coerces_absent_output_text_to_empty_message(self):
        question = TextInput(text="Use a tool")
        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn(
                None,
                history=[question, RawResponse(response={})],
                tool_calls=[ToolCall(id="call_1", name="search", args={})],
            )
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        assert trajectory.steps[1].source == "agent"
        assert trajectory.steps[1].message == ""

    def test_multi_turn_with_tools(self):
        """Agent calls a tool, gets observation, responds = agent step has tool_calls + observation."""
        question = TextInput(text="Find X")
        tool_call = ToolCall(id="call_1", name="search", args={"q": "test"})
        tool_record = self._make_tool_record("call_1", "search", "Found results")

        turn_1_history: list[InputItem] = [question, RawResponse(response={})]
        turn_2_history: list[InputItem] = [
            question,
            RawResponse(response={}),
            ToolResult(tool_call=tool_call, result="Found results"),
            RawResponse(response={}),
        ]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn(
                "Let me search.",
                history=turn_1_history,
                tool_calls=[tool_call],
                tool_records=[tool_record],
            ),
            self._make_turn("The answer is X.", history=turn_2_history),
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        # step 1: user, step 2: agent with tool call + observation, step 3: agent final
        assert len(trajectory.steps) == 3
        assert trajectory.steps[0].source == "user"
        assert trajectory.steps[1].source == "agent"
        assert trajectory.steps[1].tool_calls is not None
        assert len(trajectory.steps[1].tool_calls) == 1
        assert trajectory.steps[1].tool_calls[0].function_name == "search"
        assert trajectory.steps[1].observation is not None
        assert trajectory.steps[1].observation.results[0].content == "Found results"
        assert trajectory.steps[2].source == "agent"
        assert trajectory.steps[2].message == "The answer is X."

    def test_error_turn(self):
        """ErrorTurn becomes a system step with error in extra."""
        question = TextInput(text="Do something")
        error_turn = ErrorTurn(
            timestamp="2026-04-16T12:00:00Z",
            error=SerializableException(type="RuntimeError", message="API timeout"),
            duration_seconds=2.0,
        )
        recovery_history: list[InputItem] = [question, RawResponse(response={})]
        recovery_turn = self._make_turn("Recovered.", history=recovery_history)

        turns: list[AgentTurn | ErrorTurn] = [error_turn, recovery_turn]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        # first turn is ErrorTurn (no history) → no initial steps extracted
        # step 1: system (error), step 2: agent (recovery)
        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].source == "system"
        assert "API timeout" in trajectory.steps[0].message
        assert trajectory.steps[0].extra is not None
        assert trajectory.steps[0].extra["error_type"] == "RuntimeError"

    def test_reasoning_content(self):
        """Reasoning from thinking models populates reasoning_content."""
        question = TextInput(text="Think hard")
        turn_history: list[InputItem] = [question, RawResponse(response={})]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn(
                "The answer.",
                history=turn_history,
                reasoning="Let me think step by step...",
            )
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="claude-3.5-sonnet",
        )

        assert trajectory.steps[1].reasoning_content == "Let me think step by step..."

    def test_raw_response_not_exported_in_extra(self):
        """Provider raw responses stay in history and are not duplicated in ATIF extra."""
        question = TextInput(text="Hi")
        turn_history: list[InputItem] = [
            question,
            RawResponse(response={"id": "chatcmpl-123", "object": "chat.completion"}),
        ]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn("Hello!", history=turn_history)
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        assert trajectory.steps[1].extra is None

    def test_cost_aggregation(self):
        """Final metrics aggregate cost across turns."""
        question = TextInput(text="Q")
        turn_history: list[InputItem] = [question, RawResponse(response={})]

        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            cost=QueryResultCost(input=0.001, output=0.002),
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            cost=QueryResultCost(input=0.002, output=0.004),
        )

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn("A1", history=turn_history, metadata=meta1),
            self._make_turn("A2", history=turn_history, metadata=meta2),
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        assert trajectory.final_metrics.total_prompt_tokens == 300
        assert trajectory.final_metrics.total_completion_tokens == 150
        assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.009)

    def test_system_input_in_extra(self):
        """SystemInput becomes a source='system' step extracted from history."""
        system = SystemInput(text="Be concise.")
        question = TextInput(text="What is AI?")
        turn_history: list[InputItem] = [system, question, RawResponse(response={})]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn("AI is...", history=turn_history)
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        assert trajectory.extra is None
        assert trajectory.steps[0].source == "system"
        assert trajectory.steps[0].message == "Be concise."
        assert trajectory.steps[1].source == "user"
        assert trajectory.steps[1].message == "What is AI?"

    def test_to_json_dict_roundtrip(self):
        """to_json_dict produces valid JSON-serializable dict."""
        question = TextInput(text="Hi")
        turn_history: list[InputItem] = [question, RawResponse(response={})]

        turns: list[AgentTurn | ErrorTurn] = [
            self._make_turn("Hello!", history=turn_history)
        ]

        trajectory = ATIFTrajectory.from_agent_result(
            turns=turns,
            agent_name="agent",
            model_name="gpt-4",
        )

        d = trajectory.to_json_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "ATIF-v1.6"
        assert len(parsed["steps"]) == 2


class DoneTool(Tool):
    name = "done"
    description = "Signal completion"
    parameters = {"answer": {"type": "string"}}
    required = ["answer"]

    async def execute(self, args, state, logger):
        return ToolOutput(output=args["answer"], done=True)


@pytest.mark.unit
class TestAgentATIFExport:
    async def test_atif_file_written(self, tmp_path):
        """Agent.run() with atif_export=True writes trajectory_atif.json."""
        mock_llm = AsyncMock()
        mock_llm.model_name = "openai/gpt-4"
        mock_llm.query = AsyncMock(
            return_value=QueryResult(
                output_text="The answer is 42.",
                tool_calls=[ToolCall(id="call_1", name="done", args={"answer": "42"})],
                finish_reason=FinishReasonInfo(
                    reason=FinishReason.TOOL_CALLS, raw="tool_calls"
                ),
                metadata=QueryResultMetadata(in_tokens=50, out_tokens=20),
                history=[TextInput(text="What?"), RawResponse(response={})],
            )
        )
        mock_llm.serialize_input = lambda x: b""

        agent = Agent(
            llm=mock_llm,
            tools=[DoneTool()],
            name="test",
            log_dir=tmp_path,
            config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
        )

        result = await agent.run(
            input=[TextInput(text="What?")],
            question_id="q1",
            atif_export=True,
        )

        atif_path = result.output_dir / "trajectory_atif.json"
        assert atif_path.exists()

        data = json.loads(atif_path.read_text())
        assert data["schema_version"] == "ATIF-v1.6"
        assert data["agent"]["model_name"] == "openai/gpt-4"
        assert len(data["steps"]) >= 2  # user + agent

    async def test_atif_file_written_after_gateway_metadata_sync(self, tmp_path):
        """Agent.run() syncs GatewayLLM metadata before ATIF export reads it."""
        llm = GatewayLLM("gpt-4o-mini", "openai")
        llm.query = AsyncMock(
            return_value=QueryResult(
                output_text=None,
                tool_calls=[ToolCall(id="call_1", name="done", args={"answer": "42"})],
                finish_reason=FinishReasonInfo(
                    reason=FinishReason.TOOL_CALLS, raw="tool_calls"
                ),
                metadata=QueryResultMetadata(in_tokens=50, out_tokens=20),
                history=[TextInput(text="What?"), RawResponse(response={})],
            )
        )

        async def sync_metadata() -> None:
            object.__setattr__(llm, "_gateway_metadata_loaded", True)

        llm.sync_model_metadata = AsyncMock(side_effect=sync_metadata)
        agent = Agent(
            llm=llm,
            tools=[DoneTool()],
            name="test",
            log_dir=tmp_path,
            config=AgentConfig(turn_limit=TurnLimit(max_turns=5), time_limit=None),
        )

        result = await agent.run(
            input=[TextInput(text="What?")],
            question_id="q1",
            atif_export=True,
        )

        llm.sync_model_metadata.assert_awaited_once()
        atif_path = result.output_dir / "trajectory_atif.json"
        assert atif_path.exists()
        data = json.loads(atif_path.read_text())
        assert data["agent"]["model_name"] == "gpt-4o-mini"

    async def test_atif_not_written_by_default(self, tmp_path):
        """Agent.run() without atif_export does NOT write trajectory_atif.json."""
        mock_llm = AsyncMock()
        mock_llm.model_name = "openai/gpt-4"
        mock_llm.query = AsyncMock(
            return_value=QueryResult(
                output_text="done",
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
                metadata=QueryResultMetadata(),
                history=[TextInput(text="Hi"), RawResponse(response={})],
            )
        )
        mock_llm.serialize_input = lambda x: b""

        agent = Agent(
            llm=mock_llm,
            tools=[DoneTool()],
            name="test",
            log_dir=tmp_path,
            config=AgentConfig(turn_limit=TurnLimit(max_turns=1), time_limit=None),
        )

        result = await agent.run(
            input=[TextInput(text="Hi")],
            question_id="q2",
        )

        atif_path = result.output_dir / "trajectory_atif.json"
        assert not atif_path.exists()
