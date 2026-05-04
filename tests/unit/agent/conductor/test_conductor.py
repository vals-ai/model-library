"""Unit tests for ConductorAgent."""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from model_library.agent import Agent
from model_library.agent.conductor import (
    ConductorAgent,
    ConductorConfig,
    ConductorStopReason,
)
from model_library.base.input import InputItem, SystemInput, TextInput

from tests.unit.agent.helpers import (
    DoneTool,
    make_agent,
    make_text_response,
    make_tool_call,
    make_tool_response,
    mock_llm,
)


def make_conductor(
    auditor: Agent,
    target: Agent,
    *,
    max_exchanges: int = 5,
    time_limit: Any = None,
    name: str = "test-conductor",
    auditor_system_prompt: SystemInput = SystemInput(text="test auditor prompt"),
    target_system_prompt: SystemInput = SystemInput(text="test target prompt"),
) -> ConductorAgent:
    config = ConductorConfig(max_exchanges=max_exchanges, time_limit=time_limit)

    return ConductorAgent(
        auditor=auditor,
        target=target,
        auditor_system_prompt=auditor_system_prompt,
        target_system_prompt=target_system_prompt,
        name=name,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuditorDoneMidway:
    """Auditor signals done after a few exchanges."""

    async def test_stops_when_auditor_signals_done(self):
        """Auditor sends a text reply on exchange 1, then signals done on exchange 2."""
        auditor_llm = mock_llm(
            make_text_response("question 1"),
            make_tool_response(
                [make_tool_call("submit", {"answer": "final-answer"})],
                output_text="submitting",
            ),
        )
        target_llm = mock_llm(
            make_text_response("target reply 1"),
        )

        auditor = make_agent(auditor_llm, tools=[DoneTool()], name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=3)

        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.AUDITOR_DONE

        # 3 messages: auditor, target, auditor (done) — target not called after done
        assert len(result.messages) == 3
        assert result.messages[0].role == "auditor"
        assert result.messages[0].result.final_answer == "question 1"
        assert result.messages[1].role == "target"
        assert result.messages[1].result.final_answer == "target reply 1"
        assert result.messages[2].role == "auditor"
        assert result.messages[2].result.final_answer == "final-answer"


class TestMaxExchangesReached:
    """Auditor never signals done; conductor stops at max_exchanges."""

    async def test_stops_at_max_exchanges(self):
        auditor_llm = mock_llm(
            make_text_response("q1"),
            make_text_response("q2"),
        )
        target_llm = mock_llm(
            make_text_response("a1"),
            make_text_response("a2"),
        )

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=2)

        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.MAX_EXCHANGES
        assert len(result.messages) == 4  # 2 auditor + 2 target


class TestAuditorDoneOnFirstExchange:
    """Auditor signals done immediately — target never runs."""

    async def test_single_message(self):
        auditor_llm = mock_llm(
            make_tool_response(
                [make_tool_call("submit", {"answer": "immediate-answer"})],
                output_text="done right away",
            ),
        )
        target_llm = mock_llm()  # should not be called

        auditor = make_agent(auditor_llm, tools=[DoneTool()], name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=5)

        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.AUDITOR_DONE
        assert len(result.messages) == 1
        assert result.messages[0].role == "auditor"
        assert result.messages[0].result.final_answer == "immediate-answer"

        # Target LLM was never called
        target_llm.query.assert_not_called()


class TestHistoryAccumulates:
    """Both agents' histories grow across exchanges within a single conductor run."""

    async def test_auditor_receives_target_reply_as_input(self):
        auditor_inputs_received: list[list[InputItem]] = []
        target_inputs_received: list[list[InputItem]] = []

        original_run = Agent.run

        async def capture_auditor_run(self_agent, input, **kwargs):
            auditor_inputs_received.append(list(input))
            return await original_run(self_agent, input, **kwargs)

        async def capture_target_run(self_agent, input, **kwargs):
            target_inputs_received.append(list(input))
            return await original_run(self_agent, input, **kwargs)

        auditor_llm = mock_llm(
            make_text_response("auditor-q1"),
            make_text_response("auditor-q2"),
        )
        target_llm = mock_llm(
            make_text_response("target-a1"),
            make_text_response("target-a2"),
        )

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")

        auditor.run = lambda input, **kw: capture_auditor_run(auditor, input, **kw)  # type: ignore[assignment]
        target.run = lambda input, **kw: capture_target_run(target, input, **kw)  # type: ignore[assignment]

        conductor = make_conductor(auditor, target, max_exchanges=2)
        await conductor.run(question_id="q1")

        # Exchange 1: auditor receives system prompt + start prompt
        assert len(auditor_inputs_received[0]) == 2
        assert isinstance(auditor_inputs_received[0][0], SystemInput)
        assert auditor_inputs_received[0][0].text == "test auditor prompt"
        assert isinstance(auditor_inputs_received[0][1], TextInput)
        assert auditor_inputs_received[0][1].text == "Start the conversation. Send your first message in character."

        # Exchange 1: target receives system prompt + auditor's answer
        assert len(target_inputs_received[0]) == 2
        assert isinstance(target_inputs_received[0][0], SystemInput)
        assert target_inputs_received[0][0].text == "test target prompt"
        assert isinstance(target_inputs_received[0][1], TextInput)
        assert target_inputs_received[0][1].text == "auditor-q1"

        # Exchange 2: auditor receives prior history + target's reply
        # auditor_history (from final_history) + [TextInput(target reply)]
        assert isinstance(auditor_inputs_received[1][-1], TextInput)
        assert auditor_inputs_received[1][-1].text == "target-a1"

        # Exchange 2: target receives prior history + auditor's answer
        # target_history (from final_history) + [TextInput(auditor reply)]
        assert isinstance(target_inputs_received[1][-1], TextInput)
        assert target_inputs_received[1][-1].text == "auditor-q2"


class TestStatelessAgentFreshConversation:
    """Agent is stateless — each conductor run starts a fresh conversation automatically."""

    async def test_conductor_starts_fresh_conversation(self):
        auditor_llm = mock_llm(
            make_text_response("stale"),
            make_text_response("fresh-answer"),
        )
        target_llm = mock_llm(
            make_text_response("stale-target"),
            make_text_response("fresh-target"),
        )

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")

        # Run each agent independently before the conductor run
        await auditor.run([TextInput(text="old-input")], question_id="old")
        await target.run([TextInput(text="old-target-input")], question_id="old")

        # Agent is stateless, so conductor run starts fresh regardless
        conductor = make_conductor(auditor, target, max_exchanges=1)
        result = await conductor.run(question_id="q2")

        assert len(result.messages) == 2
        assert result.messages[0].result.final_answer == "fresh-answer"
        assert result.messages[1].result.final_answer == "fresh-target"


class TestErrorStopCondition:
    """Conductor catches agent exceptions and returns ERROR stop reason."""

    async def test_auditor_exception_returns_error(self):
        auditor_llm = mock_llm()
        target_llm = mock_llm()

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")

        # Make auditor.run() itself raise (not just the LLM query)
        auditor.run = AsyncMock(side_effect=RuntimeError("auditor crashed"))  # type: ignore[assignment]

        conductor = make_conductor(auditor, target, max_exchanges=3)
        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.ERROR
        assert len(result.messages) == 0

    async def test_target_exception_returns_error(self):
        auditor_llm = mock_llm(make_text_response("auditor msg"))
        target_llm = mock_llm()

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")

        # Make target.run() itself raise
        target.run = AsyncMock(side_effect=RuntimeError("target crashed"))  # type: ignore[assignment]

        conductor = make_conductor(auditor, target, max_exchanges=3)
        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.ERROR
        # Auditor message was recorded before target crashed
        assert len(result.messages) == 1
        assert result.messages[0].role == "auditor"


class TestEmptyAuditorResponse:
    """Conductor stops with ERROR if auditor produces an empty response."""

    async def test_empty_auditor_answer_stops_conversation(self):
        """If auditor hits its own max turns and produces empty final_answer, conductor stops."""
        # LLM returns a tool call but no done signal — agent hits max_turns
        # and default_determine_answer returns "" since there's no done tool or text
        auditor_llm = mock_llm(
            make_tool_response([make_tool_call("echo", {"text": "thinking"})]),
            make_tool_response([make_tool_call("echo", {"text": "still thinking"})]),
        )
        target_llm = mock_llm(make_text_response("should not be called"))

        from model_library.agent import AgentConfig, TurnLimit

        auditor = make_agent(
            auditor_llm,
            tools=[],  # no done tool, no echo tool — tool calls will fail
            name="auditor",
            config=AgentConfig(turn_limit=TurnLimit(max_turns=1), time_limit=None),
        )
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=5)

        result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.ERROR
        # Auditor message is recorded (with empty answer), target never called
        assert len(result.messages) == 1
        assert result.messages[0].role == "auditor"
        assert result.messages[0].result.final_answer == ""
        target_llm.query.assert_not_called()


class TestTimeLimitStopCondition:
    """Conductor stops when the time limit is exceeded."""

    async def test_stops_at_time_limit(self):
        from model_library.agent.config import TimeLimit

        auditor_llm = mock_llm(
            make_text_response("q1"),
            make_text_response("q2"),
            make_text_response("q3"),
        )
        target_llm = mock_llm(
            make_text_response("a1"),
            make_text_response("a2"),
            make_text_response("a3"),
        )

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")

        conductor = make_conductor(
            auditor, target, max_exchanges=10, time_limit=TimeLimit(max_seconds=5)
        )

        # Patch only the conductor module's time.monotonic.
        # Advances 3s per call: start_time=3, check=6 (passes, 6-3=3 < 5),
        # check=9 (fails, 9-3=6 >= 5).
        call_count = 0

        def advancing_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            return call_count * 3.0

        with patch(
            "model_library.agent.conductor.conductor.time.monotonic",
            side_effect=advancing_monotonic,
        ):
            result = await conductor.run(question_id="q1")

        assert result.stop_reason == ConductorStopReason.MAX_TIME
        # First exchange completes (elapsed=3 < 5), second blocked (elapsed=6 >= 5)
        assert len(result.messages) == 2  # auditor + target from exchange 1


class TestLogDirectoryStructure:
    """Conductor writes the expected directory layout."""

    async def test_directory_layout(self):
        auditor_llm = mock_llm(
            make_text_response("q1"),
            make_tool_response(
                [make_tool_call("submit", {"answer": "done"})],
                output_text="submitting",
            ),
        )
        target_llm = mock_llm(
            make_text_response("a1"),
        )

        auditor = make_agent(auditor_llm, tools=[DoneTool()], name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=5)

        result = await conductor.run(question_id="q1")
        output_dir = result.output_dir

        # Init directory
        init_dir = output_dir / "exchanges" / "init"
        assert init_dir.exists()
        assert (init_dir / "config.json").exists()
        assert (init_dir / "state.json").exists()

        # Top-level files
        assert (output_dir / "result.json").exists()
        assert (output_dir / "transcript.json").exists()

        # Verify config.json content
        config_data = json.loads((init_dir / "config.json").read_text())
        assert config_data["conductor_config"]["max_exchanges"] == 5
        assert config_data["auditor"]["name"] == "auditor"
        assert config_data["target"]["name"] == "target"


class TestTranscriptContent:
    """transcript.json contains the conversation messages in order."""

    async def test_transcript_interleaving(self):
        auditor_llm = mock_llm(
            make_text_response("auditor-msg-1"),
            make_text_response("auditor-msg-2"),
        )
        target_llm = mock_llm(
            make_text_response("target-msg-1"),
            make_text_response("target-msg-2"),
        )

        auditor = make_agent(auditor_llm, name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=2)

        result = await conductor.run(question_id="q1")
        transcript = json.loads(
            (result.output_dir / "transcript.json").read_text()
        )

        assert len(transcript) == 4
        assert transcript[0] == {"role": "auditor", "content": "auditor-msg-1"}
        assert transcript[1] == {"role": "target", "content": "target-msg-1"}
        assert transcript[2] == {"role": "auditor", "content": "auditor-msg-2"}
        assert transcript[3] == {"role": "target", "content": "target-msg-2"}

    async def test_transcript_ends_on_auditor_done(self):
        """When auditor signals done, transcript ends with auditor message (no target)."""
        auditor_llm = mock_llm(
            make_tool_response(
                [make_tool_call("submit", {"answer": "done-msg"})],
            ),
        )
        target_llm = mock_llm()

        auditor = make_agent(auditor_llm, tools=[DoneTool()], name="auditor")
        target = make_agent(target_llm, name="target")
        conductor = make_conductor(auditor, target, max_exchanges=5)

        result = await conductor.run(question_id="q1")
        transcript = json.loads(
            (result.output_dir / "transcript.json").read_text()
        )

        assert len(transcript) == 1
        assert transcript[0] == {"role": "auditor", "content": "done-msg"}
