"""Tests for Docent ingestion converters."""

from collections.abc import Sequence

import pytest

pytest.importorskip("docent")

from docent.data_models import AgentRun
from docent.data_models.chat import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

from model_library.agent.metadata import AgentTurn, ErrorTurn, ToolCallRecord
from model_library.agent.tool import ToolOutput
from model_library.base.input import (
    FileWithBase64,
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
    QueryResultMetadata,
)
from model_library.docent import (
    agent_turns_to_docent_agent_run,
    input_to_docent_messages,
    query_result_to_docent_agent_run,
    query_result_to_docent_messages,
)
from model_library.agent.metadata import SerializableException


# --- input_to_docent_messages ---


@pytest.mark.unit
class TestInputToDocentMessages:
    def test_system_input(self):
        messages = input_to_docent_messages([SystemInput(text="You are helpful.")])

        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are helpful."

    def test_text_input(self):
        messages = input_to_docent_messages([TextInput(text="What is 2+2?")])

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "What is 2+2?"

    def test_tool_result(self):
        tool_call = ToolCall(id="call_1", name="search", args={"query": "test"})
        tool_result = ToolResult(tool_call=tool_call, result="Found 3 results")
        messages = input_to_docent_messages([tool_result])

        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "Found 3 results"
        assert messages[0].tool_call_id == "call_1"
        assert messages[0].function == "search"

    def test_file_input_placeholder(self):
        file_input = FileWithBase64(
            type="image", name="photo.png", mime="image/png", base64="abc123"
        )
        messages = input_to_docent_messages([file_input])

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "[image: photo.png (image/png)]"

    def test_raw_response_skipped(self):
        messages = input_to_docent_messages([RawResponse(response={"some": "data"})])

        assert len(messages) == 0

    def test_raw_input_skipped(self):
        from model_library.base.input import RawInput

        messages = input_to_docent_messages([RawInput(input={"some": "data"})])

        assert len(messages) == 0


# --- query_result_to_docent_messages ---


@pytest.mark.unit
class TestQueryResultToDocentMessages:
    def test_simple_text_response(self):
        from docent.data_models.chat import ContentText

        result = QueryResult(output_text="The answer is 4.")
        messages = query_result_to_docent_messages(result)

        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert isinstance(messages[0].content, list)
        assert isinstance(messages[0].content[0], ContentText)
        assert messages[0].content[0].text == "The answer is 4."

    def test_empty_output(self):
        from docent.data_models.chat import ContentText

        result = QueryResult(output_text=None)
        messages = query_result_to_docent_messages(result)

        assert len(messages) == 1
        assert isinstance(messages[0].content, list)
        assert isinstance(messages[0].content[0], ContentText)
        assert messages[0].content[0].text == ""

    def test_with_reasoning(self):
        from docent.data_models.chat import ContentReasoning, ContentText

        result = QueryResult(
            output_text="The answer is 4.",
            reasoning="Let me think step by step...",
        )
        messages = query_result_to_docent_messages(result)

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, AssistantMessage)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], ContentReasoning)
        assert msg.content[0].reasoning == "Let me think step by step..."
        assert isinstance(msg.content[1], ContentText)
        assert msg.content[1].text == "The answer is 4."

    def test_with_tool_calls(self):
        result = QueryResult(
            output_text="Let me search for that.",
            tool_calls=[
                ToolCall(id="call_1", name="web_search", args={"query": "weather"}),
            ],
        )
        messages = query_result_to_docent_messages(result)

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, AssistantMessage)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function == "web_search"
        assert msg.tool_calls[0].arguments == {"query": "weather"}

    def test_with_unparseable_tool_call_args(self):
        result = QueryResult(
            output_text="",
            tool_calls=[
                ToolCall(id="call_1", name="search", args="not valid json {{{"),
            ],
        )
        messages = query_result_to_docent_messages(result)

        msg = messages[0]
        assert isinstance(msg, AssistantMessage)
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].arguments["raw_arguments"] == "not valid json {{{"


# --- query_result_to_docent_agent_run ---


@pytest.mark.unit
class TestQueryResultToDocentAgentRun:
    def test_basic(self):
        result = QueryResult(
            output_text="The answer is 4.",
            metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
        )
        agent_run = query_result_to_docent_agent_run([TextInput(text="What is 2+2?")], result, "q001")

        assert isinstance(agent_run, AgentRun)
        assert len(agent_run.transcripts) == 1
        assert len(agent_run.transcripts[0].messages) == 2
        assert agent_run.metadata["question_id"] == "q001"
        assert "model_metadata" in agent_run.metadata

    def test_with_extra_metadata(self):
        result = QueryResult(output_text="Yes.")
        agent_run = query_result_to_docent_agent_run(
            [TextInput(text="Is this a test?")], result, "q003", metadata={"custom": "value"}
        )

        assert agent_run.metadata["custom"] == "value"
        assert agent_run.metadata["question_id"] == "q003"


# --- agent_turns_to_docent_agent_run ---


@pytest.mark.unit
class TestAgentTurnsToDocentAgentRun:
    def _make_turn(
        self,
        output_text: str,
        history: Sequence[InputItem] | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_records: list[ToolCallRecord] | None = None,
    ) -> AgentTurn:
        return AgentTurn(
            query_result=QueryResult(
                output_text=output_text,
                tool_calls=tool_calls or [],
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
                history=history or [],
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

    def test_single_turn_no_tools_no_hooks(self):
        """Single turn, no tools, no hooks.

        _run history flow:
          history = [TextInput("question")]
          query → response.history = [TextInput, RawResponse]
          no tool calls → loop ends
        """
        question = TextInput(text="What is the meaning of life?")
        turn_1_history: list[InputItem] = [question, RawResponse(response={})]

        turns = [self._make_turn("The answer is 42.", history=turn_1_history)]
        agent_run = agent_turns_to_docent_agent_run(turns, "q001", "42")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 2
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "What is the meaning of life?"
        assert isinstance(messages[1], AssistantMessage)
        assert agent_run.metadata["final_answer"] == "42"

    def test_multi_turn_with_tools_no_hooks(self):
        """Two turns with a tool call, no hooks.

        _run history flow:
          Turn 1: history = [TextInput("Find X")]
            query → response.history = [TextInput, RawResponse]
            _execute_tool_calls appends ToolResult → history = [..., RawResponse, ToolResult]
          Turn 2: history = [TextInput, RawResponse, ToolResult]
            query → response.history = [TextInput, RawResponse, ToolResult, RawResponse]
        """
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

        turns = [
            self._make_turn(
                "Let me search.",
                history=turn_1_history,
                tool_calls=[tool_call],
                tool_records=[tool_record],
            ),
            self._make_turn("The answer is X.", history=turn_2_history),
        ]
        agent_run = agent_turns_to_docent_agent_run(turns, "q002", "X")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 4
        assert isinstance(messages[0], UserMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].content == "Found results"
        assert messages[2].tool_call_id == "call_1"
        assert isinstance(messages[3], AssistantMessage)

    def test_error_turn_then_recovery(self):
        """Error on turn 1, recovery on turn 2.

        _run history flow:
          Turn 1: history = [TextInput("Do something")]
            query fails → ErrorTurn, history NOT replaced (no RawResponse added)
          Turn 2: before_query runs (re-raises by default, but we assume handled)
            history still = [TextInput("Do something")]
            query → response.history = [TextInput, RawResponse]
        """
        question = TextInput(text="Do something")
        error_turn = ErrorTurn(
            error=SerializableException(type="RuntimeError", message="API timeout"),
            duration_seconds=2.0,
        )

        # recovery turn sees original input (no RawResponse from failed turn)
        turn_2_history: list[InputItem] = [question, RawResponse(response={})]
        normal_turn = self._make_turn("Recovered answer.", history=turn_2_history)

        turns = [error_turn, normal_turn]
        agent_run = agent_turns_to_docent_agent_run(turns, "q003", "Recovered answer.")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 3
        assert isinstance(messages[0], AssistantMessage)
        assert "[error: API timeout]" in messages[0].content
        assert isinstance(messages[1], UserMessage)
        assert messages[1].content == "Do something"
        assert isinstance(messages[2], AssistantMessage)

    def test_hook_injected_messages(self):
        """Hook messages on both turn 1 (no boundary) and turn 2 (after ToolResult).

        _run history flow:
          Turn 1: history = [TextInput("Find X")]
            turn_message appends → [..., TextInput("[Turn 1/5]")]
            query → response.history = [TextInput, TextInput("[Turn 1/5]"), RawResponse]
            _execute_tool_calls → [..., RawResponse, ToolResult]
          Turn 2: turn_message and time_message append
            history = [..., ToolResult, TextInput("[Turn 2/5]"), TextInput("[30s remaining]")]
            query → response.history = [..., TextInput("[30s remaining]"), RawResponse]
        """
        question = TextInput(text="Find X")
        tool_call = ToolCall(id="call_1", name="search", args={"q": "test"})
        tool_record = self._make_tool_record("call_1", "search", "Found results")

        turn_1_history: list[InputItem] = [
            question,
            TextInput(text="[Turn 1/5]"),
            RawResponse(response={}),
        ]
        turn_2_history: list[InputItem] = [
            question,
            TextInput(text="[Turn 1/5]"),
            RawResponse(response={}),
            ToolResult(tool_call=tool_call, result="Found results"),
            TextInput(text="[Turn 2/5]"),
            TextInput(text="[30s remaining]"),
            RawResponse(response={}),
        ]

        turns = [
            self._make_turn(
                "Let me search.",
                history=turn_1_history,
                tool_calls=[tool_call],
                tool_records=[tool_record],
            ),
            self._make_turn("The answer is X.", history=turn_2_history),
        ]
        agent_run = agent_turns_to_docent_agent_run(turns, "q001", "X")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 7
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Find X"
        assert isinstance(messages[1], UserMessage)
        assert messages[1].content == "[Turn 1/5]"
        assert isinstance(messages[2], AssistantMessage)
        assert isinstance(messages[3], ToolMessage)
        assert messages[3].content == "Found results"
        assert isinstance(messages[4], UserMessage)
        assert messages[4].content == "[Turn 2/5]"
        assert isinstance(messages[5], UserMessage)
        assert messages[5].content == "[30s remaining]"
        assert isinstance(messages[6], AssistantMessage)

    def test_system_input_preserved(self):
        """SystemInput at the start of history is preserved.

        _run history flow:
          Turn 1: history = [SystemInput("Be concise"), TextInput("question")]
            query → response.history = [SystemInput, TextInput, RawResponse]
        """
        system = SystemInput(text="Be concise.")
        question = TextInput(text="What is AI?")
        turn_1_history: list[InputItem] = [system, question, RawResponse(response={})]

        turns = [self._make_turn("AI is...", history=turn_1_history)]
        agent_run = agent_turns_to_docent_agent_run(turns, "q001", "AI is...")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be concise."
        assert isinstance(messages[1], UserMessage)
        assert isinstance(messages[2], AssistantMessage)

    def test_turn_without_tool_calls_then_another_turn(self):
        """Text-only turn followed by another turn (should_stop returned False).

        _run history flow:
          Turn 1: query → response.history = [TextInput, RawResponse]
            no tool calls, no ToolResult appended. should_stop returns False.
          Turn 2: history = [TextInput, RawResponse]
            turn_message appends → [..., RawResponse, TextInput("[Turn 2/5]")]
            query → response.history = [..., TextInput("[Turn 2/5]"), RawResponse]
            The boundary for extraction is the RawResponse from turn 1.
        """
        question = TextInput(text="Tell me more")
        turn_1_history: list[InputItem] = [question, RawResponse(response={})]
        turn_2_history: list[InputItem] = [
            question,
            RawResponse(response={}),
            TextInput(text="[Turn 2/5]"),
            RawResponse(response={}),
        ]

        turns = [
            self._make_turn("Here's some info.", history=turn_1_history),
            self._make_turn("Here's more info.", history=turn_2_history),
        ]
        agent_run = agent_turns_to_docent_agent_run(turns, "q001", "more info")

        messages = agent_run.transcripts[0].messages
        assert len(messages) == 4
        assert isinstance(messages[0], UserMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert isinstance(messages[2], UserMessage)
        assert messages[2].content == "[Turn 2/5]"
        assert isinstance(messages[3], AssistantMessage)

