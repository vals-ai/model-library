"""Docent ingestion converters for model-library results.

Converts QueryResult and AgentTurn objects into Docent AgentRun objects
for ingestion into Docent collections.

Requires the optional `docent` extra: `uv run pip install model-library[docent]` or `make install`
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ChatMessage,
    ContentReasoning,
    ContentText,
    SystemMessage,
    ToolCall as DocentToolCall,
    ToolMessage,
    UserMessage,
)

from model_library.agent.metadata import AgentTurn, ErrorTurn
from model_library.base.input import (
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolResult,
)
from model_library.base.output import QueryResult


def input_to_docent_messages(input: Sequence[InputItem]) -> list[ChatMessage]:
    """Convert input into Docent ChatMessages.

    Accepts either a string or a sequence of InputItems:
    - SystemInput → SystemMessage
    - TextInput → UserMessage
    - FileInput → UserMessage with placeholder (Docent doesn't support files yet)
    - ToolResult → ToolMessage
    - RawResponse / RawInput → skipped (provider-specific, not parseable uniformly, handled outputs by parsing QueryResult)
    """
    messages: list[ChatMessage] = []

    for item in input:
        match item:
            case SystemInput():
                messages.append(SystemMessage(content=item.text))

            case TextInput():
                messages.append(UserMessage(content=item.text))

            case ToolResult():
                messages.append(
                    ToolMessage(
                        content=str(item.result),
                        tool_call_id=item.tool_call.id,
                        function=item.tool_call.name,
                    )
                )

            case FileWithBase64() | FileWithUrl() | FileWithId():
                messages.append(
                    UserMessage(content=f"[{item.type}: {item.name} ({item.mime})]")
                )

            case RawResponse() | RawInput():
                pass

    return messages


def query_result_to_docent_messages(result: QueryResult) -> list[ChatMessage]:
    """Convert a single QueryResult's normalized outputs into Docent ChatMessages.

    Extracts assistant response, reasoning, and tool calls from the QueryResult.
    Does not parse RawResponse or history — uses only the uniform output fields.
    """
    content: list[ContentReasoning | ContentText] = []

    if result.reasoning:
        content.append(ContentReasoning(reasoning=result.reasoning))

    content.append(ContentText(text=result.output_text_str))

    assistant_msg = AssistantMessage(content=content)

    if result.tool_calls:
        assistant_msg.tool_calls = [
            DocentToolCall(
                id=tc.id,
                function=tc.name,
                arguments=tc.parsed_args
                if tc.parsed_args is not None
                else {"raw_arguments": tc.args},
                type="function",
            )
            for tc in result.tool_calls
        ]

    return [assistant_msg]


def query_result_to_docent_agent_run(
    input: Sequence[InputItem],
    result: QueryResult,
    question_id: str,
    metadata: dict[str, Any] | None = None,
) -> AgentRun:
    """Convert a single LLM query into a Docent AgentRun.

    For platform live_query/SDK runs where each test is a single LLM call.
    The question_id is used as the Docent AgentRun ID.
    """
    messages = [
        *input_to_docent_messages(input),
        *query_result_to_docent_messages(result),
    ]

    return AgentRun(
        transcripts=[Transcript(messages=messages)],
        metadata={
            "question_id": question_id,
            "model_metadata": result.metadata.model_dump(),
            **(metadata or {}),
        },
    )


def _extract_hook_injected_messages(
    history: Sequence[InputItem],
) -> list[ChatMessage]:
    """Extract hook-injected messages from a turn's history.

    Providers append a trailing RawResponse to history (the model's response).
    Hook-injected messages (turn_message, time_message) sit between the previous
    turn's ToolResult/RawResponse and this trailing RawResponse.

    Walk backwards from the second-to-last item (skipping the trailing
    RawResponse) and collect items until hitting a ToolResult or RawResponse
    from a previous turn.
    """
    injected: list[InputItem] = []

    for item in reversed(history[:-1]):
        if isinstance(item, (RawResponse, ToolResult)):
            break

        injected.append(item)

    injected.reverse()

    return input_to_docent_messages(injected)


def agent_turns_to_docent_agent_run(
    turns: Sequence[AgentTurn | ErrorTurn],
    question_id: str,
    final_answer: str,
    metadata: dict[str, Any] | None = None,
) -> AgentRun:
    """Convert a list of AgentTurns into a Docent AgentRun.

    For agentic runs via Agent.run. Walks through turns in order,
    extracting assistant responses and tool results from each turn.
    ErrorTurns are represented as assistant messages with the error.

    Hook-injected messages (turn_message, time_message) are extracted from
    each AgentTurn's query_result.history — they appear as trailing items
    after the last RawResponse/ToolResult block.
    """
    messages: list[ChatMessage] = []

    for turn in turns:
        if isinstance(turn, ErrorTurn):
            messages.append(AssistantMessage(content=f"[error: {turn.error.message}]"))
            continue

        messages.extend(_extract_hook_injected_messages(turn.query_result.history))
        messages.extend(query_result_to_docent_messages(turn.query_result))

        for record in turn.tool_call_records:
            messages.append(
                ToolMessage(
                    content=record.tool_output.output,
                    tool_call_id=record.tool_call.id,
                    function=record.tool_call.name,
                )
            )

    return AgentRun(
        transcripts=[Transcript(messages=messages)],
        metadata={
            "question_id": question_id,
            "final_answer": final_answer,
            **(metadata or {}),
        },
    )


_client: Docent | None = None


def _get_client() -> Docent:
    global _client

    if _client is None:
        from model_library import model_library_settings

        _client = Docent(api_key=model_library_settings.DOCENT_API_KEY)

    return _client


def ingest(run_id: str, agent_run: AgentRun) -> None:
    """Ingest a single AgentRun into a Docent collection.

    Deletes any existing agent runs with the same question_id before ingesting,
    so re-runs always reflect the latest attempt.
    """
    client = _get_client()
    question_id = agent_run.metadata.get("question_id") if agent_run.metadata else None

    if question_id:
        escaped_question_id = question_id.replace("'", "''")

        existing_ids = client.select_agent_run_ids(
            run_id,
            where_clause=f"metadata_json->>'question_id' = '{escaped_question_id}'",
        )

        if existing_ids:
            client.delete_agent_runs(run_id, existing_ids)

    client.add_agent_runs(run_id, [agent_run])
