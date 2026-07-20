"""Shared test helpers for agent tests."""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import tiktoken

from model_library.agent import Agent, AgentConfig, Tool, ToolOutput
from model_library.base.base import LLM
from model_library.base.input import TextInput, ToolCall
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata

_cfg = AgentConfig(turn_limit=None, time_limit=None)


def make_metadata(
    in_tokens: int = 10, out_tokens: int = 5, cost_total: float = 0.01
) -> QueryResultMetadata:
    return QueryResultMetadata(
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        cost=QueryResultCost(input=cost_total / 2, output=cost_total / 2),
    )


def make_text_response(
    text: str, metadata: QueryResultMetadata | None = None
) -> QueryResult:
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


def make_tool_call(
    name: str = "echo", args: dict[str, Any] | str | None = None
) -> ToolCall:
    return ToolCall(id="tc_1", name=name, args=args or {"text": "hello"})


class _MockLLM(MagicMock):
    def __rich_repr__(self):
        yield "model_name", self.model_name
        yield "temperature", self.temperature
        yield "max_tokens", self.max_tokens


def mock_llm(*responses: QueryResult | Exception) -> MagicMock:
    """Create a mock LLM that returns the given responses in sequence."""

    llm = _MockLLM()
    llm.model_name = "mock-model"
    llm.max_tokens = None
    llm.query = AsyncMock(side_effect=list(responses))
    # Tokenizer used by default compaction estimate. Approximate cl100k.
    llm.get_encoding = AsyncMock(return_value=tiktoken.get_encoding("cl100k_base"))

    return llm


def make_agent(
    llm: LLM | MagicMock, tools: list[Tool] | None = None, **kwargs: Any
) -> Agent:
    kwargs.setdefault("name", "test")
    kwargs.setdefault("config", _cfg)

    return Agent(llm=cast(LLM, llm), tools=tools or [], **kwargs)


class DoneTool(Tool):
    name = "submit"
    description = "Submit final answer"
    parameters = {"answer": {"type": "string"}}

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        return ToolOutput(output=args["answer"], done=True)
