"""Unit tests for custom logger passed at LLM and Agent init"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from model_library.agent import Agent, Tool, ToolOutput
from model_library.base import LLM, QueryResult
from model_library.base.input import TextInput
from model_library.base.output import QueryResultCost, QueryResultMetadata


# ── Helpers ───────────────────────────────────────────────────────────


def _make_mock_llm_class():
    return type(
        "MockLLM",
        (LLM,),
        {
            "_get_default_api_key": Mock(return_value="mock_api_key"),
            "get_client": Mock(return_value=MagicMock()),
            "build_body": AsyncMock(return_value={}),
            "_query_impl": AsyncMock(return_value=QueryResult()),
            "parse_input": AsyncMock(return_value=None),
            "parse_image": AsyncMock(return_value=None),
            "parse_file": AsyncMock(return_value=None),
            "parse_tools": AsyncMock(return_value=None),
            "upload_file": AsyncMock(return_value=None),
        },
    )


class EchoTool(Tool):
    name = "echo"
    description = "Echo"
    parameters = {"text": {"type": "string"}}

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        return ToolOutput(output=args.get("text", ""))


def _make_qr(text: str = "ok") -> QueryResult:
    return QueryResult(
        output_text=text,
        metadata=QueryResultMetadata(
            in_tokens=10,
            out_tokens=5,
            cost=QueryResultCost(input=0.001, output=0.001),
        ),
        tool_calls=[],
        history=[TextInput(text="prompt")],
    )


# ── LLM logger tests ─────────────────────────────────────────────────


async def test_default_logger_uses_llm_root():
    """Without custom logger, LLM logger parents off the 'llm' root."""
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    assert llm.logger.name.startswith("llm.")


async def test_custom_logger_is_parent_of_llm_logger():
    """Custom logger becomes the parent; LLM appends provider.model<run=...>."""
    custom = logging.getLogger("myapp.cloudwatch")
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai", logger=custom)
    assert llm.logger.name.startswith("myapp.cloudwatch.")
    assert "openai.gpt-4o" in llm.logger.name


async def test_query_child_logger_derives_from_custom_logger():
    """The per-query logger created inside query() is a grandchild of the custom logger."""
    custom = logging.getLogger("myapp")
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai", logger=custom)

    query_impl_mock = AsyncMock(return_value=QueryResult(output_text="ok"))
    llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    await llm.query("Test input")

    call_kwargs = query_impl_mock.call_args.kwargs
    query_logger = call_kwargs["query_logger"]
    assert query_logger.name.startswith("myapp.")
    assert "<question=" in query_logger.name
    assert "<query=" in query_logger.name


# ── Agent logger tests ────────────────────────────────────────────────


async def test_agent_default_logger_uses_agent_root():
    """Without custom logger, Agent logger parents off the 'agent' root."""
    llm = MagicMock()
    llm.model_name = "gpt-4o"
    llm.run_id = "abc"
    agent = Agent(name="eval", llm=llm, tools=[])
    assert agent._logger.name.startswith("agent.")


async def test_agent_custom_logger_is_parent():
    """Custom logger becomes the parent; Agent appends agent.name<model>."""
    custom = logging.getLogger("myapp.cloudwatch")
    llm = MagicMock()
    llm.model_name = "gpt-4o"
    llm.run_id = "abc"
    agent = Agent(name="eval", llm=llm, tools=[], logger=custom)
    assert agent._logger.name.startswith("myapp.cloudwatch.")
    assert "eval<gpt-4o>" in agent._logger.name


async def test_agent_custom_logger_propagates_to_llm():
    """When Agent gets a custom logger, the LLM logger it sets is a child of it."""
    custom = logging.getLogger("myapp")
    llm = MagicMock()
    llm.model_name = "gpt-4o"
    llm.run_id = "run-xyz"
    Agent(name="eval", llm=llm, tools=[], logger=custom)
    # Agent sets llm.logger = self._logger.getChild(f"<run={llm.run_id}>")
    assert llm.logger.name.startswith("myapp.")
    assert "<run=run-xyz>" in llm.logger.name
