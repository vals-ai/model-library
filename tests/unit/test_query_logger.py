"""Unit tests for custom logger passed at LLM and Agent init"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock


from model_library.agent import Agent, AgentConfig, Tool, ToolOutput
from model_library.base import LLM, QueryResult
from model_library.base.input import TextInput
from model_library.base.output import QueryResultCost, QueryResultMetadata

_cfg = AgentConfig(turn_limit=None, time_limit=None)


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
    """Without custom logger, LLM instance logger parents off the 'llm' root."""
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    assert llm.instance_logger.name.startswith("llm.")


async def test_default_llm_logger_uses_llm_root():
    """Without custom logger, query logger parents off 'llm.<provider>.<model><run=...>'."""
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")

    query_impl_mock = AsyncMock(return_value=QueryResult(output_text="ok"))
    llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    await llm.query("Test input")

    call_kwargs = query_impl_mock.call_args.kwargs
    query_logger = call_kwargs["query_logger"]
    assert query_logger.name.startswith("llm.openai.gpt-4o")
    assert "<run=" in query_logger.name
    assert "<question=" in query_logger.name
    assert "<query=" in query_logger.name


async def test_query_child_logger_derives_from_custom_logger():
    """Custom logger passed to query() becomes the parent of the query logger."""
    custom = logging.getLogger("myapp")
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")

    query_impl_mock = AsyncMock(return_value=QueryResult(output_text="ok"))
    llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    await llm.query("Test input", logger=custom)

    call_kwargs = query_impl_mock.call_args.kwargs
    query_logger = call_kwargs["query_logger"]
    assert query_logger.name.startswith("myapp.")
    assert "<question=" in query_logger.name
    assert "<query=" in query_logger.name


async def test_query_completion_logging_uses_start_level_when_logger_changes(caplog):
    custom = logging.getLogger("test.direct.level_change")
    caplog.set_level(logging.DEBUG)
    custom.setLevel(logging.INFO)
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")

    async def query_impl(*_args: Any, **_kwargs: Any) -> QueryResult:
        custom.setLevel(logging.DEBUG)
        return QueryResult(output_text="x" * 1000)

    llm._query_impl = AsyncMock(side_effect=query_impl)  # pyright: ignore[reportPrivateUsage]

    try:
        await llm.query("Test input", logger=custom)
    finally:
        custom.setLevel(logging.NOTSET)

    records = [
        record
        for record in caplog.records
        if record.name.startswith("test.direct.level_change")
    ]
    messages = [record.getMessage() for record in records]
    assert any(message.startswith("Query completed:") for message in messages)
    assert not any(record.levelno == logging.DEBUG for record in records)


# ── Agent logger tests ────────────────────────────────────────────────


def _agent_mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.ensure_metadata_loaded = AsyncMock(return_value=None)
    return llm


def _capture_logger(captured: list[logging.Logger]):
    def _capture(*_args: Any, **kwargs: Any) -> QueryResult:
        logger = kwargs.get("logger")
        assert isinstance(logger, logging.Logger)
        captured.append(logger)
        return _make_qr()

    return _capture


async def test_agent_default_logger_uses_agent_root():
    """Without custom logger, the logger passed to llm.query() parents off 'agent'."""
    llm = _agent_mock_llm()
    llm.model_name = "gpt-4o"
    captured: list[logging.Logger] = []
    llm.query = AsyncMock(side_effect=_capture_logger(captured))
    agent = Agent(name="eval", llm=llm, tools=[], config=_cfg)
    await agent.run([TextInput(text="go")], question_id="q1")
    assert captured and captured[0].name.startswith("agent.")


async def test_agent_custom_logger_is_parent():
    """Custom logger passed to run() becomes the parent of the query logger."""
    custom = logging.getLogger("myapp.cloudwatch")
    llm = _agent_mock_llm()
    llm.model_name = "gpt-4o"
    captured: list[logging.Logger] = []
    llm.query = AsyncMock(side_effect=_capture_logger(captured))
    agent = Agent(name="eval", llm=llm, tools=[], config=_cfg)
    await agent.run([TextInput(text="go")], question_id="q1", logger=custom)
    assert captured and captured[0].name.startswith("myapp.cloudwatch.")
    assert "eval<gpt-4o>" in captured[0].name


async def test_agent_custom_logger_propagates_to_llm():
    """Custom logger passed to run() is the ancestor of the logger sent to llm.query()."""
    custom = logging.getLogger("myapp")
    llm = _agent_mock_llm()
    llm.model_name = "gpt-4o"
    llm.run_id = "run-xyz"
    captured: list[logging.Logger] = []
    llm.query = AsyncMock(side_effect=_capture_logger(captured))
    agent = Agent(name="eval", llm=llm, tools=[], config=_cfg)
    await agent.run([TextInput(text="go")], question_id="q1", logger=custom)
    assert captured and captured[0].name.startswith("myapp.")
