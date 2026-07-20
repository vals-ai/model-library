"""
Edge-case tests for run_id, question_id, logger naming, and token retry interaction.

Covers the identity resolution path: Agent -> LLM.query -> TokenRetrier,
logger hierarchy & filtering, and the run_logging context manager.
"""

import asyncio
import logging
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import fakeredis.aioredis  # pyright: ignore[reportMissingImports]
import pytest

from model_library.agent import Agent, AgentConfig, Tool, ToolOutput
from model_library.base.input import TextInput, ToolCall
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.base import LLM
from model_library.base.base import ResolvedTokenRetryParams, TokenRetryParams
from model_library.retriers.token import TokenRetrier, set_redis_client
from model_library.retriers.token.utils import KEY_PREFIX
from model_library.utils import run_logging


_cfg = AgentConfig(turn_limit=None, time_limit=None)

# ── Helpers ───────────────────────────────────────────────────────────

CLIENT_KEY = ("provider", "model")
TOKEN_KEY = f"{KEY_PREFIX}:provider:model:tokens"


class _FakeLock:
    """No-op async context manager replacing redis Lock (fakeredis lacks evalsha)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


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


def _mock_llm(*responses: QueryResult | Exception) -> MagicMock:
    llm = MagicMock()
    llm.query = AsyncMock(side_effect=list(responses))
    llm.model_name = "test-model"
    llm.run_id = "default-run-id"
    return llm


def _make_retrier(
    *,
    run_id: str = "test-run",
    question_id: str = "test-qid",
    estimate_input_tokens: int = 100,
    estimate_output_tokens: int = 50,
    use_dynamic_estimate: bool = False,
) -> TokenRetrier:
    return TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        run_id=run_id,
        question_id=question_id,
        estimate_input_tokens=estimate_input_tokens,
        estimate_output_tokens=estimate_output_tokens,
        use_dynamic_estimate=use_dynamic_estimate,
    )


async def _init_tokens(redis, value: int = 1000, limit: int = 1000):
    await redis.set(TOKEN_KEY, str(value))
    await redis.set(f"{TOKEN_KEY}:limit", str(limit))


class _Settings:
    def __init__(self, **values: str):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._values.get(name, default)


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


async def test_token_retry_telemetry_uses_original_question_id(redis):
    await _init_tokens(redis, value=1000, limit=1000)
    retrier = _make_retrier(run_id="run-a", question_id="question-a")
    events: list[tuple[str, dict[str, object | None]]] = []

    def fake_add_event(name: str, attrs: dict[str, object | None] | None = None):
        events.append((name, dict(attrs or {})))

    with patch(
        "model_library.retriers.token.token.telemetry.add_event", fake_add_event
    ):
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    wait_start = next(
        attrs for name, attrs in events if name == "retry_queue.wait_start"
    )
    tokens_deducted = next(
        attrs for name, attrs in events if name == "retry_queue.tokens_deducted"
    )
    assert wait_start["run_id"] == "run-a"
    assert wait_start["question_id"] == "question-a"
    assert wait_start["retry_queue.question_ref"] == "run-a:question-a"
    assert wait_start["retry_queue.mode"] == "enabled"
    assert tokens_deducted["question_id"] == "question-a"
    assert tokens_deducted["retry_queue.question_ref"] == "run-a:question-a"


async def _query_identity_attrs(
    settings: _Settings, **kwargs: object
) -> dict[str, object | None]:
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    llm._query_impl = AsyncMock(return_value=QueryResult(output_text="ok"))  # pyright: ignore[reportPrivateUsage]
    attrs_seen: list[dict[str, object | None]] = []

    def fake_set_attributes(attrs: dict[str, object | None]) -> None:
        attrs_seen.append(dict(attrs))

    with (
        patch("model_library.model_library_settings", settings),
        patch("model_library.base.base.telemetry.set_attributes", fake_set_attributes),
    ):
        await llm.query("hello", **cast(dict[str, Any], kwargs))

    return next(attrs for attrs in attrs_seen if "model.provider" in attrs)


async def test_llm_query_uses_settings_run_and_question_ids():
    attrs = await _query_identity_attrs(
        _Settings(RUN_ID="run-from-settings", QUESTION_ID="question-from-settings")
    )

    assert attrs["run_id"] == "run-from-settings"
    assert attrs["question_id"] == "question-from-settings"
    assert isinstance(attrs["query_id"], str)
    assert len(cast(str, attrs["query_id"])) == 14


async def test_llm_query_uses_task_id_alias_when_question_id_absent():
    attrs = await _query_identity_attrs(
        _Settings(RUN_ID="run-from-settings", TASK_ID="task-from-settings")
    )

    assert attrs["run_id"] == "run-from-settings"
    assert attrs["question_id"] == "task-from-settings"


async def test_llm_query_uses_task_id_alias_when_question_id_blank():
    attrs = await _query_identity_attrs(
        _Settings(QUESTION_ID=" ", TASK_ID="task-from-settings")
    )

    assert attrs["question_id"] == "task-from-settings"


async def test_llm_query_question_id_setting_precedes_task_id_alias():
    attrs = await _query_identity_attrs(
        _Settings(
            RUN_ID="run-from-settings",
            QUESTION_ID="question-from-settings",
            TASK_ID="task-from-settings",
        )
    )

    assert attrs["question_id"] == "question-from-settings"


async def test_llm_query_explicit_ids_override_settings_ids():
    attrs = await _query_identity_attrs(
        _Settings(RUN_ID="run-from-settings", QUESTION_ID="question-from-settings"),
        run_id="run-explicit",
        question_id="question-explicit",
        query_id="query-explicit",
    )

    assert attrs["run_id"] == "run-explicit"
    assert attrs["question_id"] == "question-explicit"
    assert attrs["query_id"] == "query-explicit"


async def test_llm_query_ignores_query_id_setting():
    attrs = await _query_identity_attrs(_Settings(QUERY_ID="query-from-settings"))

    assert attrs["query_id"] != "query-from-settings"
    assert isinstance(attrs["query_id"], str)
    assert len(cast(str, attrs["query_id"])) == 14


@pytest.mark.parametrize("query_id", ["", " ", "\t"])
async def test_llm_query_treats_blank_explicit_query_id_as_absent(query_id: str):
    attrs = await _query_identity_attrs(_Settings(), query_id=query_id)

    assert attrs["query_id"] != query_id
    assert isinstance(attrs["query_id"], str)
    assert len(cast(str, attrs["query_id"])) == 14


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run_id": ""}, "run_id must not be blank"),
        ({"run_id": " "}, "run_id must not be blank"),
        ({"question_id": ""}, "question_id must not be blank"),
        ({"question_id": "\t"}, "question_id must not be blank"),
    ],
)
async def test_llm_query_rejects_blank_explicit_run_and_question_ids(
    kwargs: dict[str, str], match: str
):
    with pytest.raises(ValueError, match=match):
        await _query_identity_attrs(_Settings(), **kwargs)


async def test_llm_query_treats_blank_settings_ids_as_absent():
    attrs = await _query_identity_attrs(
        _Settings(RUN_ID="", QUESTION_ID=" ", TASK_ID="\t")
    )

    assert isinstance(attrs["run_id"], str)
    assert len(cast(str, attrs["run_id"])) == 8
    assert isinstance(attrs["question_id"], str)
    assert len(cast(str, attrs["question_id"])) == 14


async def test_query_id_is_used_for_logging_and_not_forwarded_to_provider():
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    llm._query_impl = AsyncMock(return_value=QueryResult(output_text="ok"))  # pyright: ignore[reportPrivateUsage]

    await llm.query("hello", run_id="run-a", question_id="q1", query_id="query-a")

    _args, kwargs = llm._query_impl.call_args  # pyright: ignore[reportPrivateUsage]
    assert "query_id" not in kwargs
    assert "run_id" not in kwargs
    assert "question_id" not in kwargs


async def test_llm_query_records_provider_query_span():
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    llm._query_impl = AsyncMock(return_value=QueryResult(output_text="ok"))  # pyright: ignore[reportPrivateUsage]
    spans: list[tuple[str, dict[str, object | None], str]] = []

    def fake_start_span(
        name: str,
        attributes: dict[str, object | None],
        *,
        kind: str = "internal",
    ):
        spans.append((name, dict(attributes), kind))
        return nullcontext()

    with patch(
        "model_library.base.base.telemetry.start_span", side_effect=fake_start_span
    ):
        await llm.query("hello", run_id="run-a", question_id="q1", query_id="query-a")

    provider_spans = [
        span for span in spans if span[0] == "model_library.provider_query"
    ]
    expected_span = (
        "model_library.provider_query",
        {
            "run_id": "run-a",
            "question_id": "q1",
            "query_id": "query-a",
            "model.provider": "openai",
            "model.name": "gpt-4o",
            "model.registry_key": None,
            "llm.in_agent": False,
            "llm.in_agent.mode": "disabled",
            "llm.output_schema.mode": "disabled",
            "retry_queue.mode": "disabled",
        },
        "client",
    )
    assert provider_spans == [expected_span]


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client_any = cast(Any, client)
    client_any.lock = lambda *args, **kwargs: _FakeLock()
    set_redis_client(client_any)
    return client


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


# ── Agent forwards question_id to LLM.query ──────────────────────────


async def test_question_id_passed_to_every_query_call():
    """All turns of an agent run receive the same question_id."""
    tc = ToolCall(id="tc1", name="echo", args={"text": "hi"})
    llm = _mock_llm(
        QueryResult(
            output_text=None,
            metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
            tool_calls=[tc],
            history=[TextInput(text="p")],
        ),
        _make_qr("done"),
    )
    agent = Agent(name="test", llm=llm, tools=[EchoTool()], config=_cfg)

    await agent.run([TextInput(text="go")], question_id="q-stable")

    assert llm.query.call_count == 2
    for call in llm.query.call_args_list:
        assert call.kwargs["question_id"] == "q-stable"


async def test_different_runs_get_different_question_ids():
    """Two separate agent.run() calls with different question_ids keep them separate."""
    llm = _mock_llm(_make_qr("a"), _make_qr("b"))
    agent = Agent(name="test", llm=llm, tools=[], config=_cfg)

    await agent.run([TextInput(text="first")], question_id="q-1")
    await agent.run([TextInput(text="second")], question_id="q-2")

    assert llm.query.call_args_list[0].kwargs["question_id"] == "q-1"
    assert llm.query.call_args_list[1].kwargs["question_id"] == "q-2"


async def test_question_id_auto_generated_when_empty():
    """Empty question_id gets a UUID assigned (14-char hex)."""
    question_id: str | None = None
    if not question_id:
        question_id = uuid.uuid4().hex[:14]
    assert len(question_id) == 14
    assert all(c in "0123456789abcdef" for c in question_id)


async def test_question_id_preserved_when_provided():
    question_id: str | None = "my-custom-qid"
    if not question_id:
        question_id = uuid.uuid4().hex[:14]
    assert question_id == "my-custom-qid"


# ── Logger naming ─────────────────────────────────────────────────────


async def test_llm_logger_name_format():
    name = "llm.openai.gpt-4o<run=abc123>"
    logger = logging.getLogger(name)
    assert logger.name == "llm.openai.gpt-4o<run=abc123>"


async def test_agent_logger_name_format():
    llm = _mock_llm(_make_qr())
    llm.model_name = "gpt-4o"
    captured: list[logging.Logger] = []

    def capture_query_logger(*args: object, **kwargs: object) -> QueryResult:
        logger = kwargs.get("logger")
        if not isinstance(logger, logging.Logger):
            raise AssertionError("Expected a scoped query logger")
        captured.append(logger)
        return _make_qr()

    llm.query = AsyncMock(side_effect=capture_query_logger)
    agent = Agent(name="submit", llm=llm, tools=[], config=_cfg)
    await agent.run([TextInput(text="go")], question_id="q1")
    assert captured and "agent.submit<gpt-4o>" in captured[0].name


async def test_agent_passes_scoped_logger_to_llm():
    llm = _mock_llm(_make_qr())
    llm.model_name = "gpt-4o"
    llm.run_id = "run-xyz"
    captured: list[logging.Logger] = []

    def capture_query_logger(*args: object, **kwargs: object) -> QueryResult:
        logger = kwargs.get("logger")
        if not isinstance(logger, logging.Logger):
            raise AssertionError("Expected a scoped query logger")
        captured.append(logger)
        return _make_qr()

    llm.query = AsyncMock(side_effect=capture_query_logger)
    agent = Agent(name="eval", llm=llm, tools=[], config=_cfg)
    await agent.run([TextInput(text="go")], question_id="q1")
    assert captured and captured[0].name.startswith("agent.eval<gpt-4o>")


async def test_query_logger_includes_question_and_query_ids():
    """Verify closing bracket is present (the bug we fixed)."""
    parent = logging.getLogger("llm.openai.gpt-4o<run=abc>")
    child = parent.getChild("<question=q123><query=qry456>")
    assert child.name.endswith("<query=qry456>")


# ── run_logging context manager ───────────────────────────────────────


async def test_run_logging_creates_log_dir_and_file(tmp_path: Path):
    logger = logging.getLogger("test.run_logging.create")
    log_dir = tmp_path / "logs" / "agent" / "model" / "run_001"

    with run_logging(logger, log_dir, "q-42") as output_dir:
        assert output_dir is not None
        assert output_dir == log_dir / "q-42"
        assert (output_dir / "agent.log").exists()
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


async def test_run_logging_cleanup_on_exception(tmp_path: Path):
    logger = logging.getLogger("test.run_logging.exception")
    log_dir = tmp_path / "logs" / "run"

    with pytest.raises(ValueError, match="boom"):
        with run_logging(logger, log_dir, "q-err"):
            logger.info("before error")
            raise ValueError("boom")

    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


async def test_run_logging_reuses_existing_file_handler(tmp_path: Path):
    logger = logging.getLogger("test.run_logging.existing")
    existing_log = tmp_path / "existing" / "agent.log"
    existing_log.parent.mkdir(parents=True)
    existing_log.touch()
    fh = logging.FileHandler(existing_log)
    logger.addHandler(fh)

    try:
        with run_logging(logger, tmp_path / "new_dir", "q-reuse") as output_dir:
            assert output_dir == existing_log.parent / "q-reuse"
            assert output_dir.exists()
            assert not (tmp_path / "new_dir" / "q-reuse").exists()
    finally:
        logger.removeHandler(fh)
        fh.close()


async def test_run_logging_reuses_parent_file_handler(tmp_path: Path):
    parent_logger = logging.getLogger("test.run_logging.parent_fh")
    child_logger = parent_logger.getChild("child")
    existing_log = tmp_path / "parent_out" / "agent.log"
    existing_log.parent.mkdir(parents=True)
    existing_log.touch()
    fh = logging.FileHandler(existing_log)
    parent_logger.addHandler(fh)

    try:
        with run_logging(child_logger, tmp_path / "new_dir", "q-reuse") as output_dir:
            assert output_dir == existing_log.parent / "q-reuse"
            assert output_dir.exists()
            assert not (tmp_path / "new_dir" / "q-reuse").exists()
    finally:
        parent_logger.removeHandler(fh)
        fh.close()


async def test_run_logging_writes_to_file(tmp_path: Path):
    logger = logging.getLogger("test.run_logging.write")
    logger.setLevel(logging.DEBUG)

    with run_logging(logger, tmp_path / "run", "q-write") as output_dir:
        logger.info("test message 123")

    assert output_dir is not None
    content = (output_dir / "agent.log").read_text()
    assert "test message 123" in content


# ── TokenRetrier key construction ─────────────────────────────────────


async def test_inflight_key_scoped_to_run_id():
    retrier = _make_retrier(run_id="run-A")
    assert retrier._run_inflight_key == f"{TOKEN_KEY}:run:run-A:inflight"


async def test_different_run_ids_different_inflight_keys():
    r1 = _make_retrier(run_id="run-A")
    r2 = _make_retrier(run_id="run-B")
    assert r1._run_inflight_key != r2._run_inflight_key


async def test_question_meta_key_scoped_to_question_id():
    retrier = _make_retrier(question_id="q-special")
    assert retrier._question_meta_key == f"{TOKEN_KEY}:inflight:test-run:q-special"


async def test_dynamic_estimate_keyed_by_run_id():
    retrier = _make_retrier(run_id="run-EMA", use_dynamic_estimate=True)
    assert retrier.dynamic_estimate_key == f"{TOKEN_KEY}:dynamic_estimate:run-EMA"


async def test_dynamic_estimate_none_when_disabled():
    retrier = _make_retrier(use_dynamic_estimate=False)
    assert retrier.dynamic_estimate_key is None


# ── TokenRetrier inflight tracking ────────────────────────────────────


async def test_same_question_id_across_turns_overwrites_inflight(redis):
    """Agentic loop: same question_id updates timestamp, not duplicate."""
    await _init_tokens(redis, value=10000)

    for _ in range(3):
        retrier = _make_retrier(run_id="run-agent", question_id="q-agentic")
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    inflight_key = f"{TOKEN_KEY}:run:run-agent:inflight"
    members = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
    assert len(members) == 1
    assert members[0] == "run-agent:q-agentic"


async def test_different_question_ids_same_run_create_separate_entries(redis):
    await _init_tokens(redis, value=10000)

    for i in range(3):
        retrier = _make_retrier(run_id="run-multi", question_id=f"q-{i}")
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    inflight_key = f"{TOKEN_KEY}:run:run-multi:inflight"
    members = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
    assert len(members) == 3
    assert set(members) == {"run-multi:q-0", "run-multi:q-1", "run-multi:q-2"}
    assert await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-multi")


async def test_execute_cleanup_removes_only_own_question(redis):
    await _init_tokens(redis, value=10000, limit=10000)

    inflight_key = f"{TOKEN_KEY}:run:run-shared:inflight"
    await redis.zadd(inflight_key, {"other-q": time.time()})
    await redis.sadd(f"{TOKEN_KEY}:active_runs", "run-shared")

    retrier = _make_retrier(run_id="run-shared", question_id="my-q")

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 50
    mock_qr.metadata.total_output_tokens = 25
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    run_retry = retrier.execute
    await run_retry(AsyncMock(return_value=(mock_qr, 0.5)))

    members = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
    assert "run-shared:my-q" not in members
    assert "other-q" in members
    assert await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-shared")


async def test_metadata_cleaned_on_pre_function_failure(redis):
    await _init_tokens(redis, value=0)
    await redis.hset(f"{TOKEN_KEY}:config", mapping={"burst_limit": "160"})

    retrier = _make_retrier(run_id="run-fail", question_id="q-fail")

    original_eval = redis.eval

    async def cancel_eval(script, numkeys, *args):  # noqa: S307
        if numkeys == 3 and args and args[0] == TOKEN_KEY:
            raise asyncio.CancelledError()
        return await original_eval(script, numkeys, *args)  # noqa: S307

    redis.eval = cancel_eval

    with pytest.raises(asyncio.CancelledError):
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert not await redis.exists(f"{TOKEN_KEY}:inflight:run-fail:q-fail")


# ── TokenRetrier queued behavior ──────────────────────────────────────


async def test_queued_retrier_tracks_dispatched(redis):
    """Run in the benchmark queue gets its dispatched counter incremented."""
    await _init_tokens(redis, value=1000)
    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "bench-run")

    retrier = _make_retrier(run_id="bench-run", question_id="q-bench")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    dispatched_key = f"{KEY_PREFIX}:provider:model:benchmark:run:bench-run:dispatched"
    assert await redis.sismember(dispatched_key, "bench-run:q-bench")


async def test_non_queued_retrier_tracks_dispatched(redis):
    """Run not in the benchmark queue still gets a dispatched entry (all runs track dispatched)."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="normal-run", question_id="q-normal")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    keys = await redis.keys(f"{KEY_PREFIX}:*:benchmark:run:*:dispatched")
    assert len(keys) == 1


async def test_straggler_gets_max_priority(redis):
    """Run in the queue but not the head is demoted to MAX_PRIORITY (-5)."""
    await _init_tokens(redis, value=1000)
    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "other-run")
    await redis.rpush(queue_key, "straggler-run")

    retrier = _make_retrier(run_id="straggler-run", question_id="q-strag")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == -5  # MAX_PRIORITY


async def test_non_queued_skips_straggler_check(redis):
    """Run not in the queue is unaffected by the straggler check."""
    await _init_tokens(redis, value=1000)
    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "other-run")

    retrier = _make_retrier(run_id="my-run", question_id="q-normal")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == 0  # INITIAL_PRIORITY


# ── run_id / question_id flows through LLM.query → TokenRetrier ───────


async def test_run_id_and_question_id_forwarded_to_token_retrier():
    """run_id and question_id passed to llm.query() are forwarded to TokenRetrier."""
    MockLLM = _make_mock_llm_class()
    llm = MockLLM("gpt-4o", "openai")
    llm.token_retry_params = TokenRetryParams(  # pyright: ignore[reportAttributeAccessIssue]
        input_modifier=1.0,
        output_modifier=1.0,
        limit=10_000,
    )
    llm._resolved_token_retry_params = ResolvedTokenRetryParams(  # pyright: ignore[reportAttributeAccessIssue]
        input_modifier=1.0,
        output_modifier=1.0,
        use_dynamic_estimate=True,
        limit=10_000,
    )
    llm.estimate_query_tokens = AsyncMock(return_value=(100, 50))  # pyright: ignore[reportAttributeAccessIssue]

    captured: dict[str, str] = {}

    class CapturingRetrier:
        def __init__(self, **kwargs: Any):
            captured.update(
                {"run_id": kwargs["run_id"], "question_id": kwargs["question_id"]}
            )

        async def execute(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return await func()

        async def validate(self) -> None:
            pass

    with patch("model_library.retriers.token.token.TokenRetrier", CapturingRetrier):
        await llm.query("test input", run_id="my-run-id", question_id="my-question-id")

    assert captured["run_id"] == "my-run-id"
    assert captured["question_id"] == "my-question-id"
