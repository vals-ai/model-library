"""Native files retain pre-hook evidence without changing the live agent loop."""

import asyncio
import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from model_library.agent import Agent, AgentHooks, AgentResult, Tool, ToolOutput
from model_library.base.base import LLM
from model_library.base.input import (
    InputItem,
    RawResponse,
    TextInput,
    ToolCall,
    ToolResult,
)
from model_library.base.output import QueryResult
from tests.unit.agent.helpers import (
    DoneTool,
    make_agent,
    make_metadata,
    make_text_response,
    make_tool_call,
    make_tool_response,
    mock_llm,
)


class HelperTool(Tool):
    name = "helper"
    description = "Return a helper query and a compact model-visible answer"
    parameters: dict[str, Any] = {}

    def __init__(self, response: QueryResult):
        self.response = response
        self.outputs: list[ToolOutput] = []

    async def execute(self, args, state, logger):
        output = ToolOutput(
            output="compact answer",
            metadata=self.response.metadata,
            native_query_result=self.response,
        )
        self.outputs.append(output)
        return output


class BlockingTool(Tool):
    name = "block"
    description = "Wait until cancelled"
    parameters: dict[str, Any] = {}

    def __init__(self):
        self.started = asyncio.Event()

    async def execute(self, args, state, logger):
        self.started.set()
        await asyncio.Event().wait()
        return ToolOutput(output="unreachable")


class OpaqueHistory:
    def __deepcopy__(self, memo):
        raise AssertionError("Provider history must not be deep-copied")


def turn_dir(tmp_path: Path) -> Path:
    # The existing autouse fixture fixes Agent._build_log_dir to this directory.
    return tmp_path / "test" / "mock-model" / "run" / "q1" / "turns" / "turn_001"


def load_result(path: Path) -> Any:
    return json.loads((path / "result.json").read_text())


async def test_main_response_is_saved_before_first_blocked_tool(tmp_path):
    tool = BlockingTool()
    response = make_tool_response(
        [make_tool_call("block")], output_text="main original"
    )
    response.reasoning = "main reasoning"
    response.history.append(RawResponse(response=OpaqueHistory()))
    agent = make_agent(mock_llm(response), [tool])

    task = asyncio.create_task(agent.run([TextInput(text="go")], question_id="q1"))
    try:
        await asyncio.wait_for(tool.started.wait(), timeout=5)
        native = load_result(turn_dir(tmp_path))
        assert native["query_result"]["output_text"] == "main original"
        assert native["query_result"]["reasoning"] == "main reasoning"
        assert native["tool_call_records"] == []
        assert "history" not in native["query_result"]
        assert (turn_dir(tmp_path) / "history.json").exists()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("failure", ["hook", "cancellation"])
async def test_completed_prefix_survives_later_failure(tmp_path, failure):
    helper = HelperTool(make_text_response("helper original"))
    blocker = BlockingTool()
    calls = [ToolCall(id="first", name="helper", args={})]
    calls.append(
        ToolCall(id="second", name="helper" if failure == "hook" else "block", args={})
    )

    def on_tool_result(record, state):
        if record.tool_call.id == "second":
            raise RuntimeError("later hook failed")
        record.tool_output.output = "mutated first output"
        record.tool_call.id = "mutated first id"

    agent = make_agent(
        mock_llm(make_tool_response(calls)),
        [helper, blocker],
        hooks=AgentHooks(on_tool_result=on_tool_result),
    )
    if failure == "hook":
        result = await agent.run([TextInput(text="go")], question_id="q1")
        assert result.final_error is not None
        assert result.final_error.message == "later hook failed"
        assert (
            result.turns == []
        )  # Native partial turns do not become completed summaries.
        assert result.final_aggregated_metadata.in_tokens == 0
    else:
        task = asyncio.create_task(agent.run([TextInput(text="go")], question_id="q1"))
        try:
            await asyncio.wait_for(blocker.started.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    native = load_result(turn_dir(tmp_path))
    records = native["tool_call_records"]
    assert len(records) == (2 if failure == "hook" else 1)
    assert records[0]["tool_output"]["output"] == "compact answer"
    assert records[0]["tool_call"]["id"] == "first"
    assert native["query_result"]["tool_calls"][0]["id"] == "first"
    assert (
        load_result(turn_dir(tmp_path) / "helper_queries" / "tool_000")["query_result"][
            "output_text"
        ]
        == "helper original"
    )


async def test_tool_checkpoint_keeps_loop_responsive_and_defers_cancellation(
    tmp_path, monkeypatch
):
    calls = [
        ToolCall(id="first", name="submit", args={"answer": "first"}),
        ToolCall(id="second", name="submit", args={"answer": "second"}),
    ]
    agent = make_agent(mock_llm(make_tool_response(calls)), [DoneTool()])
    original_write = agent._write_turn_dir
    checkpoint_started = threading.Event()
    release_checkpoint = threading.Event()
    checkpoint_finished = threading.Event()
    loop_serviced = asyncio.Event()
    loop = asyncio.get_running_loop()

    def blocking_write(output_dir, turn_number, turn, state, history, logger):
        if turn.tool_call_records and not checkpoint_started.is_set():
            checkpoint_started.set()
            loop.call_soon_threadsafe(loop_serviced.set)
            try:
                assert release_checkpoint.wait(timeout=2), (
                    "checkpoint release timed out"
                )
                original_write(output_dir, turn_number, turn, state, history, logger)
            finally:
                checkpoint_finished.set()
            return
        original_write(output_dir, turn_number, turn, state, history, logger)

    monkeypatch.setattr(agent, "_write_turn_dir", blocking_write)
    started_wait = asyncio.create_task(asyncio.to_thread(checkpoint_started.wait, 3))
    run_task = asyncio.create_task(agent.run([TextInput(text="go")], question_id="q1"))
    fail_safe = threading.Timer(2, release_checkpoint.set)
    fail_safe.start()
    try:
        assert await started_wait
        await asyncio.wait_for(loop_serviced.wait(), timeout=1)
        assert not checkpoint_finished.is_set()

        run_task.cancel()
        await asyncio.sleep(0)
        assert not run_task.done()
        run_task.cancel()
        await asyncio.sleep(0)
        assert not run_task.done()

        release_checkpoint.set()
        assert await asyncio.to_thread(checkpoint_finished.wait, 3)
        native = load_result(turn_dir(tmp_path))
        assert [
            record["tool_call"]["id"] for record in native["tool_call_records"]
        ] == ["first"]
        assert native["tool_call_records"][0]["tool_output"]["output"] == "first"
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        release_checkpoint.set()
        fail_safe.cancel()
        if not run_task.done():
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task


async def test_native_originals_survive_successive_and_final_writes(tmp_path):
    helper_response = make_text_response("helper original", make_metadata(50, 20, 0.08))
    helper_response.reasoning = "helper reasoning"
    helper_response.history.append(RawResponse(response=OpaqueHistory()))
    helper = HelperTool(helper_response)
    calls = [
        ToolCall(id=f"call-{i}", name="helper", args={"nested": {"value": "original"}})
        for i in range(2)
    ]
    response = make_tool_response(calls, output_text="main original")
    response.extras.provider_response_id = "main-response"
    seen_records = []

    def on_tool_result(record, state):
        index = len(seen_records)
        native = load_result(turn_dir(tmp_path))
        assert [r["tool_output"]["output"] for r in native["tool_call_records"]] == [
            "compact answer"
        ] * (index + 1)
        helper_dir = turn_dir(tmp_path) / "helper_queries" / f"tool_{index:03d}"
        captured = load_result(helper_dir)
        assert captured["tool_call"]["id"] == f"call-{index}"
        assert captured["query_result"]["output_text"] == "helper original"
        assert captured["query_result"]["reasoning"] == "helper reasoning"
        assert "history" not in captured["query_result"]
        history = json.loads((helper_dir / "history.json").read_text())
        assert history[0]["text"] == "prompt"
        assert set(history[1]["response"]) == {"pickle", "hmac"}
        assert record.tool_output.native_query_result is None
        record.tool_output.output = "hook replacement"
        record.tool_call.args["nested"]["value"] = "hook replacement"
        response.output_text = "hook main"
        response.extras.provider_response_id = "hook response id"
        seen_records.append(record)

    llm = mock_llm(response, make_text_response("finished"))
    agent = make_agent(
        llm,
        [helper],
        hooks=AgentHooks(on_tool_result=on_tool_result),
        history_secret=b"test-secret",
    )
    result = await agent.run([TextInput(text="go")], question_id="q1")

    assert result.success
    assert result.final_answer == "finished"
    native = load_result(turn_dir(tmp_path))
    assert native["query_result"]["output_text"] == "main original"
    assert native["query_result"]["extras"]["provider_response_id"] == "main-response"
    assert (
        native["query_result"]["tool_calls"][0]["args"]["nested"]["value"] == "original"
    )
    assert [r["tool_output"]["output"] for r in native["tool_call_records"]] == [
        "compact answer",
        "compact answer",
    ]
    assert (
        native["tool_call_records"][0]["tool_call"]["args"]["nested"]["value"]
        == "original"
    )
    assert result.turns[0].tool_calls[0].output_length == len("hook replacement")
    # Helper-query metadata is not added to the main-query aggregate.
    assert result.final_aggregated_metadata.in_tokens == 20
    assert result.final_aggregated_metadata.out_tokens == 10
    assert result.turns[0].tool_calls[0].metadata.in_tokens == 50
    sent_history = llm.query.call_args_list[1].kwargs["input"]
    assert [item.result for item in sent_history if isinstance(item, ToolResult)] == [
        "compact answer",
        "compact answer",
    ]
    assert "native_query_result" not in result.model_dump_json()
    assert "helper original" not in result.model_dump_json()


async def test_helper_capture_failure_is_nonfatal_and_releases_history(
    tmp_path, monkeypatch
):
    response = make_text_response("helper original")
    helper = HelperTool(response)
    original_write = Path.write_text

    def fail_helper_history(path, *args, **kwargs):
        if path.parent.name == "tool_000":
            raise OSError("capture unavailable")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_helper_history)

    def on_tool_result(record, state):
        assert record.tool_output.native_query_result is None

    agent = make_agent(
        mock_llm(
            make_tool_response([make_tool_call("helper")]), make_text_response("done")
        ),
        [helper],
        hooks=AgentHooks(on_tool_result=on_tool_result),
    )
    result = await agent.run([TextInput(text="go")], question_id="q1")
    assert result.success
    assert result.final_answer == "done"
    assert helper.outputs[0].native_query_result is None


async def test_failed_atomic_replace_preserves_previous_result(tmp_path, monkeypatch):
    original_replace = Path.replace
    writes = 0

    def fail_progress_replace(path, target):
        nonlocal writes
        if Path(target) == turn_dir(tmp_path) / "result.json":
            writes += 1
            if writes > 1:
                raise OSError("replace unavailable")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_progress_replace)
    agent = make_agent(
        mock_llm(
            make_tool_response(
                [make_tool_call("submit", {"answer": "done"})],
                output_text="main original",
            )
        ),
        [DoneTool()],
    )
    result = await agent.run([TextInput(text="go")], question_id="q1")
    assert result.success
    assert result.final_answer == "done"
    native = load_result(turn_dir(tmp_path))
    assert native["query_result"]["output_text"] == "main original"
    assert native["tool_call_records"] == []
    assert {path.name for path in turn_dir(tmp_path).iterdir()} == {
        "result.json",
        "state.json",
        "history.json",
    }


async def test_main_capture_failure_is_nonfatal(monkeypatch):
    original_mkdir = Path.mkdir

    def fail_mkdir(path, *args, **kwargs):
        if path.name == "turn_001":
            raise OSError("capture unavailable")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    agent = make_agent(mock_llm(make_text_response("done")))
    result = await agent.run([TextInput(text="go")], question_id="q1")
    assert result.success
    assert result.final_answer == "done"


class LegacyLoopAgent(Agent):
    async def _run(
        self,
        input: Sequence[InputItem],
        *,
        question_id: str,
        run_id: str | None = None,
        state: dict[str, Any] | None = None,
        output_dir: Path,
        logger: logging.Logger,
        atif_export: bool = False,
    ) -> AgentResult:
        history = list(input)
        records = await self._execute_tool_calls(
            [make_tool_call("submit", {"answer": "legacy done"})], {}, history, logger
        )
        return AgentResult(
            final_answer=records[0].tool_output.output,
            final_history=history,
            turns=[],
            final_duration_seconds=0,
            output_dir=output_dir,
        )


async def test_legacy_custom_loop_and_execute_calls_keep_their_api():
    base = make_agent(mock_llm())
    agent = LegacyLoopAgent(
        llm=cast(LLM, mock_llm()), tools=[DoneTool()], name="legacy", config=base.config
    )
    result = await agent.run([TextInput(text="go")], question_id="q1")
    assert result.final_answer == "legacy done"
    assert isinstance(result.final_history[-1], ToolResult)
    assert result.final_history[-1].result == "legacy done"


async def test_paired_legal_kwargs_capture_helper_before_hook(tmp_path):
    helper = HelperTool(make_text_response("helper original"))
    seen = []

    def on_tool_result(record, state):
        native = load_result(
            tmp_path / "turns" / "turn_007" / "helper_queries" / "tool_000"
        )
        seen.append(native["query_result"]["output_text"])
        assert record.tool_output.native_query_result is None

    agent = make_agent(
        mock_llm(), [helper], hooks=AgentHooks(on_tool_result=on_tool_result)
    )
    history = []
    records = await agent._execute_tool_calls(
        [make_tool_call("helper")],
        {},
        history,
        logging.getLogger(__name__),
        output_dir=tmp_path,
        turn_number=7,
    )
    assert seen == ["helper original"]
    assert records[0].tool_output.output == "compact answer"
    assert history[0].result == "compact answer"


@pytest.mark.parametrize("kwargs", [{"turn_number": 1}, {"output_dir": Path("unused")}])
async def test_partial_capture_kwargs_are_rejected(kwargs):
    agent = make_agent(mock_llm())
    with pytest.raises(ValueError, match="must be provided together"):
        await agent._execute_tool_calls(
            [], {}, [], logging.getLogger(__name__), **kwargs
        )
