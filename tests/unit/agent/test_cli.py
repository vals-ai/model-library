"""Unit tests for model_library.agent.cli"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.agent import AgentConfig
from model_library.agent.cli import (
    _build_parser,
    _collect_tools,
    _load_tool_module,
    _read_problem_statement,
    _run,
    main,
)
from model_library.agent.tools import TOOL_REGISTRY
from model_library.agent.tools.stop import StopTool
from model_library.agent.tools.submit import SubmitTool
from model_library.base.input import TextInput


_REQUIRED = ("--name", "test", "--question-id", "q1")


def _parse(*argv: str) -> Any:
    return _build_parser().parse_args([*_REQUIRED, *argv])


def _make_result(final_answer: str = "done") -> MagicMock:
    result = MagicMock()
    result.final_answer = final_answer
    result.model_dump_json = MagicMock(return_value=json.dumps({"final_answer": final_answer}))
    return result


class TestCLI:
    # --- _build_parser ---

    def test_required_model(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--problem-statement", "x"])

    def test_required_problem_statement(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--model", "openai/gpt-4"])

    def test_defaults(self):
        args = _parse("--model", "openai/gpt-4", "--problem-statement", "x")
        assert args.max_turns == 100
        assert args.max_time_seconds == 28800
        assert args.log_level == "INFO"
        assert args.log_file == "agent.log"
        assert args.output is None
        assert args.tools == ""

    def test_max_turns_is_int(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--max-turns", "50")
        assert args.max_turns == 50
        assert isinstance(args.max_turns, int)

    def test_max_time_seconds_is_float(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--max-time-seconds", "3600.5")
        assert args.max_time_seconds == 3600.5
        assert isinstance(args.max_time_seconds, float)

    def test_invalid_log_level_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--model", "m", "--problem-statement", "x", "--log-level", "VERBOSE"])

    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            args = _parse("--model", "m", "--problem-statement", "x", "--log-level", level)
            assert args.log_level == level

    # --- _collect_tools ---

    def test_empty_tools_returns_empty(self):
        args = _parse("--model", "m", "--problem-statement", "x")
        assert _collect_tools(args) == []

    def test_single_tool(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--tools", "submit")
        tools = _collect_tools(args)
        assert len(tools) == 1
        assert isinstance(tools[0], SubmitTool)

    def test_multiple_tools(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--tools", "stop,submit")
        tools = _collect_tools(args)
        assert len(tools) == 2
        assert isinstance(tools[0], StopTool)
        assert isinstance(tools[1], SubmitTool)

    def test_unknown_tool_raises(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--tools", "nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            _collect_tools(args)

    # --- --tool-module / _load_tool_module ---

    def test_tool_module_default_is_none(self):
        args = _parse("--model", "m", "--problem-statement", "x")
        assert args.tool_module is None

    def test_tool_module_flag_parsed(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--tool-module", "my_tools")
        assert args.tool_module == "my_tools"

    def test_load_tool_module_merges_registry(self):
        fake_tool_cls = lambda: MagicMock()
        fake_mod = MagicMock()
        fake_mod.TOOL_REGISTRY = {"custom_tool": fake_tool_cls}
        original_keys = set(TOOL_REGISTRY.keys())
        with patch("importlib.import_module", return_value=fake_mod) as mock_import:
            _load_tool_module("my_tools")
            mock_import.assert_called_once_with("my_tools")
        assert "custom_tool" in TOOL_REGISTRY
        assert TOOL_REGISTRY["custom_tool"] is fake_tool_cls
        # cleanup
        del TOOL_REGISTRY["custom_tool"]
        assert set(TOOL_REGISTRY.keys()) == original_keys

    def test_load_tool_module_missing_raises(self):
        with pytest.raises(ModuleNotFoundError):
            _load_tool_module("nonexistent_module_xyz_12345")

    def test_collect_tools_with_tool_module(self):
        fake_tool = MagicMock()
        fake_tool_cls = lambda: fake_tool
        fake_mod = MagicMock()
        fake_mod.TOOL_REGISTRY = {"custom_tool": fake_tool_cls}
        args = _parse(
            "--model", "m", "--problem-statement", "x",
            "--tool-module", "my_tools", "--tools", "custom_tool",
        )
        with patch("importlib.import_module", return_value=fake_mod):
            tools = _collect_tools(args)
        assert len(tools) == 1
        assert tools[0] is fake_tool
        # cleanup
        del TOOL_REGISTRY["custom_tool"]

    def test_collect_tools_without_tool_module_skips_import(self):
        args = _parse("--model", "m", "--problem-statement", "x", "--tools", "submit")
        with patch("importlib.import_module") as mock_import:
            tools = _collect_tools(args)
            mock_import.assert_not_called()
        assert len(tools) == 1

    # --- _read_problem_statement ---

    def test_reads_file(self, tmp_path: Path):
        f = tmp_path / "problem.txt"
        f.write_text("solve this")
        assert _read_problem_statement(str(f)) == "solve this"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _read_problem_statement("/nonexistent/path/problem.txt")

    def test_stdin(self, monkeypatch: pytest.MonkeyPatch):
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
        assert _read_problem_statement("-") == "from stdin"

    # --- _run ---

    @pytest.mark.asyncio
    async def test_config_wiring(self, tmp_path: Path):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("hello")
        args = _parse(
            "--model", "openai/gpt-4",
            "--problem-statement", str(ps_file),
            "--max-turns", "42",
            "--max-time-seconds", "1234.5",
            "--log-file", str(tmp_path / "agent.log"),
        )
        captured: list[AgentConfig] = []

        def fake_agent(*, llm, tools, name, logger, config):
            captured.append(config)
            agent = MagicMock()
            agent.run = AsyncMock(return_value=_make_result())
            return agent

        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent", side_effect=fake_agent),
        ):
            await _run(args)

        assert captured[0].max_turns == 42
        assert captured[0].max_time_seconds == 1234.5

    @pytest.mark.asyncio
    async def test_log_level_passed(self, tmp_path: Path):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("hello")
        args = _parse(
            "--model", "openai/gpt-4",
            "--problem-statement", str(ps_file),
            "--log-level", "DEBUG",
            "--log-file", str(tmp_path / "agent.log"),
        )
        captured_levels: list[int] = []
        original = __import__("model_library.utils", fromlist=["create_file_logger"]).create_file_logger

        def fake_logger(name, log_file, level=logging.INFO, **kwargs):
            captured_levels.append(level)
            return original(name, log_file, level=level, **kwargs)

        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent") as mock_agent_cls,
            patch("model_library.agent.cli.create_file_logger", side_effect=fake_logger),
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=_make_result())
            await _run(args)

        assert captured_levels[0] == logging.DEBUG

    @pytest.mark.asyncio
    async def test_problem_statement_passed_to_agent(self, tmp_path: Path):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("my problem")
        args = _parse(
            "--model", "openai/gpt-4",
            "--problem-statement", str(ps_file),
            "--log-file", str(tmp_path / "agent.log"),
        )
        captured_inputs: list[Any] = []

        async def fake_run(inputs: list[Any], *, question_id: str) -> Any:
            captured_inputs.extend(inputs)
            return _make_result()

        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent") as mock_agent_cls,
        ):
            mock_agent_cls.return_value.run = fake_run
            await _run(args)

        assert len(captured_inputs) == 1
        assert isinstance(captured_inputs[0], TextInput)
        assert captured_inputs[0].text == "my problem"

    # --- main ---

    def test_prints_final_answer(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("q")
        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent") as mock_agent_cls,
            patch("sys.argv", ["prog", *_REQUIRED, "--model", "m", "--problem-statement", str(ps_file), "--log-file", str(tmp_path / "a.log")]),
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=_make_result("the answer"))
            main()
        assert "the answer" in capsys.readouterr().out

    def test_output_written_to_file(self, tmp_path: Path):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("q")
        output_file = tmp_path / "result.json"
        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent") as mock_agent_cls,
            patch("sys.argv", [
                "prog", *_REQUIRED, "--model", "m", "--problem-statement", str(ps_file),
                "--log-file", str(tmp_path / "a.log"), "--output", str(output_file),
            ]),
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=_make_result("the answer"))
            main()
        assert output_file.exists()
        assert json.loads(output_file.read_text())["final_answer"] == "the answer"

    def test_no_output_flag_skips_file(self, tmp_path: Path):
        ps_file = tmp_path / "ps.txt"
        ps_file.write_text("q")
        with (
            patch("model_library.agent.cli.get_registry_model", return_value=MagicMock()),
            patch("model_library.agent.cli.Agent") as mock_agent_cls,
            patch("sys.argv", ["prog", *_REQUIRED, "--model", "m", "--problem-statement", str(ps_file), "--log-file", str(tmp_path / "a.log")]),
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=_make_result())
            main()
        assert not any(tmp_path.glob("*.json"))
