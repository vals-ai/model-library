"""Unit tests for model_library.agent.tools.bash"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from model_library.agent.tools.bash import BashTool, _read_stream

_logger = logging.getLogger("test_bash_tool")


# --- _read_stream ---


class TestReadStream:
    async def test_reads_full_output(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"hello world")
        reader.feed_eof()

        result = await _read_stream(reader, max_len=1_000_000)
        assert result == "hello world"

    async def test_truncates_at_max_len(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"a" * 200)
        reader.feed_eof()

        result = await _read_stream(reader, max_len=100)
        assert result.endswith(" [truncated]")
        assert result.startswith("a" * 100)
        assert "[truncated]" in result

    async def test_handles_invalid_utf8(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"hello \xff\xfe world")
        reader.feed_eof()

        result = await _read_stream(reader, max_len=1_000_000)
        assert "hello" in result
        assert "world" in result

    async def test_empty_stream(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()

        result = await _read_stream(reader, max_len=1_000_000)
        assert result == ""


# --- BashTool.execute ---


class TestBashToolExecute:
    def _make_tool(self, **kwargs: object) -> BashTool:
        defaults = {"working_dir": "/tmp"}
        defaults.update(kwargs)
        return BashTool(**defaults)  # type: ignore[arg-type]

    async def test_simple_echo(self):
        tool = self._make_tool()
        result = await tool.execute({"command": "echo hello"}, {}, _logger)

        payload = json.loads(result.output)
        assert payload["exit_code"] == 0
        assert "hello" in payload["stdout"]
        assert result.error is None

    async def test_stderr_captured(self):
        tool = self._make_tool()
        result = await tool.execute({"command": "echo err >&2"}, {}, _logger)

        payload = json.loads(result.output)
        assert "err" in payload["stderr"]

    async def test_nonzero_exit_code(self):
        tool = self._make_tool()
        result = await tool.execute({"command": "exit 42"}, {}, _logger)

        payload = json.loads(result.output)
        assert payload["exit_code"] == 42
        assert result.error == "exit_code=42"

    async def test_timeout(self):
        tool = self._make_tool()
        result = await tool.execute(
            {"command": "sleep 60", "timeout": 1}, {}, _logger
        )

        assert result.error is not None
        assert "timed out" in result.error

    async def test_invalid_working_dir(self):
        tool = self._make_tool(working_dir="/nonexistent_dir_xyz")
        result = await tool.execute({"command": "echo hi"}, {}, _logger)

        assert result.error is not None
        assert "failed to launch subprocess" in result.error

    async def test_working_dir_respected(self):
        tool = self._make_tool(working_dir="/tmp")
        result = await tool.execute({"command": "pwd"}, {}, _logger)

        payload = json.loads(result.output)
        # /tmp -> /private/tmp on macOS
        assert "tmp" in payload["stdout"]
