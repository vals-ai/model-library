"""Unit tests for model_library.agent.tools.bash"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time

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

    async def test_reads_uncapped_output(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"a" * 200)
        reader.feed_eof()

        result = await _read_stream(reader, max_len=None)
        assert result == "a" * 200

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
        result = await tool.execute({"command": "sleep 60", "timeout": 1}, {}, _logger)

        assert result.error is not None
        assert "timed out" in result.error

    async def test_large_output_timeout_is_bounded(self):
        tool = self._make_tool(max_len=10_000)
        command = "\n".join(
            [
                "python - <<'PY'",
                "import sys, time",
                "for i in range(300000):",
                "    sys.stdout.write('x' * 100 + '\\n')",
                "    if i % 1000 == 0:",
                "        sys.stdout.flush()",
                "time.sleep(30)",
                "PY",
            ]
        )

        result = await asyncio.wait_for(
            tool.execute({"command": command, "timeout": 1}, {}, _logger),
            timeout=5,
        )

        assert result.error is not None
        assert "timed out" in result.error

    async def test_uncapped_output_mode(self):
        tool = self._make_tool(max_len=None)
        result = await tool.execute(
            {"command": "python - <<'PY'\nprint('x' * 20000)\nPY"}, {}, _logger
        )

        payload = json.loads(result.output)
        assert payload["exit_code"] == 0
        assert len(payload["stdout"].strip()) == 20_000
        assert "[truncated]" not in payload["stdout"]

    async def test_completed_command_cleans_up_background_children(self, tmp_path):
        pid_file = tmp_path / "child.pid"
        tool = self._make_tool()
        result = await tool.execute(
            {"command": (f"sh -c 'sleep 60 >/dev/null 2>&1 & echo $! > {pid_file}'")},
            {},
            _logger,
        )

        payload = json.loads(result.output)
        assert payload["exit_code"] == 0
        child_pid = int(pid_file.read_text().strip())
        time.sleep(0.2)
        child_status = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "pid="],
            check=False,
            capture_output=True,
            text=True,
        )
        assert child_status.stdout.strip() == ""

    async def test_background_child_inheriting_stdout_completes_and_is_cleaned_up(
        self, tmp_path
    ):
        pid_file = tmp_path / "child.pid"
        tool = self._make_tool()
        result = await asyncio.wait_for(
            tool.execute(
                {
                    "command": (f"sh -c 'sleep 60 & echo $! > {pid_file}; echo done'"),
                    "timeout": 1,
                },
                {},
                _logger,
            ),
            timeout=5,
        )

        payload = json.loads(result.output)
        assert payload["exit_code"] == 0
        assert payload["stdout"].strip() == "done"
        assert result.error is None
        child_pid = int(pid_file.read_text().strip())
        time.sleep(0.2)
        child_status = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "pid="],
            check=False,
            capture_output=True,
            text=True,
        )
        assert child_status.stdout.strip() == ""

    async def test_cancellation_cleans_up_process_group(self, tmp_path):
        pid_file = tmp_path / "child.pid"
        tool = self._make_tool()
        task = asyncio.create_task(
            tool.execute(
                {
                    "command": (
                        f"sh -c 'sleep 60 >/dev/null 2>&1 & echo $! > {pid_file}; wait'"
                    )
                },
                {},
                _logger,
            )
        )

        for _ in range(50):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text().strip())

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        time.sleep(0.2)
        child_status = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "pid="],
            check=False,
            capture_output=True,
            text=True,
        )
        assert child_status.stdout.strip() == ""

    async def test_cancellation_during_startup_cleans_up_process_group(
        self, tmp_path, monkeypatch
    ):
        # If cancellation lands while the subprocess is still launching, execute()
        # has not yet captured `proc`. The whole process group must still be
        # killed -- otherwise a backgrounded child that inherited stdout leaks
        # (and asyncio's own startup-cancel cleanup blocks on the open pipe).
        # Widen the launch window so the cancellation deterministically lands in
        # it on any platform.
        pid_file = tmp_path / "child.pid"
        real_create = asyncio.create_subprocess_shell
        launched = asyncio.Event()

        async def slow_create(*args, **kwargs):
            proc = await real_create(*args, **kwargs)
            launched.set()
            await asyncio.sleep(1)
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_shell", slow_create)

        tool = self._make_tool()
        task = asyncio.create_task(
            tool.execute(
                {"command": f"sleep 60 & echo $! > {pid_file}; echo started"},
                {},
                _logger,
            )
        )
        await launched.wait()
        for _ in range(50):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text().strip())

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        time.sleep(0.2)
        child_status = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "pid="],
            check=False,
            capture_output=True,
            text=True,
        )
        assert child_status.stdout.strip() == ""

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
