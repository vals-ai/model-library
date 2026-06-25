from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import signal
from typing import Any

from model_library.agent.tool import Tool, ToolOutput

_DEFAULT_MAX_LEN = 1_000_000  # 1 MB
_CHUNK_SIZE = 4096


def _set_resource_limits(
    *,
    cpu_secs: int | None = None,
    mem_bytes: int | None = None,
    fsize_bytes: int | None = None,
) -> None:
    """Set resource limits in a forked child process (via *preexec_fn*)."""
    if cpu_secs is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_secs, cpu_secs))
    if mem_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    if fsize_bytes is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))


async def _read_stream(stream: asyncio.StreamReader | None, max_len: int | None) -> str:
    """Drain a stream to EOF while retaining at most max_len bytes."""
    if stream is None:
        return ""
    chunks: list[bytes] = []
    curr_len = 0
    truncated = False
    while True:
        chunk = await stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        if max_len is None:
            chunks.append(chunk)
        elif curr_len < max_len:
            remaining = max_len - curr_len
            chunks.append(chunk[:remaining])
            curr_len += min(len(chunk), remaining)
            truncated = truncated or len(chunk) > remaining
        else:
            truncated = True
    text = b"".join(chunks).decode(errors="replace")
    if truncated:
        return text + " [truncated]"
    return text


async def _wait_for_returncode(proc: asyncio.subprocess.Process, timeout: float) -> int:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while proc.returncode is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        await asyncio.sleep(min(0.05, remaining))
    return proc.returncode


def _kill_process_group(pgid: int, logger: logging.Logger) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        logger.debug("process group exited before cleanup")


async def _cancel_tasks(*tasks: asyncio.Task[Any]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command from the specified working directory. "
        "Each command is run in a separate sub-process, "
        "so the working directory and shell state do NOT persist between calls. "
        "Use absolute paths or chain steps with && for multi-step operations. "
        "Returns output as JSON with fields 'exit_code', 'stdout', and 'stderr'. "
        "Background processes in the spawned process group are terminated when the command returns. "
        "Timeouts are configurable per-call using the `timeout` parameter, defaulting to 300 seconds (5 minutes)."
    )
    parameters: dict[str, Any] = {
        "command": {
            "type": "string",
            "description": "The bash command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "(optional) Max time in seconds to allow the command to run before killing it. Default is 300 seconds (5 minutes).",
        },
    }
    required = ["command"]

    def __init__(
        self,
        *,
        working_dir: str,
        max_len: int | None = _DEFAULT_MAX_LEN,
        cpu_secs: int | None = None,
        mem_bytes: int | None = None,
        fsize_bytes: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._working_dir = working_dir
        # max_len=None disables parent-side output retention limits; reserve it
        # for trusted callers because output is joined and JSON-serialized.
        self._max_len = max_len
        self._cpu_secs = cpu_secs
        self._mem_bytes = mem_bytes
        self._fsize_bytes = fsize_bytes

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        command = args["command"]
        timeout = args.get("timeout", 300)

        def preexec_fn() -> None:
            _set_resource_limits(
                cpu_secs=self._cpu_secs,
                mem_bytes=self._mem_bytes,
                fsize_bytes=self._fsize_bytes,
            )

        logger.info(f"launching subprocess with command: {command}")
        # Launch under a shield: create_subprocess_shell fork/execs the shell
        # before it returns, so a cancellation arriving mid-startup would leave a
        # running process that execute() never captured. asyncio's own
        # startup-cancel cleanup only SIGKILLs that single PID (not the group) and
        # then blocks until the pipes close, which a backgrounded child inheriting
        # stdout can hold open indefinitely. Shielding guarantees we learn the
        # pgid so we can kill the whole group before propagating the cancellation.
        create_proc = asyncio.create_task(
            asyncio.create_subprocess_shell(
                command,
                cwd=self._working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec_fn,
            )
        )
        try:
            proc = await asyncio.shield(create_proc)
        except asyncio.CancelledError:
            try:
                proc = await asyncio.wait_for(create_proc, timeout=5)
            except Exception:
                proc = None  # launch itself failed; nothing was spawned to clean up
            if proc is not None:
                _kill_process_group(proc.pid, logger)
            raise
        except Exception as e:
            error_msg = f"failed to launch subprocess: {e}"
            logger.error(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)

        pgid = proc.pid
        read_stdout = asyncio.create_task(_read_stream(proc.stdout, self._max_len))
        read_stderr = asyncio.create_task(_read_stream(proc.stderr, self._max_len))

        try:
            logger.debug(f"waiting for command with timeout: {timeout}s")
            # asyncio's Process.wait() can remain pending after the shell exits if
            # background children inherit stdout/stderr and keep those pipes open.
            exit_code = await _wait_for_returncode(proc, timeout=timeout)
            _kill_process_group(pgid, logger)
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.gather(read_stdout, read_stderr),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                logger.warning("bash: output streams did not close after SIGKILL")
                await _cancel_tasks(read_stdout, read_stderr)
                stdout = stderr = ""
        except asyncio.TimeoutError:
            _kill_process_group(pgid, logger)
            try:
                await asyncio.wait_for(
                    asyncio.gather(read_stdout, read_stderr),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                logger.warning("bash: process did not exit after SIGKILL")
                await _cancel_tasks(read_stdout, read_stderr)
            error_msg = f"timed out after {timeout}s"
            logger.warning(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
        except asyncio.CancelledError:
            _kill_process_group(pgid, logger)
            await _cancel_tasks(read_stdout, read_stderr)
            raise
        except Exception as e:
            _kill_process_group(pgid, logger)
            await _cancel_tasks(read_stdout, read_stderr)
            error_msg = f"unexpected error waiting for command: {e}"
            logger.error(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)

        payload = json.dumps(
            {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
        )
        if exit_code != 0:
            return ToolOutput(output=payload, error=f"exit_code={exit_code}")
        return ToolOutput(output=payload)
