from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import signal
from typing import Any

from model_library.agent.tool import Tool, ToolOutput

_CHUNK_SIZE = 4096
_DEFAULT_MAX_LEN = 1_000_000  # 1 MB


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


async def _read_stream(stream: asyncio.StreamReader | None, max_len: int) -> str:
    """Read from *stream* incrementally, stopping early once *max_len* bytes are collected."""
    if stream is None:
        return ""
    chunks: list[bytes] = []
    curr_len = 0
    while True:
        chunk = await stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        curr_len += len(chunk)
        if curr_len >= max_len:
            break
    text = b"".join(chunks).decode(errors="replace")
    if curr_len >= max_len:
        return text + " [truncated]"
    return text


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command from the specified working directory. "
        "Each command is run in a separate sub-process, "
        "so the working directory and shell state do NOT persist between calls. "
        "Use absolute paths or chain steps with && for multi-step operations. "
        "Returns output as JSON with fields 'exit_code', 'stdout', and 'stderr'. "
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
        max_len: int = _DEFAULT_MAX_LEN,
        cpu_secs: int | None = None,
        mem_bytes: int | None = None,
        fsize_bytes: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._working_dir = working_dir
        self._max_len: int = max_len
        self._cpu_secs = cpu_secs
        self._mem_bytes = mem_bytes
        self._fsize_bytes = fsize_bytes

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        # unpack args
        command = args["command"]
        timeout = args.get("timeout", 300)

        # configures resource limits
        # does nothing if no resource limits specified
        def preexec_fn() -> None:
            _set_resource_limits(
                cpu_secs=self._cpu_secs,
                mem_bytes=self._mem_bytes,
                fsize_bytes=self._fsize_bytes,
            )

        try:
            logger.info(f"launching subprocess with command: {command}")
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self._working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec_fn,
            )
        except Exception as e:
            error_msg = f"failed to launch subprocess: {e}"
            logger.error(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)

        try:
            logger.debug(f"listening for output with timeout: {timeout}s")
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(proc.stdout, self._max_len),
                    _read_stream(proc.stderr, self._max_len),
                ),
                timeout=timeout,
            )
            await proc.wait()
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                logger.debug("process group exited before cleanup")
            await proc.wait()
            error_msg = f"timed out after {timeout}s"
            logger.warning(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            error_msg = f"unexpected error reading output: {e}"
            logger.error(f"bash: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)

        assert proc.returncode is not None  # should always be set after wait()
        exit_code: int = proc.returncode

        payload = json.dumps(
            {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
        )
        if exit_code != 0:
            return ToolOutput(output=payload, error=f"exit_code={exit_code}")
        return ToolOutput(output=payload)
