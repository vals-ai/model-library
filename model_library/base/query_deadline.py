"""Process-local query deadline enforcement."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import math

from model_library.exceptions import QueryDeadlineExceededError


@asynccontextmanager
async def query_deadline_scope(deadline: float | None) -> AsyncGenerator[None]:
    """Enforce an optional absolute deadline from the current event loop clock."""
    if deadline is not None:
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if deadline <= asyncio.get_running_loop().time():
            raise QueryDeadlineExceededError()

    timeout = asyncio.timeout_at(deadline)
    try:
        async with timeout:
            yield
    except TimeoutError as exc:
        if not timeout.expired():
            raise
        raise QueryDeadlineExceededError() from exc
