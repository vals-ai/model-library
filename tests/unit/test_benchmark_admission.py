import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from model_gateway.benchmark_admission_types import (
    DEFAULT_BENCHMARK_WAIT_SECONDS,
    BenchmarkAdmissionOutcome,
    BenchmarkAdmissionResponse,
    BenchmarkCoordinatorError,
)
from model_library.base import TokenRetryParams
from model_library.base.gateway import GatewayLLM
from model_library.retriers.token import benchmark_admission_client
from model_library.retriers.token.benchmark_admission import (
    BenchmarkAdmissionCancelled,
    gateway_benchmark_admission,
)
from model_library.retriers.token.benchmark_queue import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TTL,
)


class FakeCoordinator:
    def __init__(
        self,
        *responses: BenchmarkAdmissionResponse,
        renew_error: Exception | None = None,
        renew_started: asyncio.Event | None = None,
        release_error: Exception | None = None,
        release_outcome: BenchmarkAdmissionOutcome | None = None,
        release_started: asyncio.Event | None = None,
        release_waiter: asyncio.Event | None = None,
    ) -> None:
        self.responses = deque(responses)
        self.renew_error = renew_error
        self.renew_started = renew_started
        self.release_error = release_error
        self.release_outcome: BenchmarkAdmissionOutcome | None = release_outcome
        self.release_started = release_started
        self.release_waiter = release_waiter
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def acquire(self, **kwargs: object) -> BenchmarkAdmissionResponse:
        self.calls.append(("acquire", kwargs))
        return self.responses.popleft()

    async def wait(self, **kwargs: object) -> BenchmarkAdmissionResponse:
        self.calls.append(("wait", kwargs))
        return self.responses.popleft()

    async def renew(self, **kwargs: object) -> BenchmarkAdmissionResponse:
        self.calls.append(("renew", kwargs))
        if self.renew_started is not None:
            self.renew_started.set()
        if self.renew_error is not None:
            raise self.renew_error
        return _response("acquired")

    async def release(
        self,
        *,
        run_id: str,
        outcome: BenchmarkAdmissionOutcome,
    ) -> BenchmarkAdmissionResponse:
        self.calls.append(("release", {"run_id": run_id, "outcome": outcome}))
        if self.release_started is not None:
            self.release_started.set()
        if self.release_waiter is not None:
            await self.release_waiter.wait()
        if self.release_error is not None:
            raise self.release_error
        release_outcome: BenchmarkAdmissionOutcome = (
            outcome if self.release_outcome is None else self.release_outcome
        )
        return _response("released", outcome=release_outcome)


def _response(
    state: Literal["waiting", "acquired", "released"],
    *,
    run_id: str = "run-123",
    outcome: BenchmarkAdmissionOutcome | None = None,
) -> BenchmarkAdmissionResponse:
    return BenchmarkAdmissionResponse(
        state=state,
        model="openai/gpt-4o",
        run_id=run_id,
        effective_token_limit=10_000,
        outcome=outcome,
    )


def _model() -> GatewayLLM:
    return GatewayLLM("gpt-4o", "openai")


def _context(
    coordinator: FakeCoordinator,
    *,
    run_id: str = "run-123",
    enabled: bool = True,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
):
    return patch(
        "model_library.retriers.token.benchmark_admission.GatewayBenchmarkAdmissionClient",
        return_value=coordinator,
    ), gateway_benchmark_admission(
        _model(),
        run_id,
        token_retry_params=TokenRetryParams(
            input_modifier=1.0,
            output_modifier=1.0,
            limit=10_000,
        ),
        enabled=enabled,
        total_requests=12,
        early_release=True,
        immediate_queue_release=True,
        is_cancelled=is_cancelled,
    )


def test_heartbeat_ttl_exceeds_wait_and_initial_renewal_margin() -> None:
    assert HEARTBEAT_TTL > (
        DEFAULT_BENCHMARK_WAIT_SECONDS
        + HEARTBEAT_INTERVAL
        + benchmark_admission_client._LONG_POLL_TIMEOUT_BUFFER_SECONDS
    )


async def test_context_heartbeats_while_waiting_for_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_release = asyncio.Event()
    renew_started = asyncio.Event()
    body_started = asyncio.Event()

    class WaitingCoordinator(FakeCoordinator):
        async def wait(self, **kwargs: object) -> BenchmarkAdmissionResponse:
            self.calls.append(("wait", kwargs))
            await wait_release.wait()
            return self.responses.popleft()

    async def heartbeat_while_live(_delay: float) -> None:
        if renew_started.is_set():
            await asyncio.Event().wait()

    coordinator = WaitingCoordinator(
        _response("waiting"),
        _response("acquired"),
        renew_started=renew_started,
    )
    patched_client, context = _context(coordinator)
    monkeypatch.setattr(asyncio, "sleep", heartbeat_while_live)

    async def run_context() -> None:
        with patched_client:
            async with context:
                body_started.set()

    task = asyncio.create_task(run_context())
    await renew_started.wait()

    assert not body_started.is_set()
    assert [name for name, _kwargs in coordinator.calls[:3]] == [
        "acquire",
        "wait",
        "renew",
    ]

    wait_release.set()
    await task
    assert body_started.is_set()


async def test_context_waits_then_releases_finished() -> None:
    coordinator = FakeCoordinator(_response("waiting"), _response("acquired"))
    patched_client, context = _context(
        coordinator,
        is_cancelled=AsyncMock(return_value=False),
    )

    with patched_client:
        async with context as effective_token_limit:
            assert effective_token_limit == 10_000
            assert [name for name, _kwargs in coordinator.calls] == ["acquire", "wait"]

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "finished"}


async def test_disabled_context_is_noop() -> None:
    with patch(
        "model_library.retriers.token.benchmark_admission.GatewayBenchmarkAdmissionClient",
    ) as client_type:
        async with gateway_benchmark_admission(
            _model(),
            "run-123",
            enabled=False,
        ) as effective_token_limit:
            assert effective_token_limit is None

    client_type.assert_not_called()


async def test_cancellation_before_admission_skips_transport() -> None:
    is_cancelled = AsyncMock(return_value=True)
    with patch(
        "model_library.retriers.token.benchmark_admission.GatewayBenchmarkAdmissionClient",
    ) as client_type:
        with pytest.raises(BenchmarkAdmissionCancelled, match="before admission"):
            async with gateway_benchmark_admission(
                _model(),
                "run-123",
                token_retry_params=TokenRetryParams(
                    input_modifier=1.0,
                    output_modifier=1.0,
                    limit=10_000,
                ),
                is_cancelled=is_cancelled,
            ):
                raise AssertionError("context body must not run")

    client_type.assert_not_called()


async def test_released_cancelled_admission_skips_body() -> None:
    coordinator = FakeCoordinator(_response("released", outcome="cancelled"))
    patched_client, context = _context(coordinator)

    with patched_client:
        with pytest.raises(BenchmarkAdmissionCancelled, match="before admission"):
            async with context:
                raise AssertionError("context body must not run")

    assert [name for name, _kwargs in coordinator.calls] == ["acquire"]


async def test_released_finished_admission_skips_body() -> None:
    coordinator = FakeCoordinator(_response("released", outcome="finished"))
    patched_client, context = _context(coordinator)

    with patched_client:
        with pytest.raises(BenchmarkCoordinatorError, match="already released"):
            async with context:
                raise AssertionError("context body must not run")

    assert [name for name, _kwargs in coordinator.calls] == ["acquire"]


async def test_cancellation_after_admission_releases_cancelled() -> None:
    coordinator = FakeCoordinator(_response("acquired"))
    is_cancelled = AsyncMock(side_effect=[False, True])
    patched_client, context = _context(coordinator, is_cancelled=is_cancelled)

    with patched_client:
        with pytest.raises(BenchmarkAdmissionCancelled, match="after admission"):
            async with context:
                raise AssertionError("context body must not run")

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "cancelled"}


async def test_cancellation_while_waiting_releases_cancelled() -> None:
    coordinator = FakeCoordinator(_response("waiting"))
    is_cancelled = AsyncMock(side_effect=[False, True])
    patched_client, context = _context(coordinator, is_cancelled=is_cancelled)

    with patched_client:
        with pytest.raises(BenchmarkAdmissionCancelled, match="while waiting"):
            async with context:
                raise AssertionError("context body must not run")

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "cancelled"}


async def test_durable_cancellation_after_body_is_raised() -> None:
    coordinator = FakeCoordinator(_response("acquired"))
    is_cancelled = AsyncMock(side_effect=[False, False, True])
    patched_client, context = _context(coordinator, is_cancelled=is_cancelled)

    with patched_client:
        with pytest.raises(BenchmarkAdmissionCancelled, match="after admission"):
            async with context:
                pass

    assert coordinator.calls[-1][1]["outcome"] == "cancelled"


async def test_durable_cancellation_wins_heartbeat_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_started = asyncio.Event()

    async def heartbeat_now(_delay: float) -> None:
        heartbeat_started.set()

    coordinator = FakeCoordinator(
        _response("acquired"),
        renew_error=BenchmarkCoordinatorError("heartbeat expired"),
    )
    is_cancelled = AsyncMock(side_effect=[False, False, True])
    patched_client, context = _context(coordinator, is_cancelled=is_cancelled)
    monkeypatch.setattr(asyncio, "sleep", heartbeat_now)

    with patched_client:
        with pytest.raises(BenchmarkAdmissionCancelled) as exc_info:
            async with context:
                await heartbeat_started.wait()
                await asyncio.Event().wait()

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "cancelled"}
    assert exc_info.value.__notes__ == [
        "Admission heartbeat also failed: heartbeat expired"
    ]


async def test_body_failure_is_preserved_when_release_also_fails() -> None:
    coordinator = FakeCoordinator(
        _response("acquired"),
        release_error=RuntimeError("release failed"),
    )
    patched_client, context = _context(coordinator)

    with patched_client:
        with pytest.raises(ValueError, match="body failed"):
            async with context:
                raise ValueError("body failed")

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "failed"}


async def test_heartbeat_failure_reports_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_started = asyncio.Event()

    async def heartbeat_now(_delay: float) -> None:
        heartbeat_started.set()

    coordinator = FakeCoordinator(
        _response("acquired"),
        renew_error=BenchmarkCoordinatorError("heartbeat expired"),
        release_error=RuntimeError("release failed"),
    )
    patched_client, context = _context(coordinator)
    monkeypatch.setattr(asyncio, "sleep", heartbeat_now)

    with patched_client:
        with pytest.raises(
            BenchmarkCoordinatorError, match="heartbeat expired"
        ) as exc_info:
            async with context:
                await heartbeat_started.wait()
                await asyncio.Event().wait()

    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "failed"}
    assert exc_info.value.__notes__ == ["Admission release also failed: release failed"]


async def test_successful_body_raises_when_server_terminal_outcome_differs() -> None:
    coordinator = FakeCoordinator(
        _response("acquired"),
        release_outcome="failed",
    )
    patched_client, context = _context(coordinator)

    with patched_client:
        with pytest.raises(BenchmarkCoordinatorError, match="terminal outcome"):
            async with context:
                pass


async def test_body_error_remains_primary_when_server_outcome_differs() -> None:
    coordinator = FakeCoordinator(
        _response("acquired"),
        release_outcome="failed",
    )
    patched_client, context = _context(coordinator)

    with patched_client:
        with pytest.raises(ValueError, match="body failed"):
            async with context:
                raise ValueError("body failed")


async def test_task_cancellation_waits_for_release_cleanup() -> None:
    release_started = asyncio.Event()
    release_waiter = asyncio.Event()
    coordinator = FakeCoordinator(
        _response("acquired"),
        release_started=release_started,
        release_waiter=release_waiter,
    )
    body_started = asyncio.Event()

    async def run_context() -> None:
        patched_client, context = _context(coordinator)
        with patched_client:
            async with context:
                body_started.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(run_context())
    await body_started.wait()
    task.cancel()
    await release_started.wait()

    assert not task.done()
    release_name, release_args = coordinator.calls[-1]
    assert release_name == "release"
    assert release_args == {"run_id": "run-123", "outcome": "cancelled"}

    release_waiter.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancellation_during_release_logs_failure_without_masking_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release_started = asyncio.Event()
    release_waiter = asyncio.Event()
    coordinator = FakeCoordinator(
        _response("acquired"),
        release_error=RuntimeError("release failed"),
        release_started=release_started,
        release_waiter=release_waiter,
    )
    body_started = asyncio.Event()

    async def run_context() -> None:
        patched_client, context = _context(coordinator)
        with patched_client:
            async with context:
                body_started.set()
                await asyncio.Event().wait()

    caplog.set_level(
        logging.ERROR,
        logger="model_library.retriers.token.benchmark_admission",
    )
    task = asyncio.create_task(run_context())
    await body_started.wait()
    task.cancel()
    await release_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release_waiter.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    records = [
        record
        for record in caplog.records
        if record.name == "model_library.retriers.token.benchmark_admission"
    ]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "Benchmark admission release failed after caller cancellation for run-123"
    )
    exc_info = records[0].exc_info
    assert exc_info is not None
    assert isinstance(exc_info[1], RuntimeError)
