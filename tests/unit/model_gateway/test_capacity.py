import asyncio
from types import SimpleNamespace

import pytest

from model_gateway import capacity, metrics
from model_gateway.capacity import (
    CapacityQueueTimeoutError,
    CapacityRejectedError,
    GatewayCapacityLimiter,
    GatewayRequestTimeoutError,
)


@pytest.mark.asyncio
async def test_capacity_attach_request_identity_sets_searchable_fields(monkeypatch):
    seen: dict[str, object | None] = {}

    class FakeRequest:
        url = SimpleNamespace(path="/query")
        headers: dict[str, str] = {}

        async def json(self):
            return {
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hello"}],
                "run_id": "run-a",
                "question_id": "q1",
                "query_id": "query-a",
                "identity": {"custom": "value"},
            }

    def fake_set_attributes(attrs: dict[str, object | None]) -> None:
        seen.update(attrs)

    monkeypatch.setattr(capacity.telemetry, "is_recording", lambda: True)
    monkeypatch.setattr(capacity.telemetry, "set_attributes", fake_set_attributes)

    await capacity._attach_request_identity(FakeRequest())  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert seen["run_id"] == "run-a"
    assert seen["question_id"] == "q1"
    assert seen["query_id"] == "query-a"
    assert seen["identity"] == '{"custom":"value"}'
    assert seen["gen_ai.request.model"] == "openai/gpt-4o"
    assert seen["gateway.operation"] == "query"
    assert "gateway.request_json" not in seen


@pytest.mark.asyncio
async def test_capacity_attach_request_identity_generates_missing_query_id(monkeypatch):
    seen: dict[str, object | None] = {}

    class FakeRequest:
        url = SimpleNamespace(path="/query")
        headers: dict[str, str] = {}

        async def json(self):
            return {
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hello"}],
                "run_id": "run-a",
                "question_id": "q1",
            }

    monkeypatch.setattr(capacity.telemetry, "is_recording", lambda: True)
    monkeypatch.setattr(capacity.telemetry, "set_attributes", seen.update)

    await capacity._attach_request_identity(FakeRequest())  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert seen["run_id"] == "run-a"
    assert seen["question_id"] == "q1"
    assert isinstance(seen["query_id"], str)
    assert len(seen["query_id"]) == 14


@pytest.mark.asyncio
@pytest.mark.parametrize("query_id", ["", " ", "\t"])
async def test_capacity_attach_request_identity_generates_blank_query_id(
    monkeypatch, query_id: str
):
    seen: dict[str, object | None] = {}

    class FakeRequest:
        url = SimpleNamespace(path="/query")
        headers: dict[str, str] = {}

        async def json(self):
            return {
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hello"}],
                "run_id": "run-a",
                "question_id": "q1",
                "query_id": query_id,
            }

    monkeypatch.setattr(capacity.telemetry, "is_recording", lambda: True)
    monkeypatch.setattr(capacity.telemetry, "set_attributes", seen.update)

    await capacity._attach_request_identity(FakeRequest())  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert seen["run_id"] == "run-a"
    assert seen["question_id"] == "q1"
    assert isinstance(seen["query_id"], str)
    assert seen["query_id"].strip()
    assert len(seen["query_id"]) == 14
    assert seen["query_id"] != query_id


@pytest.mark.asyncio
async def test_capacity_attach_request_identity_skips_large_body_parse(monkeypatch):
    seen: dict[str, object | None] = {}

    class FakeRequest:
        url = SimpleNamespace(path="/files/upload")
        headers = {"content-length": str(capacity.MAX_CAPACITY_IDENTITY_BODY_BYTES + 1)}

        async def json(self):
            raise AssertionError("large rejected bodies should not be parsed")

    monkeypatch.setattr(capacity.telemetry, "is_recording", lambda: True)
    monkeypatch.setattr(capacity.telemetry, "set_attributes", seen.update)

    await capacity._attach_request_identity(FakeRequest())  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert seen == {
        "gateway.route": "/files/upload",
        "gateway.operation": "files_upload",
        "http.request.body.size": capacity.MAX_CAPACITY_IDENTITY_BODY_BYTES + 1,
    }


@pytest.mark.asyncio
async def test_capacity_rejects_when_active_and_queue_are_full():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=0,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    blocker = asyncio.Event()

    async def wait_forever():
        await blocker.wait()

    active_task = asyncio.create_task(limiter.run(wait_forever))
    await asyncio.sleep(0)

    with pytest.raises(CapacityRejectedError):
        await limiter.run(wait_forever)

    blocker.set()
    await active_task
    metrics.flush_metrics()


@pytest.mark.asyncio
async def test_capacity_queued_request_runs_when_slot_is_released():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=1,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first():
        order.append("first-start")
        first_started.set()
        await release_first.wait()
        order.append("first-end")

    async def second():
        order.append("second")
        return "ok"

    first_task = asyncio.create_task(limiter.run(first))
    await first_started.wait()
    second_task = asyncio.create_task(limiter.run(second))
    await asyncio.sleep(0)
    assert limiter.snapshot().queued == 1

    release_first.set()
    assert await second_task == "ok"
    await first_task
    assert order == ["first-start", "first-end", "second"]
    metrics.flush_metrics()


@pytest.mark.asyncio
async def test_capacity_new_arrival_does_not_bypass_existing_waiter():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=2,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    existing_waiter = object()
    async with limiter._condition:  # pyright: ignore[reportPrivateUsage]
        limiter._waiters.append(existing_waiter)  # pyright: ignore[reportPrivateUsage]
        limiter._queued = 1  # pyright: ignore[reportPrivateUsage]

    async def operation():
        return "ok"

    task = asyncio.create_task(limiter.run(operation))
    await asyncio.sleep(0)

    assert not task.done()
    assert limiter.snapshot().queued == 2

    async with limiter._condition:  # pyright: ignore[reportPrivateUsage]
        limiter._waiters.popleft()  # pyright: ignore[reportPrivateUsage]
        limiter._queued -= 1  # pyright: ignore[reportPrivateUsage]
        limiter._condition.notify_all()  # pyright: ignore[reportPrivateUsage]

    assert await task == "ok"
    metrics.flush_metrics()


@pytest.mark.asyncio
async def test_capacity_queue_timeout():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=1,
        queue_timeout_seconds=0.01,
        request_timeout_seconds=1,
    )
    blocker = asyncio.Event()

    async def wait_forever():
        await blocker.wait()

    active_task = asyncio.create_task(limiter.run(wait_forever))
    await asyncio.sleep(0)

    with pytest.raises(CapacityQueueTimeoutError):
        await limiter.run(wait_forever)

    blocker.set()
    await active_task
    metrics.flush_metrics()


@pytest.mark.asyncio
async def test_capacity_request_timeout_does_not_wrap_provider_timeout():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=0,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )

    async def provider_timeout():
        raise TimeoutError("provider timed out")

    with pytest.raises(TimeoutError, match="provider timed out"):
        await limiter.run(provider_timeout)
    metrics.flush_metrics()


@pytest.mark.asyncio
async def test_capacity_request_timeout_cancels_slow_operation():
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=0,
        queue_timeout_seconds=1,
        request_timeout_seconds=0.01,
    )
    cancelled = False

    async def slow_operation():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    with pytest.raises(GatewayRequestTimeoutError):
        await limiter.run(slow_operation)
    assert cancelled
    metrics.flush_metrics()
