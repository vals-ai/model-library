import asyncio
import logging
from asyncio import CancelledError
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

from model_gateway.benchmark_admission_types import (
    BenchmarkAdmissionOutcome,
    BenchmarkAdmissionResponse,
    BenchmarkCoordinatorError,
)
from model_library.base import TokenRetryParams
from model_library.base.gateway import GatewayLLM
from model_library.retriers.token.benchmark_admission_client import (
    GatewayBenchmarkAdmissionClient,
)

CancellationCheck = Callable[[], Awaitable[bool]]
HEARTBEAT_INTERVAL_SECONDS = 2
logger = logging.getLogger(__name__)


class BenchmarkAdmissionCancelled(RuntimeError):
    pass


async def _run_heartbeats(
    client: GatewayBenchmarkAdmissionClient,
    run_id: str,
    owner: asyncio.Task[object],
) -> None:
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            await client.renew(run_id=run_id)
    except Exception:
        owner.cancel()
        raise


async def _release_admission(
    client: GatewayBenchmarkAdmissionClient,
    run_id: str,
    outcome: BenchmarkAdmissionOutcome,
) -> BenchmarkAdmissionResponse:
    release_task = asyncio.create_task(client.release(run_id=run_id, outcome=outcome))
    try:
        return await asyncio.shield(release_task)
    except CancelledError:
        try:
            await release_task
        except CancelledError:
            logger.warning(
                "Benchmark admission release was cancelled for %s",
                run_id,
            )
        except Exception:
            logger.exception(
                "Benchmark admission release failed after caller cancellation for %s",
                run_id,
            )
        raise


@asynccontextmanager
async def gateway_benchmark_admission(
    model: GatewayLLM,
    run_id: str,
    *,
    token_retry_params: TokenRetryParams | None = None,
    enabled: bool = True,
    total_requests: int | None = None,
    early_release: bool = True,
    immediate_queue_release: bool = False,
    is_cancelled: CancellationCheck | None = None,
) -> AsyncGenerator[int | None, None]:
    if not enabled:
        yield None
        return

    if token_retry_params is None:
        raise BenchmarkCoordinatorError(
            "Benchmark admission requires token retry parameters"
        )

    if is_cancelled is not None and await is_cancelled():
        raise BenchmarkAdmissionCancelled(f"Run {run_id} cancelled before admission")

    client = GatewayBenchmarkAdmissionClient(model)
    admission = await client.acquire(
        run_id=run_id,
        token_retry_params=token_retry_params,
        total_requests=total_requests,
        early_release=early_release,
        immediate_queue_release=immediate_queue_release,
    )
    if admission.state == "released":
        if admission.outcome == "cancelled":
            raise BenchmarkAdmissionCancelled(
                f"Run {run_id} cancelled before admission"
            )
        raise BenchmarkCoordinatorError(
            f"Run {run_id} admission was already released as {admission.outcome}"
        )

    owner = cast(asyncio.Task[object], asyncio.current_task())
    heartbeat_task = asyncio.create_task(_run_heartbeats(client, run_id, owner))
    body_error: BaseException | None = None
    heartbeat_error: Exception | None = None

    try:
        while admission.state == "waiting":
            if is_cancelled is not None and await is_cancelled():
                raise BenchmarkAdmissionCancelled(
                    f"Run {run_id} cancelled while waiting for admission"
                )
            admission = await client.wait(run_id=run_id)
            if admission.state == "released":
                if admission.outcome == "cancelled":
                    raise BenchmarkAdmissionCancelled(
                        f"Run {run_id} cancelled while waiting for admission"
                    )
                raise BenchmarkCoordinatorError(
                    f"Run {run_id} admission was released as {admission.outcome}"
                )

        if is_cancelled is not None and await is_cancelled():
            raise BenchmarkAdmissionCancelled(f"Run {run_id} cancelled after admission")
        yield admission.effective_token_limit
    except BaseException as exc:
        body_error = exc
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError as _:
            pass
        except Exception as exc:
            heartbeat_error = exc

    cancellation_error: BaseException | None = None
    durable_cancelled = False
    if (
        body_error is None or isinstance(body_error, asyncio.CancelledError)
    ) and is_cancelled is not None:
        try:
            durable_cancelled = await is_cancelled()
        except BaseException as exc:
            cancellation_error = exc

    outcome_error = cancellation_error or body_error
    if durable_cancelled or isinstance(outcome_error, BenchmarkAdmissionCancelled):
        outcome = "cancelled"
    elif heartbeat_error is not None:
        outcome = "failed"
    elif isinstance(outcome_error, asyncio.CancelledError):
        outcome = "cancelled"
    else:
        outcome = "failed" if outcome_error is not None else "finished"

    primary_error = body_error or heartbeat_error or cancellation_error
    if durable_cancelled:
        primary_error = BenchmarkAdmissionCancelled(
            f"Run {run_id} cancelled after admission"
        )
        if heartbeat_error is not None:
            primary_error.add_note(
                f"Admission heartbeat also failed: {heartbeat_error}"
            )
    elif isinstance(body_error, asyncio.CancelledError) and heartbeat_error is not None:
        primary_error = heartbeat_error

    try:
        released = await _release_admission(client, run_id, outcome)
        if released.outcome != outcome:
            mismatch_error = BenchmarkCoordinatorError(
                f"Run {run_id} terminal outcome is {released.outcome}, not {outcome}"
            )
            if primary_error is None:
                primary_error = mismatch_error
            else:
                primary_error.add_note(str(mismatch_error))
    except BaseException as exc:
        if primary_error is None:
            raise
        primary_error.add_note(f"Admission release also failed: {exc}")

    if primary_error is not None:
        raise primary_error
