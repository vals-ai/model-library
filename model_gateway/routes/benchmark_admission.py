import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from model_gateway.benchmark_admission_state import (
    BenchmarkAdmissionConflict,
    BenchmarkAdmissionStore,
)
from model_gateway.benchmark_admission_types import (
    BenchmarkAcquireRequest,
    BenchmarkAdmissionRequest,
    BenchmarkAdmissionResponse,
    BenchmarkReleaseRequest,
    BenchmarkWaitRequest,
)
from model_gateway.cache import ModelCache
from model_gateway.errors import ErrorBody, ErrorResponse, map_exception_to_error
from model_gateway.model_helpers import (
    get_query_llm,
    resolve_gateway_token_retry_params,
)
from model_library.retriers.token import utils as token_utils

logger = logging.getLogger("model_gateway.benchmark_admission")


def _error_response(error: ErrorResponse) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error.body.model_dump(mode="json"),
    )


def _admission_error(status_code: int, code: str, message: str) -> JSONResponse:
    return _error_response(
        ErrorResponse(status_code, ErrorBody(code=code, message=message))
    )


async def _run_admission_operation(
    operation: Callable[
        [BenchmarkAdmissionStore], Awaitable[BenchmarkAdmissionResponse]
    ],
) -> BenchmarkAdmissionResponse | JSONResponse:
    try:
        await token_utils.validate_redis_client()
    except Exception:
        return _admission_error(
            503,
            "benchmark_admission_unavailable",
            "Benchmark admission Redis is unavailable",
        )

    store = BenchmarkAdmissionStore(token_utils.redis_client, logger)
    try:
        return await operation(store)
    except BenchmarkAdmissionConflict as exc:
        return _admission_error(409, "benchmark_admission_conflict", str(exc))
    except RedisError:
        return _admission_error(
            503,
            "benchmark_admission_unavailable",
            "Benchmark admission Redis is unavailable",
        )
    except Exception as exc:
        return _error_response(map_exception_to_error(exc))


def register_benchmark_admission_routes(app: FastAPI, *, cache: ModelCache) -> None:
    @app.post(
        "/benchmark-runs/acquire",
        response_model=BenchmarkAdmissionResponse,
    )
    async def acquire(
        body: BenchmarkAcquireRequest,
    ) -> BenchmarkAdmissionResponse | JSONResponse:
        async def operation(
            store: BenchmarkAdmissionStore,
        ) -> BenchmarkAdmissionResponse:
            resolved_token_retry_params = resolve_gateway_token_retry_params(
                body.model,
                body.token_retry_params,
            )
            llm = await get_query_llm(
                cache,
                body,
                resolved_token_retry_params=resolved_token_retry_params,
            )
            return await store.acquire(
                model=body.model,
                model_registry_key=llm._client_registry_key_model_specific,  # pyright: ignore[reportPrivateUsage]
                run_id=body.run_id,
                effective_token_limit=resolved_token_retry_params.limit,
                total_requests=body.total_requests,
                early_release=body.early_release,
                immediate_queue_release=body.immediate_queue_release,
            )

        return await _run_admission_operation(operation)

    @app.post(
        "/benchmark-runs/wait",
        response_model=BenchmarkAdmissionResponse,
    )
    async def wait(
        body: BenchmarkWaitRequest,
    ) -> BenchmarkAdmissionResponse | JSONResponse:
        async def operation(
            store: BenchmarkAdmissionStore,
        ) -> BenchmarkAdmissionResponse:
            return await store.wait(
                model=body.model,
                run_id=body.run_id,
                timeout_seconds=body.timeout_seconds,
            )

        return await _run_admission_operation(operation)

    @app.post(
        "/benchmark-runs/renew",
        response_model=BenchmarkAdmissionResponse,
    )
    async def renew(
        body: BenchmarkAdmissionRequest,
    ) -> BenchmarkAdmissionResponse | JSONResponse:
        async def operation(
            store: BenchmarkAdmissionStore,
        ) -> BenchmarkAdmissionResponse:
            return await store.renew(model=body.model, run_id=body.run_id)

        return await _run_admission_operation(operation)

    @app.post(
        "/benchmark-runs/release",
        response_model=BenchmarkAdmissionResponse,
    )
    async def release(
        body: BenchmarkReleaseRequest,
    ) -> BenchmarkAdmissionResponse | JSONResponse:
        async def operation(
            store: BenchmarkAdmissionStore,
        ) -> BenchmarkAdmissionResponse:
            return await store.release(
                model=body.model,
                run_id=body.run_id,
                outcome=body.outcome,
            )

        return await _run_admission_operation(operation)
