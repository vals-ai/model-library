from pydantic import ValidationError

from model_gateway.benchmark_admission_types import (
    DEFAULT_BENCHMARK_WAIT_SECONDS,
    BenchmarkAcquireRequest,
    BenchmarkAdmissionOutcome,
    BenchmarkAdmissionRequest,
    BenchmarkAdmissionResponse,
    BenchmarkCoordinatorError,
    BenchmarkReleaseRequest,
    BenchmarkWaitRequest,
)
from model_library.base import TokenRetryParams
from model_library.base.gateway import GatewayLLM

_HTTP_TIMEOUT_SECONDS = 10.0
_LONG_POLL_TIMEOUT_BUFFER_SECONDS = 5.0


class GatewayBenchmarkAdmissionClient:
    def __init__(self, model: GatewayLLM) -> None:
        self.gateway = model
        self.model = model.gateway_model_key

    async def acquire(
        self,
        *,
        run_id: str,
        token_retry_params: TokenRetryParams,
        total_requests: int | None,
        early_release: bool,
        immediate_queue_release: bool,
    ) -> BenchmarkAdmissionResponse:
        return await self._post(
            "/benchmark-runs/acquire",
            BenchmarkAcquireRequest(
                model=self.model,
                config=self.gateway.gateway_config,
                run_id=run_id,
                token_retry_params=token_retry_params,
                total_requests=total_requests,
                early_release=early_release,
                immediate_queue_release=immediate_queue_release,
            ),
        )

    async def wait(
        self,
        *,
        run_id: str,
        timeout_seconds: float = DEFAULT_BENCHMARK_WAIT_SECONDS,
    ) -> BenchmarkAdmissionResponse:
        return await self._post(
            "/benchmark-runs/wait",
            BenchmarkWaitRequest(
                model=self.model,
                run_id=run_id,
                timeout_seconds=timeout_seconds,
            ),
            long_poll_seconds=timeout_seconds,
        )

    async def renew(self, *, run_id: str) -> BenchmarkAdmissionResponse:
        result = await self._post(
            "/benchmark-runs/renew",
            BenchmarkAdmissionRequest(model=self.model, run_id=run_id),
        )
        if result.state == "released":
            raise BenchmarkCoordinatorError(
                f"Run {run_id} admission was released during heartbeat"
            )
        return result

    async def release(
        self,
        *,
        run_id: str,
        outcome: BenchmarkAdmissionOutcome,
    ) -> BenchmarkAdmissionResponse:
        result = await self._post(
            "/benchmark-runs/release",
            BenchmarkReleaseRequest(
                model=self.model,
                run_id=run_id,
                outcome=outcome,
            ),
        )
        if result.state != "released":
            raise BenchmarkCoordinatorError("Release returned a non-terminal outcome")
        return result

    async def _post(
        self,
        path: str,
        request: BenchmarkAdmissionRequest | BenchmarkAcquireRequest,
        *,
        long_poll_seconds: float = 0,
    ) -> BenchmarkAdmissionResponse:
        try:
            payload = await self.gateway.post_gateway(
                path,
                request,
                timeout=max(
                    _HTTP_TIMEOUT_SECONDS,
                    long_poll_seconds + _LONG_POLL_TIMEOUT_BUFFER_SECONDS,
                ),
            )
        except Exception as exc:
            raise BenchmarkCoordinatorError(str(exc)) from exc

        try:
            result = BenchmarkAdmissionResponse.model_validate(payload)
        except ValidationError as exc:
            raise BenchmarkCoordinatorError(
                f"Invalid coordinator success response: {exc}"
            ) from exc

        if result.model != request.model or result.run_id != request.run_id:
            raise BenchmarkCoordinatorError(
                "Coordinator response identity does not match the request"
            )
        return result
