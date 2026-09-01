"""HTTP routes for the shared rate-limit monitor."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_gateway.rate_limit_monitor.manager import (
    MonitorInvalidModel,
    RateLimitMonitor,
)
from model_gateway.rate_limit_monitor.types import (
    MonitorActivationRequest,
    MonitorActivationResponse,
    MonitorListResponse,
)

OPERATION_TIMEOUT_SECONDS = 5.0


def _error_response(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
    )


def register_rate_limit_monitor_routes(app: FastAPI) -> None:
    @app.get(
        "/rate-limit-monitor",
        response_model=MonitorListResponse,
    )
    async def list_rate_limit_monitor(
        request: Request,
    ) -> MonitorListResponse:
        monitor = cast(RateLimitMonitor, request.app.state.rate_limit_monitor)
        async with asyncio.timeout(OPERATION_TIMEOUT_SECONDS):
            return await monitor.list_states()

    @app.post(
        "/rate-limit-monitor/activate",
        response_model=MonitorActivationResponse,
    )
    async def activate_rate_limit_monitor(
        body: MonitorActivationRequest,
        request: Request,
    ) -> MonitorActivationResponse | JSONResponse:
        monitor = cast(RateLimitMonitor, request.app.state.rate_limit_monitor)
        try:
            async with asyncio.timeout(OPERATION_TIMEOUT_SECONDS):
                return await monitor.activate(body.model)
        except MonitorInvalidModel:
            return _error_response(
                400,
                "invalid_model",
                "Invalid model activation request",
            )
