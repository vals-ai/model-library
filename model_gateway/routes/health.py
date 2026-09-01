"""Health route registration."""

from typing import cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from model_gateway.rate_limit_monitor.manager import RateLimitMonitor
from model_library.retriers.token import utils as token_utils


def register_health_routes(
    app: FastAPI,
    *,
    valid_keys: set[str],
    hmac_secret: bytes,
    control_enabled: bool = False,
) -> None:
    @app.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready():
        if not valid_keys:
            return JSONResponse(status_code=503, content={"status": "no gateway keys"})
        if not hmac_secret:
            return JSONResponse(status_code=503, content={"status": "no hmac secret"})
        if control_enabled:
            try:
                await token_utils.validate_redis_client()
                await token_utils.redis_client.ping()
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={"status": "redis unavailable"},
                )
            monitor = cast(RateLimitMonitor, app.state.rate_limit_monitor)
            monitor.check_health()
        canary = cast(dict[str, str | bool], app.state.startup_canary)
        if canary["enabled"] and canary["status"] != "passed":
            return JSONResponse(
                status_code=503,
                content={"status": f"startup canary {canary['status']}"},
            )
        return {"status": "ok"}
