"""Health route registration."""

from typing import cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register_health_routes(
    app: FastAPI, *, valid_keys: set[str], hmac_secret: bytes
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
        canary = cast(dict[str, str | bool], app.state.startup_canary)
        if canary["enabled"] and canary["status"] != "passed":
            return JSONResponse(
                status_code=503,
                content={"status": f"startup canary {canary['status']}"},
            )
        return {"status": "ok"}
