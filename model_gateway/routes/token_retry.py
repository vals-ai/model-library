"""Token-retry route registration."""

import asyncio
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from model_library.retriers.token.utils import (
    get_status as get_token_retry_status,
    validate_redis_client,
)

TOKEN_RETRY_STATUS_CACHE_SECONDS = 1.0


def register_token_retry_routes(app: FastAPI) -> None:
    token_retry_status_cache: dict[str, Any] | None = None
    token_retry_status_cache_expires_at = 0.0
    token_retry_status_cache_lock = asyncio.Lock()

    @app.get("/token-retry/status")
    async def token_retry_status():
        nonlocal token_retry_status_cache, token_retry_status_cache_expires_at
        try:
            await validate_redis_client()
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"code": "redis_not_configured", "message": str(exc)},
            )

        now = time.monotonic()
        if (
            token_retry_status_cache is not None
            and now < token_retry_status_cache_expires_at
        ):
            return token_retry_status_cache

        async with token_retry_status_cache_lock:
            now = time.monotonic()
            if (
                token_retry_status_cache is not None
                and now < token_retry_status_cache_expires_at
            ):
                return token_retry_status_cache

            token_retry_status_cache = (await get_token_retry_status()).model_dump(
                mode="json"
            )
            token_retry_status_cache_expires_at = (
                time.monotonic() + TOKEN_RETRY_STATUS_CACHE_SECONDS
            )
            return token_retry_status_cache
