"""FastAPI app factory for the model gateway."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from functools import partial

import redis.asyncio as async_redis
from dotenv import load_dotenv
from redis.exceptions import TimeoutError as RedisTimeoutError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from model_library import model_library_settings
from model_library.registry_utils import get_model_names
from model_library.retriers.token import set_redis_client

from model_gateway.asgi_observability import GatewayObservabilityMiddleware
from model_gateway.auth import create_auth_middleware
from model_gateway.cache import ModelCache
from model_gateway.capacity import GatewayCapacityLimiter, create_capacity_middleware
from model_gateway.errors import ErrorBody, ErrorResponse
from model_gateway.metrics import (
    create_metrics_middleware,
    publish_metrics_periodically,
    record_runtime,
)
from model_gateway.observability import (
    install_loop_exception_handler,
    log_process_lifecycle,
    runtime_snapshot,
)
from model_gateway.routes.health import register_health_routes
from model_gateway.routes.models import register_model_routes
from model_gateway.routes.provider_ops import register_provider_ops_routes
from model_gateway.routes.query import register_query_routes
from model_gateway.routes.token_retry import register_token_retry_routes
from model_gateway.startup_canary import run_startup_canary, startup_canary_state
from model_gateway.telemetry_helpers import error_telemetry_attributes
from model_gateway.usage_ledger.store import create_usage_ledger_from_env

logger = logging.getLogger("model_proxy_server")

load_dotenv()

# The server must never proxy to itself — clear the client-side gateway URL.
model_library_settings.unset("MODEL_GATEWAY_URL")


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _record_runtime_current(loop: asyncio.AbstractEventLoop) -> None:
    lag_start = loop.time()
    await asyncio.sleep(0)
    event_loop_lag_ms = int((loop.time() - lag_start) * 1000)
    snapshot = runtime_snapshot()
    snapshot["event_loop_lag_ms"] = event_loop_lag_ms
    record_runtime(snapshot)


def create_app() -> FastAPI:
    api_keys = model_library_settings.get("MODEL_GATEWAY_API_KEYS", None)
    if not isinstance(api_keys, str):
        raise RuntimeError("MODEL_GATEWAY_API_KEYS must be set")
    valid_keys = {k for k in api_keys.split(",") if k}
    if not valid_keys:
        raise RuntimeError("MODEL_GATEWAY_API_KEYS must be set")

    hmac_secret_value = model_library_settings.get("MODEL_GATEWAY_HMAC_SECRET", None)
    if not isinstance(hmac_secret_value, str) or not hmac_secret_value:
        raise RuntimeError("MODEL_GATEWAY_HMAC_SECRET must be set")
    hmac_secret = hmac_secret_value.encode()
    cache = ModelCache()
    capacity_limiter = GatewayCapacityLimiter()
    startup_canary_enabled = env_flag("GATEWAY_STARTUP_CANARY_ENABLED")
    usage_ledger = create_usage_ledger_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        previous_exception_handler = install_loop_exception_handler(loop)
        model_count = len(get_model_names())
        logger.info("Loaded gateway model registry with %s models", model_count)
        log_process_lifecycle("gateway.process.startup")
        telemetry.configure_telemetry(app)

        redis_client: async_redis.Redis | None = None
        redis_url = os.environ.get("REDIS_URL", "")
        if redis_url:
            redis_client = async_redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                redis_url,
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=10,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_error=[RedisTimeoutError],
                max_connections=200,
            )
            set_redis_client(redis_client)

        app.state.cache = cache
        app.state.hmac_secret = hmac_secret
        app.state.capacity_limiter = capacity_limiter
        app.state.usage_ledger = usage_ledger

        metrics_stop = asyncio.Event()
        metrics_task = asyncio.create_task(
            publish_metrics_periodically(
                metrics_stop,
                publishers=[
                    capacity_limiter.record_current,
                    partial(_record_runtime_current, loop),
                ],
            )
        )
        usage_ledger_started = False
        canary_task: asyncio.Task[None] | None = None
        if startup_canary_enabled:
            canary_task = asyncio.create_task(
                run_startup_canary(app, sorted(valid_keys)[0])
            )
        try:
            await usage_ledger.start()
            usage_ledger_started = True
            yield
        finally:
            log_process_lifecycle("gateway.process.shutdown_start")
            if canary_task is not None and not canary_task.done():
                canary_task.cancel()
                with suppress(asyncio.CancelledError):
                    await canary_task
            metrics_stop.set()
            await metrics_task
            if usage_ledger_started:
                try:
                    await usage_ledger.close()
                except Exception:
                    logger.exception(
                        "Gateway usage ledger close failed during shutdown"
                    )
            if redis_client is not None:
                try:
                    await redis_client.aclose()
                except Exception:
                    logger.exception("Gateway Redis close failed during shutdown")
            telemetry.shutdown_telemetry()
            loop.set_exception_handler(previous_exception_handler)
            log_process_lifecycle("gateway.process.shutdown_done")

    app = FastAPI(title="Model Proxy", lifespan=lifespan)
    app.state.cache = cache
    app.state.hmac_secret = hmac_secret
    app.state.capacity_limiter = capacity_limiter
    app.state.usage_ledger = usage_ledger
    app.state.startup_canary = startup_canary_state(startup_canary_enabled)
    app.middleware("http")(create_capacity_middleware())
    app.middleware("http")(create_metrics_middleware())
    app.middleware("http")(create_auth_middleware(valid_keys))
    app.add_middleware(GatewayObservabilityMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        err = ErrorResponse(
            400,
            ErrorBody(code="invalid_request", message=str(exc)),
        )
        error_attrs = error_telemetry_attributes(err, phase="request_validation")
        telemetry.set_attributes(error_attrs)
        telemetry.set_status_error(err.body.code)
        telemetry.add_event("gateway.request_validation.error", error_attrs)
        return JSONResponse(status_code=err.status_code, content=err.body.model_dump())

    register_health_routes(app, valid_keys=valid_keys, hmac_secret=hmac_secret)
    register_model_routes(app)
    register_token_retry_routes(app)
    register_query_routes(app, cache=cache)
    register_provider_ops_routes(app, cache=cache)

    return app
