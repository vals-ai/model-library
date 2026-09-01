"""FastAPI app factory for the model gateway."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from functools import partial
from typing import Literal, cast

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
    MetricPublisher,
    create_metrics_middleware,
    install_telemetry_delivery_metric_handler,
    publish_metrics_periodically,
    record_runtime,
    remove_telemetry_delivery_metric_handler,
)
from model_gateway.observability import (
    install_loop_exception_handler,
    log_process_lifecycle,
    runtime_snapshot,
)
from model_gateway.rate_limit_monitor import RateLimitMonitor
from model_gateway.rate_limit_monitor.routes import register_rate_limit_monitor_routes
from model_gateway.rate_limit_monitor.state import MonitorRedis, RateLimitMonitorStore
from model_gateway.routes.benchmark_admission import register_benchmark_admission_routes
from model_gateway.routes.health import register_health_routes
from model_gateway.routes.models import register_model_routes
from model_gateway.routes.provider_ops import register_provider_ops_routes
from model_gateway.routes.rate_limit import (
    RateLimitProbeService,
    register_rate_limit_route,
)
from model_gateway.routes.query import register_query_routes
from model_gateway.routes.token_retry import register_token_retry_routes
from model_gateway.startup_canary import run_startup_canary, startup_canary_state
from model_gateway.telemetry_helpers import error_telemetry_attributes
from model_gateway.usage_ledger.store import (
    NoopUsageLedger,
    create_usage_ledger_from_env,
)

logger = logging.getLogger("model_proxy_server")

load_dotenv()

# The server must never proxy to itself — clear the client-side gateway URL.
model_library_settings.unset("MODEL_GATEWAY_URL")


GatewayRuntimeRole = Literal["combined", "query", "control"]


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def gateway_runtime_role() -> GatewayRuntimeRole:
    value = os.environ.get("GATEWAY_RUNTIME_ROLE", "combined").strip().lower()
    if value not in {"combined", "query", "control"}:
        raise ValueError("GATEWAY_RUNTIME_ROLE must be combined, query, or control")
    return cast(GatewayRuntimeRole, value)


async def _record_runtime_current(loop: asyncio.AbstractEventLoop) -> None:
    lag_start = loop.time()
    await asyncio.sleep(0)
    event_loop_lag_ms = int((loop.time() - lag_start) * 1000)
    snapshot = runtime_snapshot()
    snapshot["event_loop_lag_ms"] = event_loop_lag_ms
    record_runtime(snapshot)


def create_app() -> FastAPI:
    runtime_role = gateway_runtime_role()
    query_enabled = runtime_role in {"combined", "query"}
    control_enabled = runtime_role in {"combined", "control"}
    api_keys = model_library_settings.get("MODEL_GATEWAY_API_KEYS", None)
    if not isinstance(api_keys, str) or not api_keys:
        raise RuntimeError("MODEL_GATEWAY_API_KEYS must be set")
    api_keys_by_name = cast(dict[str, str], json.loads(api_keys))
    valid_keys = set(api_keys_by_name.values())

    hmac_secret_value = model_library_settings.get("MODEL_GATEWAY_HMAC_SECRET", None)
    if not isinstance(hmac_secret_value, str) or not hmac_secret_value:
        raise RuntimeError("MODEL_GATEWAY_HMAC_SECRET must be set")
    hmac_secret = hmac_secret_value.encode()
    cache = ModelCache()
    capacity_limiter = GatewayCapacityLimiter()
    startup_canary_enabled = query_enabled and env_flag(
        "GATEWAY_STARTUP_CANARY_ENABLED"
    )
    usage_ledger = (
        create_usage_ledger_from_env() if query_enabled else NoopUsageLedger()
    )
    rate_limit_probe_service: RateLimitProbeService | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_url = os.environ.get("REDIS_URL", "")
        if control_enabled and not redis_url:
            raise RuntimeError("REDIS_URL must be set for a control-enabled runtime")

        loop = asyncio.get_running_loop()
        previous_exception_handler = install_loop_exception_handler(loop)
        model_count = len(get_model_names())
        logger.info("Loaded gateway model registry with %s models", model_count)
        log_process_lifecycle("gateway.process.startup")

        redis_client: async_redis.Redis | None = None
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

        metrics_stop = asyncio.Event()
        metrics_publishers: list[MetricPublisher] = [
            partial(_record_runtime_current, loop)
        ]
        if query_enabled:
            metrics_publishers.insert(0, capacity_limiter.record_current)
        metrics_task: asyncio.Task[None] | None = None
        usage_ledger_started = False
        canary_task: asyncio.Task[None] | None = None
        rate_limit_monitor: RateLimitMonitor | None = None
        delivery_metric_handler = install_telemetry_delivery_metric_handler()
        try:
            telemetry.configure_telemetry(app)
            if control_enabled:
                assert redis_client is not None
                rate_limit_monitor = RateLimitMonitor(
                    RateLimitMonitorStore(cast(MonitorRedis, redis_client))
                )
                app.state.rate_limit_monitor = rate_limit_monitor
                rate_limit_monitor.start()
            metrics_task = asyncio.create_task(
                publish_metrics_periodically(
                    metrics_stop,
                    publishers=metrics_publishers,
                )
            )
            if startup_canary_enabled:
                canary_task = asyncio.create_task(
                    run_startup_canary(app, next(iter(api_keys_by_name.values())))
                )
            await usage_ledger.start()
            usage_ledger_started = True
            yield
        finally:
            log_process_lifecycle("gateway.process.shutdown_start")
            if canary_task is not None and not canary_task.done():
                canary_task.cancel()
                with suppress(asyncio.CancelledError):
                    await canary_task
            if rate_limit_probe_service is not None:
                try:
                    await rate_limit_probe_service.close()
                except Exception:
                    logger.exception(
                        "Gateway rate-limit probe service close failed during shutdown"
                    )
            if usage_ledger_started:
                try:
                    await usage_ledger.close()
                except Exception:
                    logger.exception(
                        "Gateway usage ledger close failed during shutdown"
                    )
            if rate_limit_monitor is not None:
                try:
                    await rate_limit_monitor.close()
                except Exception:
                    logger.exception(
                        "Gateway rate-limit monitor close failed during shutdown"
                    )
                finally:
                    app.state.rate_limit_monitor = None
            if redis_client is not None:
                try:
                    await redis_client.aclose()
                except Exception:
                    logger.exception("Gateway Redis close failed during shutdown")
            try:
                telemetry.shutdown_telemetry()
            finally:
                try:
                    remove_telemetry_delivery_metric_handler(delivery_metric_handler)
                finally:
                    metrics_stop.set()
                    try:
                        if metrics_task is not None:
                            await metrics_task
                    finally:
                        loop.set_exception_handler(previous_exception_handler)
                        log_process_lifecycle("gateway.process.shutdown_done")

    app = FastAPI(
        title="Model Proxy",
        lifespan=lifespan,
        redirect_slashes=False,
        docs_url=None if runtime_role == "control" else "/docs",
        redoc_url=None if runtime_role == "control" else "/redoc",
        openapi_url=None if runtime_role == "control" else "/openapi.json",
    )
    app.state.cache = cache
    app.state.hmac_secret = hmac_secret
    app.state.runtime_role = runtime_role
    app.state.capacity_limiter = capacity_limiter
    app.state.usage_ledger = usage_ledger
    app.state.rate_limit_monitor = None
    app.state.startup_canary = startup_canary_state(startup_canary_enabled)
    if query_enabled:
        app.middleware("http")(create_capacity_middleware())
    app.middleware("http")(create_metrics_middleware())
    app.middleware("http")(create_auth_middleware(api_keys_by_name))
    app.add_middleware(GatewayObservabilityMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
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

    register_health_routes(
        app,
        valid_keys=valid_keys,
        hmac_secret=hmac_secret,
        control_enabled=control_enabled,
    )
    if control_enabled:
        register_benchmark_admission_routes(app, cache=cache)
        register_rate_limit_monitor_routes(app)
    if query_enabled:
        register_model_routes(app)
        register_token_retry_routes(app)
        register_query_routes(app, cache=cache)
        rate_limit_probe_service = register_rate_limit_route(app, cache=cache)
        register_provider_ops_routes(app, cache=cache)

    return app
