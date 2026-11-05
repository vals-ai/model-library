"""FastAPI app factory for the model gateway."""

import asyncio
import json
import logging
import os
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from decimal import Decimal
from functools import partial
from typing import Literal, cast
from uuid import UUID

import redis.asyncio as async_redis
from dotenv import load_dotenv
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

import model_library.telemetry as telemetry
from model_library import model_library_settings
from model_library.registry_utils import get_model_names
from model_library.retriers.token import set_redis_client

from model_gateway.asgi_observability import GatewayObservabilityMiddleware
from model_gateway.auth import create_auth_middleware
from model_gateway.budgets.service import BudgetService
from model_gateway.cache import ModelCache
from model_gateway.capacity import GatewayCapacityLimiter, create_capacity_middleware
from model_gateway.errors import ErrorBody, ErrorResponse, RunTokenAuthorizationError
from model_gateway.key_assignments import create_key_assignment_store_from_env
from model_gateway.key_pool_config import pool_config_publisher_from_env
from model_gateway.managed_auth.api import router as managed_auth_router
from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.database import (
    DatabaseSecret,
    build_database_url,
    create_engine as create_managed_engine,
)
from model_gateway.managed_auth.identity import HumanVerifier, MachineVerifier
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.runtime import ManagedAuthorityResolver
from model_gateway.managed_auth.service import ManagedAuthService
from model_gateway.managed_auth.store import ManagedAuthStore
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
    log_failure,
    log_process_lifecycle,
    runtime_snapshot,
    set_latest_runtime_snapshot,
)
from model_gateway.rate_limit_monitor import RateLimitMonitor
from model_gateway.rate_limit_monitor.routes import register_rate_limit_monitor_routes
from model_gateway.rate_limit_monitor.state import MonitorRedis, RateLimitMonitorStore
from model_gateway.routes.benchmark_admission import register_benchmark_admission_routes
from model_gateway.routes.health import register_health_routes
from model_gateway.routes.key_pools import register_key_pool_routes
from model_gateway.routes.models import register_model_routes
from model_gateway.routes.provider_ops import register_provider_ops_routes
from model_gateway.routes.passthrough import register_passthrough_routes
from model_gateway.routes.rate_limit import (
    RateLimitProbeService,
    register_rate_limit_route,
)
from model_gateway.routes.query import register_query_routes
from model_gateway.routes.run_tokens import register_run_token_routes
from model_gateway.routes.token_retry import register_token_retry_routes
from model_gateway.runtime_config import (
    RuntimeConfigState,
    runtime_config_manager_from_env,
)
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


async def _wait_for_initial_authority(
    resolver: ManagedAuthorityResolver, ready: asyncio.Event
) -> None:
    while True:
        try:
            if await resolver.has_published_generation():
                ready.set()
                return
        except (RedisConnectionError, RedisTimeoutError):
            logger.warning("Waiting for initial managed authority publication")
        await asyncio.sleep(0.25)


async def _record_runtime_current(loop: asyncio.AbstractEventLoop) -> None:
    lag_start = loop.time()
    await asyncio.sleep(0)
    event_loop_lag_ms = int((loop.time() - lag_start) * 1000)
    snapshot = runtime_snapshot()
    snapshot["event_loop_lag_ms"] = event_loop_lag_ms
    set_latest_runtime_snapshot(snapshot)
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
    value_counts = Counter(api_keys_by_name.values())
    unique_api_key_names = frozenset(
        name for name, value in api_keys_by_name.items() if value_counts[value] == 1
    )

    hmac_secret_value = model_library_settings.get("MODEL_GATEWAY_HMAC_SECRET", None)
    if not isinstance(hmac_secret_value, str) or not hmac_secret_value:
        raise RuntimeError("MODEL_GATEWAY_HMAC_SECRET must be set")
    hmac_secret = hmac_secret_value.encode()
    cache = ModelCache()
    runtime_config_state = RuntimeConfigState.bootstrap()
    runtime_config_manager = runtime_config_manager_from_env(runtime_config_state)
    capacity_limiter = GatewayCapacityLimiter()
    startup_canary_enabled = query_enabled and env_flag(
        "GATEWAY_STARTUP_CANARY_ENABLED"
    )
    usage_ledger = (
        create_usage_ledger_from_env() if query_enabled else NoopUsageLedger()
    )
    key_assignments = create_key_assignment_store_from_env() if query_enabled else None
    pool_config_publisher = (
        pool_config_publisher_from_env() if control_enabled else None
    )
    rate_limit_probe_service: RateLimitProbeService | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_url = os.environ.get("REDIS_URL", "")
        if not redis_url:
            raise RuntimeError("REDIS_URL must be set for managed authority")
        authority_namespace = os.environ["MODEL_GATEWAY_AUTH_NAMESPACE"]
        managed_engine = None
        if control_enabled:
            managed_database_url = build_database_url(
                cast(
                    DatabaseSecret,
                    json.loads(os.environ["MODEL_GATEWAY_AUTH_DATABASE_SECRET"]),
                ),
                os.environ["MODEL_GATEWAY_AUTH_DATABASE_NAME"],
            )
            managed_engine = create_managed_engine(
                managed_database_url,
                pool_size=int(os.environ["MODEL_GATEWAY_AUTH_POOL_SIZE"]),
            )
        loop = asyncio.get_running_loop()
        previous_exception_handler = install_loop_exception_handler(loop)
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
        key_assignments_started = False
        canary_task: asyncio.Task[None] | None = None
        rate_limit_monitor: RateLimitMonitor | None = None
        runtime_refresh_stop: asyncio.Event | None = None
        runtime_refresh_task: asyncio.Task[None] | None = None
        authority_publisher: ManagedAuthorityPublisher | None = None
        authority_publisher_task: asyncio.Task[None] | None = None
        authority_initial_task: asyncio.Task[None] | None = None
        budget_stop = asyncio.Event()
        budget_tasks: list[asyncio.Task[None]] = []
        delivery_metric_handler = install_telemetry_delivery_metric_handler()
        budget_bucket = (
            os.environ.get("GATEWAY_BUDGET_SNAPSHOT_BUCKET")
            if control_enabled
            else None
        )
        budget_service = BudgetService(
            stage=os.environ.get("GATEWAY_STAGE", "local"),
            redis=redis_client,
            bucket=budget_bucket,
            key=os.environ.get("GATEWAY_MANAGED_BUDGET_SNAPSHOT_KEY")
            if control_enabled
            else None,
        )
        app.state.budget_service = budget_service
        app.state.managed_authority = ManagedAuthorityResolver(
            redis_client, namespace=authority_namespace
        )
        if control_enabled:
            assert managed_engine is not None
            descope_project_id = os.environ["DESCOPE_PROJECT_ID"]
            descope_issuer = os.environ["DESCOPE_EXPECTED_ISSUER"]
            descope_organization = os.environ["DESCOPE_ORGANIZATION_ID"]
            registered_machine_key_id = os.environ["DESCOPE_MACHINE_ACCESS_KEY_ID"]
            machine_service_owner_id = UUID(
                os.environ["DESCOPE_MACHINE_SERVICE_OWNER_ID"]
            )
            descope_credential = os.environ["DESCOPE_MANAGEMENT_CREDENTIAL"]
            app.state.managed_auth_store = ManagedAuthStore(managed_engine)
            app.state.managed_auth_service = ManagedAuthService(
                app.state.managed_auth_store,
                Decimal(
                    os.environ.get(
                        "MODEL_GATEWAY_AUTH_AUTO_APPROVAL_THRESHOLD_USD", "500"
                    )
                ),
            )
            app.state.managed_auth_verifier = HumanVerifier(
                project_id=descope_project_id,
                expected_issuer=descope_issuer,
                required_organization=descope_organization,
                management_credential=descope_credential,
                registered_machine_key_id=registered_machine_key_id,
            )
            app.state.managed_machine_verifier = MachineVerifier(
                project_id=descope_project_id,
                expected_issuer=descope_issuer,
                required_organization=descope_organization,
                registered_access_key_id=registered_machine_key_id,
                service_id=machine_service_owner_id,
            )
            authority_publisher = ManagedAuthorityPublisher(
                managed_engine,
                redis_client,
                HumanReconciler(
                    app.state.managed_auth_store,
                    app.state.managed_auth_verifier,
                    os.environ["DESCOPE_EXPECTED_ISSUER"],
                ),
                namespace=authority_namespace,
            )
        try:
            if authority_publisher is not None:
                authority_publisher_task = asyncio.create_task(
                    authority_publisher.run()
                )
            if runtime_config_manager is not None:
                initial_refresh = await runtime_config_manager.refresh_once()
                if (initial_refresh.failed | initial_refresh.rejected) - {"access"}:
                    raise RuntimeError("Initial runtime configuration refresh failed")
                runtime_refresh_stop = asyncio.Event()
                runtime_refresh_task = asyncio.create_task(
                    runtime_config_manager.run(runtime_refresh_stop)
                )
            if control_enabled:
                try:
                    await budget_service.refresh_control_once()
                except Exception:
                    logger.exception("Initial managed budget refresh failed")
                budget_tasks.append(
                    asyncio.create_task(budget_service.run_control(budget_stop))
                )
            with runtime_config_state.capture():
                model_count = len(get_model_names())
            logger.info("Loaded gateway model registry with %s models", model_count)
            log_process_lifecycle("gateway.process.startup")
            telemetry.configure_telemetry(app)
            if control_enabled:
                rate_limit_monitor = RateLimitMonitor(
                    RateLimitMonitorStore(
                        cast(MonitorRedis, redis_client),
                        stage=os.environ.get("GATEWAY_STAGE", "local"),
                    ),
                    runtime_capture=runtime_config_state.capture,
                )
                app.state.rate_limit_monitor = rate_limit_monitor
                rate_limit_monitor.start()
            metrics_task = asyncio.create_task(
                publish_metrics_periodically(
                    metrics_stop,
                    publishers=metrics_publishers,
                )
            )
            if query_enabled:
                authority_initial_task = asyncio.create_task(
                    _wait_for_initial_authority(
                        app.state.managed_authority, app.state.managed_authority_ready
                    )
                )
            if startup_canary_enabled:
                canary_task = asyncio.create_task(run_startup_canary(app))
            await usage_ledger.start()
            usage_ledger_started = True
            if query_enabled and app.state.passthrough_exchanges is not None:
                await app.state.passthrough_exchanges.start()
            if query_enabled and app.state.passthrough_exchange_log is not None:
                await app.state.passthrough_exchange_log.start()
            if key_assignments is not None:
                await key_assignments.start()
                key_assignments_started = True
            yield
        finally:
            log_process_lifecycle("gateway.process.shutdown_start")
            background_error: Exception | None = None
            try:
                if authority_initial_task is not None:
                    authority_initial_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await authority_initial_task
            except Exception as exc:
                background_error = exc
            try:
                if authority_publisher is not None:
                    await authority_publisher.stop()
                    if authority_publisher_task is not None:
                        await authority_publisher_task
            except Exception as exc:
                if background_error is None:
                    background_error = exc
            if control_enabled:
                await app.state.managed_machine_verifier.aclose()
                await app.state.managed_auth_verifier.aclose()
            if managed_engine is not None:
                await managed_engine.dispose()
            budget_stop.set()
            for task in budget_tasks:
                await task
            if runtime_refresh_stop is not None:
                runtime_refresh_stop.set()
            if runtime_refresh_task is not None:
                await runtime_refresh_task
            if runtime_config_manager is not None:
                await runtime_config_manager.aclose()
            if canary_task is not None and not canary_task.done():
                canary_task.cancel()
                with suppress(asyncio.CancelledError):
                    await canary_task
            if rate_limit_probe_service is not None:
                with log_failure(
                    logger,
                    "Gateway rate-limit probe service close failed during shutdown",
                ):
                    await rate_limit_probe_service.close()
            if query_enabled:
                with log_failure(
                    logger,
                    "Gateway pass-through client close failed during shutdown",
                ):
                    await app.state.passthrough_client.aclose()
                if app.state.passthrough_exchanges is not None:
                    with log_failure(
                        logger,
                        "Gateway pass-through exchange store close failed during shutdown",
                    ):
                        await app.state.passthrough_exchanges.close()
                if app.state.passthrough_exchange_log is not None:
                    with log_failure(
                        logger,
                        "Gateway pass-through exchange log close failed during shutdown",
                    ):
                        await app.state.passthrough_exchange_log.close()
            if usage_ledger_started:
                with log_failure(
                    logger, "Gateway usage ledger close failed during shutdown"
                ):
                    await usage_ledger.close()
            if key_assignments is not None and key_assignments_started:
                with log_failure(
                    logger, "Gateway key assignments close failed during shutdown"
                ):
                    await key_assignments.close()
            if rate_limit_monitor is not None:
                try:
                    with log_failure(
                        logger,
                        "Gateway rate-limit monitor close failed during shutdown",
                    ):
                        await rate_limit_monitor.close()
                finally:
                    app.state.rate_limit_monitor = None
            with log_failure(logger, "Gateway Redis close failed during shutdown"):
                await redis_client.aclose()
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
            if background_error is not None:
                raise background_error

    app = FastAPI(
        title="Model Proxy",
        lifespan=lifespan,
        redirect_slashes=False,
        docs_url=None if runtime_role == "control" else "/docs",
        redoc_url=None if runtime_role == "control" else "/redoc",
        openapi_url=None if runtime_role == "control" else "/openapi.json",
    )
    app.state.cache = cache
    app.state.runtime_config_state = runtime_config_state
    app.state.budget_service = BudgetService(
        stage=os.environ.get("GATEWAY_STAGE", "local"), redis=None
    )
    app.state.unique_api_key_names = unique_api_key_names
    app.state.pool_config_publisher = pool_config_publisher
    app.state.hmac_secret = hmac_secret
    app.state.runtime_role = runtime_role
    app.state.capacity_limiter = capacity_limiter
    app.state.usage_ledger = usage_ledger
    app.state.key_assignments = key_assignments
    app.state.rate_limit_monitor = None
    if query_enabled:
        app.state.managed_authority_ready = asyncio.Event()
    app.state.startup_canary = startup_canary_state(startup_canary_enabled)

    @app.middleware("http")
    async def capture_runtime_configuration(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        with runtime_config_state.capture():
            return await call_next(request)

    if query_enabled:
        app.middleware("http")(create_capacity_middleware())
    app.middleware("http")(create_metrics_middleware())
    app.middleware("http")(create_auth_middleware(api_keys_by_name))
    app.add_middleware(GatewayObservabilityMiddleware)

    @app.exception_handler(RunTokenAuthorizationError)
    async def run_token_authorization_handler(
        request: Request, exc: RunTokenAuthorizationError
    ) -> JSONResponse:
        claims = request.state.run_token_claims
        attributes = {"gateway.error.message": str(exc)}
        if claims is not None:
            attributes["gateway.run_token.run_id"] = claims.run_id
            attributes["gateway.run_token.task_id"] = claims.task_id
        telemetry.add_event("gateway.run_token.violation", attributes)
        return JSONResponse(
            status_code=403,
            content={"code": "model_not_authorized", "message": str(exc)},
        )

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
        query_enabled=query_enabled,
    )
    if control_enabled:
        register_benchmark_admission_routes(app, cache=cache)
        app.include_router(managed_auth_router)
        register_run_token_routes(app)
        register_rate_limit_monitor_routes(app)
    if control_enabled:
        register_model_routes(app)
        register_key_pool_routes(app)
    if query_enabled:
        register_token_retry_routes(app)
        register_query_routes(app, cache=cache)
        register_passthrough_routes(app)
        rate_limit_probe_service = register_rate_limit_route(app, cache=cache)
        register_provider_ops_routes(app, cache=cache)
    return app
