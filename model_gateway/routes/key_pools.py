"""Control routes for operator key pools: read and replace the whole document."""

from dataclasses import asdict
from time import time_ns
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_library import model_library_settings
from model_library.register_models import get_model_registry

from model_gateway.key_pool_config import (
    KeyCatalog,
    KeyPoolReplacement,
    PoolConfigDeploymentInProgressError,
    PoolConfigSnapshot,
    PoolConfigPublisher,
    PublishedPoolDocument,
    key_catalog,
    pool_config_from_document,
)
from model_gateway.key_pool_validation import validate_mapping
from model_gateway.observability import log_gateway_event
from model_gateway.runtime_config import (
    runtime_admin_api_key_names,
    runtime_credential_fields,
    runtime_pool_snapshot,
)

ROUTE = "/admin/key-pools"


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"code": code, "message": message}
    )


def _no_pools_profile() -> JSONResponse:
    return _error_response(
        503, "pool_config_unavailable", "This Gateway serves no pools profile"
    )


def _admin_denied(request: Request) -> JSONResponse | None:
    # Run tokens never reach here: auth refuses them on every /admin path.
    key_name = request.state.gateway_api_key_name
    unique_names = cast(frozenset[str], request.app.state.unique_api_key_names)
    if key_name not in runtime_admin_api_key_names() or key_name not in unique_names:
        return _error_response(
            403, "forbidden", "This Gateway key cannot manage key pools"
        )
    return None


def _served_catalog(snapshot: PoolConfigSnapshot) -> KeyCatalog:
    return key_catalog(
        snapshot.document.pools,
        runtime_credential_fields(),
        model_library_settings.get,
    )


def register_key_pool_routes(app: FastAPI) -> None:
    @app.get(ROUTE)
    async def get_key_pools(request: Request) -> Any:
        if (denied := _admin_denied(request)) is not None:
            return denied
        snapshot = runtime_pool_snapshot()
        if snapshot is None:
            return _no_pools_profile()
        catalog = _served_catalog(snapshot)
        return {
            "version": str(snapshot.document.revision),
            **snapshot.document.model_dump(exclude={"revision"}),
            "keys": [
                {
                    "name": name,
                    "provider": catalog.providers[name],
                    "internal": name in catalog.internal,
                }
                for name in sorted(catalog.providers)
            ],
        }

    @app.put(ROUTE)
    async def put_key_pools(request: Request, document: KeyPoolReplacement) -> Any:
        if (denied := _admin_denied(request)) is not None:
            return denied
        publisher = cast(
            PoolConfigPublisher | None, request.app.state.pool_config_publisher
        )
        snapshot = runtime_pool_snapshot()
        if publisher is None or snapshot is None:
            return _no_pools_profile()
        catalog = _served_catalog(snapshot)
        errors = validate_mapping(
            pool_config_from_document(document),
            registry=get_model_registry(),
            known_keys=catalog.known_keys(),
            internal_keys=catalog.internal,
            pooled_providers=catalog.pooled_providers,
            described=document.descriptions,
        )
        if errors:
            # Dashboards reads a 422 with the errors list (vals-ai/dashboards#208);
            # a body of the wrong shape keeps the Gateway's 400 `invalid_request`.
            return JSONResponse(
                status_code=422,
                content={"errors": [asdict(error) for error in errors]},
            )
        # As the release pipeline numbers model revisions: the clock in
        # microseconds, so a PUT made before this task sees the last one still
        # outranks it on every worker.
        published = PublishedPoolDocument(
            **document.model_dump(),
            revision=max(snapshot.document.revision + 1, time_ns() // 1_000),
        )
        try:
            configuration_version = await publisher.publish(
                published.model_dump_json().encode()
            )
        except PoolConfigDeploymentInProgressError:
            return _error_response(
                409,
                "pool_config_deployment_in_progress",
                "Another AppConfig deployment is in progress; retry in a minute",
            )
        log_gateway_event(
            "gateway.key_pools.published",
            revision=published.revision,
            configuration_version=configuration_version,
            published_by=request.state.gateway_api_key_name,
        )
        return {"version": str(published.revision)}
