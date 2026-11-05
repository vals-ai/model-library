"""Bearer token auth middleware."""

import hashlib
import hmac
from collections.abc import Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from model_gateway.managed_auth.credentials import managed_key_verifier
from model_gateway.managed_auth.permissions import intersect_permissions
from model_gateway.run_token_types import PASSTHROUGH_PREFIX, RUN_TOKEN_OPERATION_PATHS
from model_gateway.run_tokens import (
    RUN_TOKEN_PREFIX,
    LegacyRunTokenClaims,
    get_run_token,
    sha256_hex,
)

EXEMPT_PATHS = {"/health/live", "/health/ready"}
TRACE_AUTH_FAILURE_PATHS = telemetry.HTTP_TRACE_ALLOWED_ROUTES

# Read-only routes every run token may call regardless of its `allowed_operations`.
RUN_TOKEN_READ_PATHS = frozenset({"/registry"})
RUN_TOKEN_ALLOWED_PATHS = frozenset(
    {*RUN_TOKEN_OPERATION_PATHS.values(), *RUN_TOKEN_READ_PATHS}
)
MANAGED_ROOT_ALLOWED_PATHS = frozenset(
    {
        *RUN_TOKEN_OPERATION_PATHS.values(),
        "/registry",
        "/service-auth",
        "/service-auth/revoke",
    }
)
OPERATION_BY_PATH = {
    path: operation for operation, path in RUN_TOKEN_OPERATION_PATHS.items()
}


def _presented_credential(request: Request, passthrough: bool) -> str | None:
    authorization = request.headers.get("Authorization")
    if authorization is not None and authorization.startswith("Bearer "):
        return authorization[7:]
    if not passthrough:
        return None
    # Google SDKs can also send the key as `?key=`, which isn't read here: query strings
    # end up in uvicorn and load balancer access logs.
    for name in ("x-api-key", "x-goog-api-key"):
        credential = request.headers.get(name)
        if credential is not None:
            return credential
    return None


def create_auth_middleware(api_keys_by_name: Mapping[str, str]):
    # Pre-hash keys for constant-time comparison (prevents timing side-channel)
    hashed_keys = tuple(
        (name, hashlib.sha256(api_key.encode()).digest())
        for name, api_key in api_keys_by_name.items()
    )
    accepted_digests = {digest.hex() for _, digest in hashed_keys}

    async def auth_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if (
            request.url.path in EXEMPT_PATHS
            or request.url.path.startswith("/admin/managed-auth/")
            or request.url.path == "/admin/managed-auth"
        ):
            return await call_next(request)

        passthrough = request.url.path.startswith(f"{PASSTHROUGH_PREFIX}/")
        presented = _presented_credential(request, passthrough)
        if presented is None or (passthrough and not presented):
            return _unauthorized_response(
                request,
                "Missing or malformed Authorization header",
            )

        request.state.run_token_claims = None
        request.state.managed_authority = None

        if passthrough and not presented.startswith(RUN_TOKEN_PREFIX):
            return _auth_error_response(
                request, 403, "forbidden", "Pass-through routes accept only run tokens"
            )

        if presented.startswith(RUN_TOKEN_PREFIX):
            operation_path = PASSTHROUGH_PREFIX if passthrough else request.url.path
            # Before the lookup, so a denied path cannot leak token validity.
            if operation_path not in RUN_TOKEN_ALLOWED_PATHS:
                return _auth_error_response(
                    request, 403, "forbidden", "Run tokens cannot call this route"
                )
            claims = await get_run_token(sha256_hex(presented))
            if claims is None:
                return _unauthorized_response(request, "Invalid or expired run token")
            if isinstance(claims, LegacyRunTokenClaims):
                if claims.minted_with not in accepted_digests:
                    return _unauthorized_response(
                        request, "Invalid or expired run token"
                    )
                request.state.gateway_api_key_name = f"run-token:{claims.minted_by}"
                request.state.gateway_api_key_digest = claims.minted_with
            else:
                authority = await request.app.state.managed_authority.by_key_id(
                    claims.managed_key_id
                )
                if authority is None:
                    return _unauthorized_response(
                        request, "Invalid or expired run token"
                    )
                effective = intersect_permissions(claims, authority.permissions)
                claims = claims.model_copy(update=effective.model_dump())
                request.state.managed_authority = authority
                request.state.gateway_api_key_name = f"run-token:{claims.minted_by}"
                request.state.gateway_api_key_digest = str(claims.managed_key_id)
            if claims.allowed_operations is not None and operation_path not in {
                *RUN_TOKEN_READ_PATHS,
                *(RUN_TOKEN_OPERATION_PATHS[op] for op in claims.allowed_operations),
            }:
                return _auth_error_response(
                    request, 403, "forbidden", "Run token does not allow this operation"
                )
            request.state.run_token_claims = claims
            return await call_next(request)

        if presented.startswith("mgwk_"):
            if request.url.path not in MANAGED_ROOT_ALLOWED_PATHS:
                return _auth_error_response(
                    request, 403, "forbidden", "Managed keys cannot call this route"
                )
            authority = await request.app.state.managed_authority.by_verifier(
                managed_key_verifier(presented)
            )
            if authority is None:
                return _unauthorized_response(request, "Invalid managed key")
            operations = authority.permissions.allowed_operations
            operation = OPERATION_BY_PATH.get(request.url.path)
            if (
                operation is not None
                and operations is not None
                and operation not in operations
            ):
                return _auth_error_response(
                    request,
                    403,
                    "forbidden",
                    "Managed key does not allow this operation",
                )
            request.state.managed_authority = authority
            request.state.gateway_api_key_name = f"managed:{authority.key_id}"
            request.state.gateway_api_key_digest = str(authority.key_id)
            return await call_next(request)

        token_hash = hashlib.sha256(presented.encode()).digest()
        matched_name: str | None = None
        for name, expected_hash in hashed_keys:
            if hmac.compare_digest(token_hash, expected_hash):
                matched_name = name
        if matched_name is None:
            return _unauthorized_response(request, "Invalid API key")

        request.state.gateway_api_key_name = matched_name
        request.state.gateway_api_key_digest = token_hash.hex()
        return await call_next(request)

    return auth_middleware


def _unauthorized_response(request: Request, message: str) -> JSONResponse:
    return _auth_error_response(request, 401, "unauthorized", message)


def _auth_error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    route = telemetry.http_route(request.url.path)
    attrs = {
        "gateway.route": route,
        "gateway.operation": "access_check",
        "gateway.error.code": "access_denied",
        "gateway.error.phase": "access_control",
        "gateway.status_code": status_code,
        "http.request.method": request.method,
        "http.status_code": status_code,
        "http.response.status_code": status_code,
    }
    if route in TRACE_AUTH_FAILURE_PATHS:
        with telemetry.start_span(
            telemetry.http_server_span_name(request.method, route),
            attrs,
            kind="server",
        ):
            telemetry.set_attributes(attrs)
            telemetry.set_status_error("access_denied")
            telemetry.add_event("gateway.auth.error", attrs)
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
    )
