"""Bearer token auth middleware."""

import hashlib
import hmac
from collections.abc import Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from model_gateway.run_token_types import RUN_TOKEN_OPERATION_PATHS
from model_gateway.run_tokens import RUN_TOKEN_PREFIX, get_run_token, sha256_hex

EXEMPT_PATHS = {"/health/live", "/health/ready"}
TRACE_AUTH_FAILURE_PATHS = telemetry.HTTP_TRACE_ALLOWED_ROUTES

RUN_TOKEN_ALLOWED_PATHS = frozenset({*RUN_TOKEN_OPERATION_PATHS.values(), "/registry"})


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
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized_response(
                request,
                "Missing or malformed Authorization header",
            )

        presented = auth_header[7:]
        request.state.run_token_claims = None

        if presented.startswith(RUN_TOKEN_PREFIX):
            # Before the lookup, so a denied path cannot leak token validity.
            if request.url.path not in RUN_TOKEN_ALLOWED_PATHS:
                return _auth_error_response(
                    request, 403, "forbidden", "Run tokens cannot call this route"
                )
            claims = await get_run_token(sha256_hex(presented))
            # Dev and prod share Redis but not keys.
            if claims is None or claims.minted_with not in accepted_digests:
                return _unauthorized_response(request, "Invalid or expired run token")
            if claims.allowed_operations is not None and request.url.path not in {
                "/registry",
                *(RUN_TOKEN_OPERATION_PATHS[op] for op in claims.allowed_operations),
            }:
                return _auth_error_response(
                    request, 403, "forbidden", "Run token does not allow this operation"
                )
            request.state.gateway_api_key_name = f"run-token:{claims.minted_by}"
            request.state.gateway_api_key_digest = claims.minted_with
            request.state.run_token_claims = claims
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
    attrs = {
        "gateway.route": request.url.path,
        "gateway.operation": "access_check",
        "gateway.error.code": "access_denied",
        "gateway.error.phase": "access_control",
        "gateway.status_code": status_code,
        "http.request.method": request.method,
        "http.status_code": status_code,
        "http.response.status_code": status_code,
    }
    if request.url.path in TRACE_AUTH_FAILURE_PATHS:
        with telemetry.start_span(
            telemetry.http_server_span_name(request.method, request.url.path),
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
