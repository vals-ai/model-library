"""Bearer token auth middleware."""

import hashlib
import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

EXEMPT_PATHS = {"/health/live", "/health/ready"}
TRACE_AUTH_FAILURE_PATHS = {
    "/embeddings",
    "/files/upload",
    "/models",
    "/models/resolve",
    "/moderation",
    "/query",
    "/rate-limit",
    "/registry",
    "/token-retry/status",
    "/tokens/count",
}


def create_auth_middleware(valid_keys: set[str]):
    # Pre-hash keys for constant-time comparison (prevents timing side-channel)
    hashed_keys = {hashlib.sha256(k.encode()).digest() for k in valid_keys}

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

        token_hash = hashlib.sha256(auth_header[7:].encode()).digest()
        if not any(hmac.compare_digest(token_hash, h) for h in hashed_keys):
            return _unauthorized_response(request, "Invalid API key")

        request.state.gateway_api_key_fingerprint = token_hash.hex()[:16]
        return await call_next(request)

    return auth_middleware


def _unauthorized_response(request: Request, message: str) -> JSONResponse:
    attrs = {
        "gateway.route": request.url.path,
        "gateway.operation": "access_check",
        "gateway.error.code": "access_denied",
        "gateway.error.phase": "access_control",
        "gateway.status_code": 401,
        "http.request.method": request.method,
        "http.status_code": 401,
        "http.response.status_code": 401,
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
        status_code=401,
        content={"code": "unauthorized", "message": message},
    )
