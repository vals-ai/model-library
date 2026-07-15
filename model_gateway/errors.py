"""Map library exceptions to structured HTTP error responses."""

import httpx
from pydantic import BaseModel, ValidationError


class ErrorBody(BaseModel):
    code: str
    message: str
    provider: str | None = None


class ErrorResponse:
    def __init__(self, status_code: int, body: ErrorBody):
        self.status_code = status_code
        self.body = body


_PROVIDERS = ["openai", "anthropic", "google", "mistral", "xai", "bedrock", "cohere"]


def _extract_provider(msg: str) -> str | None:
    lower = msg.lower()
    for p in _PROVIDERS:
        if p in lower:
            return p
    return None


def map_exception_to_error(
    exc: Exception, *, provider: str | None = None
) -> ErrorResponse:
    msg = str(exc)
    provider = _extract_provider(msg) or provider
    lower = msg.lower()

    if "not found in registry" in msg:
        return ErrorResponse(
            400, ErrorBody(code="invalid_model", message=msg, provider=provider)
        )

    if "version mismatch" in lower:
        return ErrorResponse(
            400, ErrorBody(code="history_version_mismatch", message=msg)
        )

    if "custom_api_key" in lower:
        return ErrorResponse(400, ErrorBody(code="custom_key_rejected", message=msg))

    if "content_base64" in lower:
        return ErrorResponse(400, ErrorBody(code="invalid_request", message=msg))

    if isinstance(exc, ValidationError):
        return ErrorResponse(
            400, ErrorBody(code="invalid_request", message=msg, provider=provider)
        )

    if "does not support structured outputs" in lower:
        return ErrorResponse(
            400,
            ErrorBody(
                code="structured_output_unsupported",
                message=msg,
                provider=provider,
            ),
        )

    if (
        "hmac verification failed" in lower
        or "expected hmac-signed pickle" in lower
        or "invalid signed raw history blob" in lower
    ):
        return ErrorResponse(
            400, ErrorBody(code="hmac_verification_failed", message=msg)
        )

    if (
        "context window" in lower
        or "prompt is too long" in lower
        or "maximum context length" in lower
    ):
        return ErrorResponse(
            400,
            ErrorBody(code="context_window_exceeded", message=msg, provider=provider),
        )

    if "rate limit" in lower or "rate_limit" in lower:
        return ErrorResponse(
            429,
            ErrorBody(code="provider_rate_limit", message=msg, provider=provider),
        )

    if "quota" in lower or "insufficient_quota" in lower:
        return ErrorResponse(
            429,
            ErrorBody(code="provider_quota_exceeded", message=msg, provider=provider),
        )

    if "authentication" in lower or "invalid api key" in lower:
        return ErrorResponse(
            502,
            ErrorBody(code="provider_auth_error", message=msg, provider=provider),
        )

    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return ErrorResponse(
            504, ErrorBody(code="timeout", message=msg, provider=provider)
        )

    return ErrorResponse(
        500, ErrorBody(code="internal_error", message=msg, provider=provider)
    )
