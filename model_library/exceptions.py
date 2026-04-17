from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from model_library.base.output import FinishReasonInfo

from ai21 import TooManyRequestsError as AI21RateLimitError
from anthropic import InternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from httpcore import ReadError as HTTPCoreReadError
from httpx import ConnectError as HTTPXConnectError
from httpx import ReadError as HTTPXReadError
from httpx import RemoteProtocolError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import InternalServerError as OpenAIInternalServerError
from openai import RateLimitError as OpenAIRateLimitError
from openai import UnprocessableEntityError as OpenAIUnprocessableEntityError


# Base class for a retryable exception
class RetryException(Exception): ...


# Exception to be retried immediately
class ImmediateRetryException(RetryException): ...


# Exception to be retried with backoff
class BackoffRetryException(RetryException): ...


# Base class for a non-retryable exception
class NoRetryException(Exception): ...


class RateLimitException(BackoffRetryException):
    """Raised when the rate limit is exceeded"""

    DEFAULT_MESSAGE: str = "Rate limit exceeded."

    def __init__(self, message: str | None = None):
        super().__init__(message or RateLimitException.DEFAULT_MESSAGE)


class MaxOutputTokensExceededError(NoRetryException):
    """
    Raised when the output exceeds the allowed max output tokens limit
    AND the output (including reasoning) was empty
    """

    DEFAULT_MESSAGE: str = (
        "Output exceeded max tokens limit and model produced no useful content."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or MaxOutputTokensExceededError.DEFAULT_MESSAGE)


class MaxContextWindowExceededError(NoRetryException):
    """
    Raised when the context window exceeds the allowed max context window limit
    """

    DEFAULT_MESSAGE: str = (
        "Context window exceeded the maximum allowed context window. "
        "Consider reducing the context window size."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or MaxContextWindowExceededError.DEFAULT_MESSAGE)


class ContentFilterError(NoRetryException):
    """
    Raised when the model's content filter is triggered
    """

    DEFAULT_MESSAGE: str = "Model's content filter triggered"

    def __init__(self, message: str | None = None):
        super().__init__(message or ContentFilterError.DEFAULT_MESSAGE)


# List of regex patterns to match context window exceeded errors
# This was made after forcing context window errors on a model from each provider
CONTEXT_WINDOW_PATTERN = re.compile(
    r"maximum context length is \d+ tokens|"
    r"context length is \d+ tokens|"
    r"exceed.* context (limit|window|length)|"
    r"input token count exceeds the maximum number of tokens allowed|"
    r"context window exceeds|"
    r"exceeds maximum length|"
    r"too long.*tokens.*maximum|"
    r"too large for model with \d+ maximum context length|"
    r"longer than the model's context length|"
    r"too many tokens.*size limit exceeded|"
    r"prompt is too long|"
    r"maximum prompt length|"
    r"input length should be|"
    r"sent message larger than max|"
    r"input tokens exceeded|"
    r"(messages?|total length).*too long|"
    r"payload.*too large|"
    r"string too long|"
    r"input exceeded the context window"
)


def is_context_window_error(e: Exception) -> bool:
    return CONTEXT_WINDOW_PATTERN.search(str(e).lower()) is not None


class ModelNoOutputError(ImmediateRetryException):
    """
    Raised when the model fails to produce any output or response.
    """

    DEFAULT_MESSAGE: str = (
        "Model failed to produce any output. "
        "This may indicate an issue with the model or input."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or ModelNoOutputError.DEFAULT_MESSAGE)


def handle_empty_response(
    finish_reason: FinishReasonInfo,
    context: dict[str, Any] | None = None,
) -> NoReturn:
    """Raise the appropriate error when a model produces no output."""
    from model_library.base.output import FinishReason as _FinishReason

    log_message = str({"finish_reason": finish_reason.raw, **(context or {})})
    if finish_reason.reason == _FinishReason.CONTEXT_WINDOW_EXCEEDED:
        raise MaxContextWindowExceededError(log_message)
    if finish_reason.reason == _FinishReason.MAX_TOKENS:
        raise MaxOutputTokensExceededError(log_message)
    if (
        finish_reason.reason == _FinishReason.CONTENT_FILTER
        or finish_reason.reason == _FinishReason.GUARDRAIL
    ):
        raise ContentFilterError(log_message)

    raise ModelNoOutputError(log_message)


class InvalidStructuredOutputError(ImmediateRetryException):
    """
    Raised when the model produces an invalid structured output.
    We assume this is a transient model error and retry
    """

    DEFAULT_MESSAGE: str = "Model produced invalid structured output"

    def __init__(self, message: str | None = None):
        super().__init__(message or InvalidStructuredOutputError.DEFAULT_MESSAGE)


class ToolCallingNotSupportedError(Exception):
    """
    Raised when the model does not support tool calling
    """

    DEFAULT_MESSAGE: str = (
        "Tool calling is not supported by this model. "
        "Consider using a model that supports tools."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or ToolCallingNotSupportedError.DEFAULT_MESSAGE)


class BadInputError(Exception):
    """
    Raised when the input is not supported by the model.
    """

    DEFAULT_MESSAGE: str = "Invalid input provided."

    def __init__(self, message: str | None = None):
        super().__init__(message or BadInputError.DEFAULT_MESSAGE)


class UnexpectedSystemInputError(Exception):
    """
    Raised when a SystemInput is encountered inside parse_input.
    SystemInput must be at position 0 and is consumed by build_body before parse_input is called.
    """

    ...


class NoMatchingToolCallError(Exception):
    """
    Raised when a tool call result is provided with no matching tool call
    """

    DEFAULT_MESSAGE: str = "Tool call result provided with no matching tool call"

    def __init__(self, message: str | None = None):
        super().__init__(message or NoMatchingToolCallError.DEFAULT_MESSAGE)


# Add more retriable exceptions as needed
# Providers that don't have an explicit rate limit error are handled manually
# by wrapping errored Http/gRPC requests with a BackoffRetryException
RETRIABLE_EXCEPTIONS = [
    OpenAIRateLimitError,
    OpenAIInternalServerError,
    OpenAIAPITimeoutError,
    OpenAIUnprocessableEntityError,
    OpenAIAPIConnectionError,
    AnthropicRateLimitError,
    InternalServerError,
    AI21RateLimitError,
    RemoteProtocolError,  # httpx connection closing when running models from sdk
    HTTPXReadError,
    HTTPXConnectError,
    HTTPCoreReadError,
]

# Some providers do not have typed exceptions, so we need to manually check for
# api status codes.
RETRIABLE_EXCEPTION_CODES = [
    "429",  # rate limit / too many requests
    "500",  # internal server error
    "502",  # bad gateway
    "503",  # service unavailable
    "504",  # gateway timeout
    "529",  # overloaded, service unavailable
    "retry",
    "timeout",
    "connection_error",
    "service_unavailable",
    "rate_limit",
    "rate limit",
    "internal_error",
    "server_error",
    "overloaded",
    "throttling",  # AWS throttling errors
    "internal server error",
    "InternalServerError",
    "statuscode.internal",  # gRPC INTERNAL errors (e.g. xAI native SDK)
    "statuscode.unavailable",  # gRPC UNAVAILABLE errors (e.g. broken pipe)
    "statuscode.invalid_argument",  # gRPC INVALID_ARGUMENT (transient under concurrency)
    "statuscode.unknown",  # gRPC UNKNOWN errors (e.g. "Stream removed")
    "stream removed",  # gRPC stream dropped by peer
    "rst_stream",  # gRPC RST_STREAM errors
]


def is_retriable_error(e: Exception) -> bool:
    if isinstance(e, NoRetryException):
        return False

    if isinstance(e, RetryException):
        return True

    if is_context_window_error(e):
        return False

    if any(isinstance(e, exception) for exception in RETRIABLE_EXCEPTIONS):
        return True

    # check error message
    error_message = str(e).lower()
    return any(code in error_message for code in RETRIABLE_EXCEPTION_CODES)


def exception_message(exception: Exception | Any) -> str:
    if not isinstance(exception, Exception):
        return str(exception)
    return (
        f"{type(exception).__name__}: {str(exception)}"
        if str(exception)
        else type(exception).__name__
    )
