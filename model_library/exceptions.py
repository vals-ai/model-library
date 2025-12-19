import logging
import random
import re
from typing import Any, Callable

import backoff
from ai21 import TooManyRequestsError as AI21RateLimitError
from anthropic import InternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from backoff._typing import Details
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


class RateLimitException(BackoffRetryException):
    """Raised when the rate limit is exceeded"""

    DEFAULT_MESSAGE: str = "Rate limit exceeded."

    def __init__(self, message: str | None = None):
        super().__init__(message or RateLimitException.DEFAULT_MESSAGE)


class MaxOutputTokensExceededError(Exception):
    """
    Raised when the output exceeds the allowed max output tokens limit
    AND the output (including reasoning) was empty
    """

    DEFAULT_MESSAGE: str = (
        "Output exceeded your 'Max Output Tokens' limit. "
        "Consider increasing the limit in 'Run Suite' > 'Model Parameters'."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or MaxOutputTokensExceededError.DEFAULT_MESSAGE)


class MaxContextWindowExceededError(Exception):
    """
    Raised when the context window exceeds the allowed max context window limit
    """

    DEFAULT_MESSAGE: str = (
        "Context window exceeded the maximum allowed context window. "
        "Consider reducing the context window size."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or MaxContextWindowExceededError.DEFAULT_MESSAGE)


# List of regex patterns to match context window exceeded errors
# This was made after forcing context window errors on a model from each provider
CONTEXT_WINDOW_PATTERN = re.compile(
    r"maximum context length is \d+ tokens|"
    r"context length is \d+ tokens|"
    r"exceed.* context (limit|window|length)|"
    r"exceeds maximum length|"
    r"too long.*tokens.*maximum|"
    r"too large for model with \d+ maximum context length|"
    r"longer than the model's context length|"
    r"too many tokens.*size limit exceeded|"
    r"prompt is too long|"
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
]


def is_retriable_error(e: Exception) -> bool:
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


RETRY_MAX_TRIES: int = 20
RETRY_INITIAL: float = 10.0
RETRY_EXPO: float = 1.4
RETRY_MAX_BACKOFF_WAIT: float = 240.0  # 4 minutes (more with jitter)


def jitter(wait: float) -> float:
    """
    Increase or decrease the wait time by up to 20%.
    """
    jitter_fraction = 0.2
    min_wait = wait * (1 - jitter_fraction)
    max_wait = wait * (1 + jitter_fraction)
    return random.uniform(min_wait, max_wait)


def retry_llm_call(
    logger: logging.Logger,
    max_tries: int = RETRY_MAX_TRIES,
    max_time: float | None = None,
    backoff_callback: (
        Callable[[int, Exception | None, float, float], None] | None
    ) = None,
):
    def on_backoff(details: Details):
        exception = details.get("exception")
        tries = details.get("tries", 0)
        elapsed = details.get("elapsed", 0.0)
        wait = details.get("wait", 0.0)

        logger.warning(
            f"[Retrying] Exception: {exception_message(exception)} | Attempt: {tries} | "
            + f"Elapsed: {elapsed:.1f}s | Next wait: {wait:.1f}s"
        )

        if backoff_callback:
            backoff_callback(tries, exception, elapsed, wait)

    def giveup(e: Exception) -> bool:
        return not is_retriable_error(e)

    def on_giveup(details: Details) -> None:
        exception: Exception | None = details.get("exception", None)
        if not exception:
            return

        logger.error(
            f"Giving up after retries. Final exception: {exception_message(exception)}"
        )

        if is_context_window_error(exception):
            message = exception.args[0] if exception.args else str(exception)
            raise MaxContextWindowExceededError(message)

        raise exception

    return backoff.on_exception(
        wait_gen=lambda: backoff.expo(
            base=RETRY_EXPO,
            factor=RETRY_INITIAL,
            max_value=RETRY_MAX_BACKOFF_WAIT,
        ),
        exception=Exception,
        max_tries=max_tries,
        max_time=max_time,
        giveup=giveup,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        jitter=jitter,
    )
