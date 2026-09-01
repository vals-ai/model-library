"""
Unit tests for retry logic.
"""

import asyncio
from contextlib import nullcontext
from typing import Callable, Type
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import aiohttp
import httpcore
import httpx
import pytest
from anthropic import InternalServerError as AnthropicInternalServerError
from openai import BadRequestError
from openai import InternalServerError as OpenAIInternalServerError

from model_library.base import LLM, QueryResult
from model_library.base import FinishReason
from model_library.base.output import FinishReasonInfo
from model_library.exceptions import (
    BackoffRetryException,
    BadInputError,
    ContentFilterError,
    GatewayMethodNotSupported,
    ImmediateRetryException,
    ImmediateRetryExhaustedError,
    InvalidStructuredOutputError,
    MaxContextWindowExceededError,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
    NoMatchingToolCallError,
    QueryDeadlineExceededError,
    RateLimitException,
    RetryException,
    ToolCallingNotSupportedError,
    UnexpectedSystemInputError,
    exception_to_provider_error,
    handle_empty_response,
    is_retriable_error,
)
from model_library.retriers.backoff import ExponentialBackoffRetrier
from model_library.retriers.base import BaseRetrier, retry_decorator
from model_library.retriers.utils import jitter


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


def _openai_internal_server_error() -> OpenAIInternalServerError:
    body = {"error": {"message": "Internal Server Error"}}
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(500, request=request, json=body)
    return OpenAIInternalServerError(
        "Error code: 500 - Internal Server Error",
        response=response,
        body=body,
    )


def _anthropic_internal_server_error() -> AnthropicInternalServerError:
    body = {"error": {"message": "Internal Server Error"}}
    request = httpx.Request("POST", "https://provider.example/v1/messages")
    response = httpx.Response(500, request=request, json=body)
    return AnthropicInternalServerError(
        "Error code: 500 - Internal Server Error",
        response=response,
        body=body,
    )


_PROVIDER_HTTP_500_ERRORS: tuple[
    tuple[Callable[[], Exception], type[Exception]], ...
] = (
    (_openai_internal_server_error, OpenAIInternalServerError),
    (_anthropic_internal_server_error, AnthropicInternalServerError),
)


@pytest.mark.parametrize(
    "exc",
    [
        RateLimitException(),
        MaxOutputTokensExceededError(),
        MaxContextWindowExceededError(),
        ContentFilterError(),
        ModelNoOutputError(),
        InvalidStructuredOutputError(),
        ToolCallingNotSupportedError(),
        BadInputError(),
        UnexpectedSystemInputError(),
        GatewayMethodNotSupported(),
        NoMatchingToolCallError(),
        QueryDeadlineExceededError(),
    ],
)
def test_model_library_exceptions_serialize_to_provider_errors(exc: Exception):
    payload = exception_to_provider_error(exc)

    assert payload["message"] == str(exc)
    assert payload["exception_type"] == type(exc).__name__


def test_provider_error_serialization_redacts_raw_context_when_default_exists():
    exc = ContentFilterError(
        "{'finish_reason': 'content_filter', 'response': 'SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK'}"
    )

    payload = exception_to_provider_error(exc)

    assert payload == {
        "message": ContentFilterError.DEFAULT_MESSAGE,
        "exception_type": "ContentFilterError",
    }


def test_provider_error_serialization_unwraps_immediate_retry_exhaustion():
    original = ModelNoOutputError("model returned empty response")
    exc = ImmediateRetryExhaustedError(10, 10, original)

    payload = exception_to_provider_error(exc)

    assert payload == {
        "message": "model returned empty response",
        "exception_type": "ModelNoOutputError",
    }


def test_provider_error_serialization_best_effort_for_unknown_exceptions():
    class ProviderStatusError(Exception):
        code = "rate_limit_exceeded"
        status_code = 429

    payload = exception_to_provider_error(
        ProviderStatusError("provider says slow down")
    )

    assert payload == {
        "message": "provider says slow down",
        "exception_type": "ProviderStatusError",
        "code": "rate_limit_exceeded",
        "status_code": 429,
    }


async def test_query_deadline_cancels_provider_attempt(mock_llm: LLM):
    cancelled = False

    async def blocked_query(*args, **kwargs):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    query_impl = AsyncMock(side_effect=blocked_query)
    mock_llm._query_impl = query_impl  # pyright: ignore[reportPrivateUsage]
    deadline = asyncio.get_running_loop().time() + 0.1

    with pytest.raises(QueryDeadlineExceededError) as exc_info:
        await asyncio.wait_for(mock_llm.query("hi", deadline=deadline), timeout=1)

    assert isinstance(exc_info.value, TimeoutError)
    assert is_retriable_error(exc_info.value) is False
    assert cancelled
    assert query_impl.await_count == 1


async def test_query_deadline_in_past_expires_before_provider_attempt(mock_llm: LLM):
    query_impl = AsyncMock()
    mock_llm._query_impl = query_impl  # pyright: ignore[reportPrivateUsage]
    deadline = asyncio.get_running_loop().time() - 1

    with pytest.raises(QueryDeadlineExceededError):
        await mock_llm.query("hi", deadline=deadline)

    query_impl.assert_not_awaited()


async def test_query_deadline_covers_custom_retrier(mock_llm: LLM):
    cancelled = False

    def custom_retrier(_query_func):
        async def blocked_retrier():
            nonlocal cancelled
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled = True
                raise

        return blocked_retrier

    query_impl = AsyncMock()
    mock_llm._query_impl = query_impl  # pyright: ignore[reportPrivateUsage]
    mock_llm.custom_retrier = custom_retrier
    deadline = asyncio.get_running_loop().time() + 0.1

    with pytest.raises(QueryDeadlineExceededError):
        await asyncio.wait_for(mock_llm.query("hi", deadline=deadline), timeout=1)

    assert cancelled
    query_impl.assert_not_awaited()


async def test_query_deadline_does_not_rewrite_provider_timeout(mock_llm: LLM):
    mock_llm.custom_retrier = lambda query_func: query_func
    mock_llm._query_impl = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=TimeoutError("provider timed out")
    )
    deadline = asyncio.get_running_loop().time() + 60

    with pytest.raises(TimeoutError, match="provider timed out") as exc_info:
        await mock_llm.query("hi", deadline=deadline)

    assert type(exc_info.value) is TimeoutError


@pytest.mark.parametrize("deadline", [float("-inf"), float("inf"), float("nan")])
async def test_query_deadline_rejects_non_finite_values(mock_llm: LLM, deadline: float):
    query_impl = AsyncMock()
    mock_llm._query_impl = query_impl  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match="deadline must be finite"):
        await mock_llm.query("hi", deadline=deadline)

    query_impl.assert_not_awaited()


async def test_jitter():
    """
    Test that jitter function returns values within expected range
    """
    wait = 10.0
    for _ in range(100):
        jittered = jitter(wait)
        # Should be within 20% of original wait time
        assert 8.0 <= jittered <= 12.0


async def test_retry_with_backoff_callback():
    """
    Test retry behavior with custom backoff_callback
    """

    # function raises RetryException, and retries
    # callback raises exception after 2 retries

    def callback(tries: int, exception: Exception | None, elapsed: float, wait: float):
        if tries > 2:
            raise Exception(tries, exception, elapsed, wait)

    callback_mock = Mock(side_effect=callback)

    retrier = ExponentialBackoffRetrier(
        MagicMock(), max_tries=2, retry_callback=callback_mock
    )
    decorator = retry_decorator(retrier)

    mock_func = Mock(side_effect=RetryException())

    @decorator
    async def func():
        mock_func()

    with pytest.raises(Exception):
        await func()

    assert callback_mock.call_count >= 1
    assert mock_func.call_count == 2


async def test_max_retries_giveup():
    """
    Test that after max retries, it gives up and re raises the exception
    """
    error = RetryException()
    mock_func = Mock(side_effect=error)
    logger = MagicMock()

    retrier = ExponentialBackoffRetrier(logger, max_tries=3, initial=0)
    decorator = retry_decorator(retrier)

    @decorator
    async def func():
        mock_func()

    with (
        patch(
            "model_library.retriers.base.telemetry.log_sentry_info"
        ) as log_sentry_info,
        pytest.raises(RetryException) as exc_info,
    ):
        await func()

    assert exc_info.value is error
    assert mock_func.call_count == 3
    assert logger.info.call_count == 2
    logger.warning.assert_not_called()
    logger.error.assert_called_once()
    assert all(
        "Retry Recovered" not in call.args[0]
        for call in logger.info.call_args_list
    )
    assert log_sentry_info.call_count == 2
    assert all(
        "Retry Recovered" not in call.args[0]
        for call in log_sentry_info.call_args_list
    )


@pytest.mark.parametrize(
    ("error_factory", "error_type"),
    _PROVIDER_HTTP_500_ERRORS,
    ids=("openai", "anthropic"),
)
async def test_http_500_retry_budget_allows_success_on_eighth_call(
    error_factory: Callable[[], Exception],
    error_type: type[Exception],
):
    failures = [error_factory() for _ in range(7)]
    mock_func = AsyncMock(side_effect=[*failures, "success"])
    retrier = ExponentialBackoffRetrier(MagicMock())

    result = await retrier.execute(mock_func)

    assert result == "success"
    assert mock_func.await_count == 8
    assert all(isinstance(error, error_type) for error in failures)


@pytest.mark.parametrize(
    ("error_factory", "error_type"),
    _PROVIDER_HTTP_500_ERRORS,
    ids=("openai", "anthropic"),
)
@pytest.mark.parametrize(
    ("max_tries", "expected_calls"),
    [(3, 3), (20, 8)],
    ids=("stricter-limit", "default-limit"),
)
async def test_http_500_retry_budget_gives_up_at_effective_limit(
    error_factory: Callable[[], Exception],
    error_type: type[Exception],
    max_tries: int,
    expected_calls: int,
):
    errors = [error_factory() for _ in range(20)]
    mock_func = AsyncMock(side_effect=errors)
    retrier = ExponentialBackoffRetrier(MagicMock(), max_tries=max_tries)

    with pytest.raises(error_type) as exc_info:
        await retrier.execute(mock_func)

    assert exc_info.value is errors[expected_calls - 1]
    assert mock_func.await_count == expected_calls


async def test_max_time_precedes_http_500_retry_budget():
    errors = [_openai_internal_server_error() for _ in range(8)]
    mock_func = AsyncMock(side_effect=errors)
    logger = MagicMock()
    retrier = ExponentialBackoffRetrier(logger, max_time=5)

    with (
        patch(
            "model_library.retriers.base.time.time",
            side_effect=[100.0, *([100.0] * 7), 106.0],
        ),
        pytest.raises(OpenAIInternalServerError) as exc_info,
    ):
        await retrier.execute(mock_func)

    assert exc_info.value is errors[-1]
    assert mock_func.await_count == 8
    assert "max_time exceeded" in logger.error.call_args.args[0]


async def test_generic_retryable_http_500_uses_http_500_budget():
    errors = [RetryException() for _ in range(20)]
    for error in errors:
        setattr(error, "status_code", 500)
    mock_func = AsyncMock(side_effect=errors)
    retrier = ExponentialBackoffRetrier(MagicMock())

    with pytest.raises(RetryException) as exc_info:
        await retrier.execute(mock_func)

    assert exc_info.value is errors[7]
    assert mock_func.await_count == 8


@pytest.mark.parametrize("status_code", [True, "500", 500.0])
async def test_non_integer_status_code_does_not_consume_http_500_budget(
    status_code: object,
):
    errors = [RetryException() for _ in range(9)]
    for error in errors:
        setattr(error, "status_code", status_code)
    mock_func = AsyncMock(side_effect=errors)
    retrier = ExponentialBackoffRetrier(MagicMock(), max_tries=9)

    with pytest.raises(RetryException) as exc_info:
        await retrier.execute(mock_func)

    assert exc_info.value is errors[-1]
    assert mock_func.await_count == 9


async def test_non_500_retries_keep_default_budget():
    mock_func = AsyncMock(side_effect=RetryException())
    retrier = ExponentialBackoffRetrier(MagicMock())

    with pytest.raises(RetryException):
        await retrier.execute(mock_func)

    assert mock_func.await_count == 20


async def test_non_500_failures_do_not_consume_http_500_budget():
    mock_func = AsyncMock(
        side_effect=[
            RetryException(),
            *[_openai_internal_server_error() for _ in range(7)],
            "success",
        ]
    )
    retrier = ExponentialBackoffRetrier(MagicMock())

    result = await retrier.execute(mock_func)

    assert result == "success"
    assert mock_func.await_count == 9


async def test_retry_success_after_failures():
    """
    Test that after some retryable exceptions, the function succeeds
    """
    mock_func = Mock()
    mock_func.side_effect = [
        RetryException(),
        RetryException(),
        "success",
    ]

    retrier = ExponentialBackoffRetrier(MagicMock(), max_tries=5)
    decorator = retry_decorator(retrier)

    @decorator
    async def failing_func():
        result = mock_func()
        if isinstance(result, Exception):
            raise result
        return result

    result = await failing_func()
    assert result == "success"
    assert mock_func.call_count == 3


@pytest.mark.parametrize(
    "exception,retriable",
    [
        (RetryException, True),
        (ImmediateRetryException, True),
        (BackoffRetryException, True),
        (RateLimitException, True),
        (MaxOutputTokensExceededError, False),
        (MaxContextWindowExceededError, False),
        (ContentFilterError, False),
        (ModelNoOutputError, True),
        (ToolCallingNotSupportedError, False),
        (BadInputError, False),
        (ValueError, False),
        # aiohttp/httpx/httpcore
        (aiohttp.ClientPayloadError, True),
        (httpx.ReadError, True),
        (httpx.ConnectError, True),
        (httpcore.ReadError, True),
        (httpx.RemoteProtocolError, True),
    ],
)
async def test_core_errors(
    mock_llm: LLM,
    exception: Type[Exception],
    retriable: bool,
):
    """
    Test that core errors are / are not retriable
    """

    query_impl_mock = AsyncMock(
        side_effect=[exception("Mock"), QueryResult(output_text="success")]
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    if retriable:
        await mock_llm.query("Mock Input")
        assert query_impl_mock.call_count == 2
    else:
        with pytest.raises(exception):
            await mock_llm.query("Mock Input")
        assert query_impl_mock.call_count == 1


async def test_immediate_retry_logs_attempts_and_recovery_at_info():
    succeeds_after_retries = AsyncMock(
        side_effect=[
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            "success",
        ]
    )
    logger = MagicMock()

    with (
        patch(
            "model_library.retriers.base.time.time",
            side_effect=[100.0, 103.0],
        ),
        patch("model_library.retriers.base.telemetry.set_attributes") as set_attrs,
        patch(
            "model_library.retriers.base.telemetry.log_sentry_info"
        ) as log_sentry_info,
    ):
        result = await BaseRetrier.immediate_retry_wrapper(
            succeeds_after_retries,
            logger,
        )

    assert result == "success"
    set_attrs.assert_called_once_with({"retry.immediate_attempts": 2})
    logger.warning.assert_not_called()
    assert logger.info.call_args_list == [
        call(
            "[Immediate Retry] | 1/10 | Exception "
            "ImmediateRetryException: Immediate retry"
        ),
        call(
            "[Immediate Retry] | 2/10 | Exception "
            "ImmediateRetryException: Immediate retry"
        ),
        call("[Immediate Retry Recovered] | Retries: 2/10 | Elapsed: 3.0s"),
    ]
    assert log_sentry_info.call_args_list == [
        call(
            "[Immediate Retry] | 1/10 | Exception "
            "ImmediateRetryException: Immediate retry",
            {
                "retry.strategy": "immediate",
                "retry.attempt": 1,
                "retry.max_tries": 10,
                "exception.type": "ImmediateRetryException",
            },
        ),
        call(
            "[Immediate Retry] | 2/10 | Exception "
            "ImmediateRetryException: Immediate retry",
            {
                "retry.strategy": "immediate",
                "retry.attempt": 2,
                "retry.max_tries": 10,
                "exception.type": "ImmediateRetryException",
            },
        ),
        call(
            "[Immediate Retry Recovered] | Retries: 2/10 | Elapsed: 3.0s",
            {
                "retry.strategy": "immediate",
                "retry.immediate_attempts": 2,
                "retry.max_tries": 10,
                "retry.elapsed_seconds": 3.0,
            },
        ),
    ]


async def test_immediate_retry_without_failure_has_no_retry_logs():
    logger = MagicMock()

    with patch(
        "model_library.retriers.base.telemetry.log_sentry_info"
    ) as log_sentry_info:
        result = await BaseRetrier.immediate_retry_wrapper(
            AsyncMock(return_value="success"),
            logger,
        )

    assert result == "success"
    logger.info.assert_not_called()
    log_sentry_info.assert_not_called()


async def test_backoff_retry_logs_attempt_and_recovery_at_info():
    succeeds_after_retry = AsyncMock(side_effect=[RetryException("retry"), "success"])
    logger = MagicMock()
    retrier = ExponentialBackoffRetrier(
        logger,
        max_tries=3,
        initial=0,
    )

    with (
        patch(
            "model_library.retriers.base.time.time",
            side_effect=[100.0, 101.0, 103.0],
        ),
        patch("model_library.retriers.base.telemetry.set_attributes") as set_attrs,
        patch(
            "model_library.retriers.base.telemetry.log_sentry_info"
        ) as log_sentry_info,
    ):
        result = await retry_decorator(retrier)(succeeds_after_retry)()

    assert result == "success"
    set_attrs.assert_any_call({"retry.attempts": 1})
    logger.warning.assert_not_called()
    assert logger.info.call_args_list == [
        call(
            "[Retry] | backoff | Attempt: 1 | Elapsed: 1.0s | "
            "Next wait: 0.0s | Exception: RetryException: retry "
        ),
        call("[Retry Recovered] | backoff | Attempts: 1 | Elapsed: 3.0s"),
    ]
    assert log_sentry_info.call_args_list == [
        call(
            "[Retry] | backoff | Attempt: 1 | Elapsed: 1.0s | "
            "Next wait: 0.0s | Exception: RetryException: retry ",
            {
                "retry.strategy": "backoff",
                "retry.attempt": 1,
                "retry.max_tries": 3,
                "retry.elapsed_seconds": 1.0,
                "retry.next_wait_seconds": 0.0,
                "exception.type": "RetryException",
            },
        ),
        call(
            "[Retry Recovered] | backoff | Attempts: 1 | Elapsed: 3.0s",
            {
                "retry.strategy": "backoff",
                "retry.attempts": 1,
                "retry.max_tries": 3,
                "retry.elapsed_seconds": 3.0,
            },
        ),
    ]


async def test_backoff_without_failure_has_no_retry_logs():
    logger = MagicMock()
    retrier = ExponentialBackoffRetrier(logger, max_tries=3)

    with patch(
        "model_library.retriers.base.telemetry.log_sentry_info"
    ) as log_sentry_info:
        result = await retrier.execute(AsyncMock(return_value="success"))

    assert result == "success"
    logger.info.assert_not_called()
    log_sentry_info.assert_not_called()


async def test_backoff_retry_records_attempt_and_sleep_spans():
    succeeds_after_retry = AsyncMock(side_effect=[RetryException("retry"), "success"])
    retrier = ExponentialBackoffRetrier(MagicMock(), max_tries=3)
    spans: list[tuple[str, dict[str, object | None]]] = []
    events: list[tuple[str, dict[str, object | None]]] = []

    def fake_start_span(name: str, attributes: dict[str, object | None]):
        spans.append((name, dict(attributes)))
        return nullcontext()

    def fake_add_event(name: str, attributes: dict[str, object | None]):
        events.append((name, dict(attributes)))

    with (
        patch(
            "model_library.retriers.base.telemetry.start_span",
            side_effect=fake_start_span,
        ),
        patch(
            "model_library.retriers.base.telemetry.add_event",
            side_effect=fake_add_event,
        ),
    ):
        result = await retry_decorator(retrier)(succeeds_after_retry)()

    assert result == "success"
    assert spans[0] == (
        "model_library.retry_attempt",
        {"retry.strategy": "backoff", "retry.attempt": 0},
    )
    assert spans[1][0] == "model_library.retry_sleep"
    assert spans[1][1]["retry.strategy"] == "backoff"
    assert spans[1][1]["retry.attempt"] == 1
    assert spans[1][1]["exception.type"] == "RetryException"
    assert spans[2] == (
        "model_library.retry_attempt",
        {"retry.strategy": "backoff", "retry.attempt": 1},
    )
    assert events[0] == (
        "model_library.retry_attempt_start",
        {"retry.strategy": "backoff", "retry.attempt": 0},
    )
    assert ("model_library.retry_sleep", spans[1][1]) in events
    assert events[-1] == (
        "model_library.retry_attempt_success",
        {"retry.strategy": "backoff", "retry.attempt": 1},
    )


async def test_immediate_retry_exception_success(mock_llm: LLM):
    """
    Test that ImmediateRetryException triggers immediate retries
    """
    # raise ImmediateRetryException twice, then succeed
    query_impl_mock = AsyncMock(
        side_effect=[
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            QueryResult(output_text="success"),
        ]
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    # track calls
    with (
        patch.object(
            BaseRetrier,
            "immediate_retry_wrapper",
            wraps=BaseRetrier.immediate_retry_wrapper,
        ) as mock_immediate_retry,
        patch.object(
            ExponentialBackoffRetrier,
            "execute",
            autospec=True,
            wraps=ExponentialBackoffRetrier.execute,
        ) as mock_backoff_retry,
    ):
        result = await mock_llm.query("Mock Input")

    assert result.output_text == "success"
    # called by immediate_retry_wrapper (2 immediate retries + 1 success)
    assert query_impl_mock.call_count == 3
    # called once by backoff_retry_wrapper
    assert mock_immediate_retry.call_count == 1
    # called once by query()
    assert mock_backoff_retry.call_count == 1


async def test_immediate_retry_exception_limit(mock_llm: LLM):
    """
    Test that ImmediateRetryException reaches limit and the resulting
    ImmediateRetryExhaustedError is NOT retried by the outer backoff retrier.
    """

    query_impl_mock = AsyncMock(
        side_effect=[ImmediateRetryException("Immediate retry")] * 11
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    # track calls
    with (
        patch.object(
            BaseRetrier,
            "immediate_retry_wrapper",
            wraps=BaseRetrier.immediate_retry_wrapper,
        ) as mock_immediate_retry,
        patch.object(
            ExponentialBackoffRetrier,
            "execute",
            autospec=True,
            wraps=ExponentialBackoffRetrier.execute,
        ) as mock_backoff_retry,
    ):
        with pytest.raises(ImmediateRetryExhaustedError):
            await mock_llm.query("Mock Input")

    # only one immediate-retry cycle: 1 initial + 10 retries before giving up
    assert query_impl_mock.call_count == 11
    assert mock_immediate_retry.call_count == 1
    assert mock_backoff_retry.call_count == 1


async def test_immediate_retry_exhausted_is_not_retriable():
    """
    The error raised after immediate retries are exhausted must not be
    retriable, even though its message contains keywords like 'retry' that
    appear in RETRIABLE_EXCEPTION_CODES.
    """
    original = ModelNoOutputError("model produced no output")
    exhausted = ImmediateRetryExhaustedError(10, 10, original)

    assert "retry" in str(exhausted).lower()
    assert is_retriable_error(exhausted) is False


@pytest.mark.parametrize(
    "exception_message,expected_retriable",
    [
        ("Error 429", True),  # rate limit
        ("Error 500", True),  # internal server error
        ("Error 502", True),  # bad gateway
        ("Error 503", True),  # service unavailable
        ("Error 504", True),  # gateway timeout
        ("Error 529", True),  # overloaded, service unavailable
        ("retry failed", True),  # retry keyword
        ("timeout occurred", True),  # timeout keyword
        ("connection_error happened", True),  # connection_error keyword
        ("service_unavailable now", True),  # service_unavailable keyword
        ("rate_limit exceeded", True),  # rate_limit keyword
        ("internal_error detected", True),  # internal_error keyword
        ("server_error occurred", True),  # server_error keyword
        ("Error 404", False),  # not found
        ("Error 400", False),  # bad request
        ("Misc Error", False),  # generic error
        ("overloaded", True),  # overloaded error from anthropic
        (
            "unknown error, 999 (1000)",
            True,
        ),
        (
            "The model is currently at capacity due to high demand.",
            True,
        ),
    ],
)
async def test_retry_by_exception_message(
    exception_message: str,
    expected_retriable: bool,
):
    """
    Test retriable exception messages
    """
    exc = ValueError(exception_message)
    assert is_retriable_error(exc) == expected_retriable


@pytest.mark.parametrize("status_attribute", ["status_code", "status", "code"])
def test_arbitrary_status_attributes_do_not_make_errors_retriable(
    status_attribute: str,
):
    exc = RuntimeError("provider rejected the request")
    setattr(exc, status_attribute, 500)

    assert is_retriable_error(exc) is False


@pytest.mark.parametrize(
    ("error_factory", "error_type"),
    _PROVIDER_HTTP_500_ERRORS,
    ids=("openai", "anthropic"),
)
def test_typed_provider_http_500_is_retriable(
    error_factory: Callable[[], Exception],
    error_type: type[Exception],
):
    error = error_factory()

    assert isinstance(error, error_type)
    assert isinstance(error, (OpenAIInternalServerError, AnthropicInternalServerError))
    assert error.status_code == 500
    assert is_retriable_error(error) is True


async def test_context_window_error_gives_up(mock_llm: LLM):
    """
    Tests against various context window exceeded errors from model providers,
    ensures that we are correctly identifying them and raising with the correct message

    Separate test was created with real api calls to remove the controlled environment aspect
    """

    # List that was generated directly from model providers' error messages
    exception_messages = [
        "The input token count exceeds the maximum number of tokens allowed.",
        "This model's maximum context length is 262144 tokens. However, your request has 1000010 input tokens. Please reduce the length of the input messages. None",
        "Range of input length should be [1, 258048]",
        "Input Tokens Exceeded: Number of input tokens exceeds maximum length. Please update the input to try again.",
        "prompt is too long: 200005 tokens > 200000 maximum",
        "Your input exceeds the context window of this model. Please adjust your input and try again.",
        "This model's maximum context length is 131072 tokens. However, you requested 1008004 tokens (1000004 in the messages, 8000 in the completion). Please reduce the length of the messages or completion.",
        "The prompt is too long: 1000008, model maximum context length: 131071",
        "Prompt contains 1000172 tokens and 0 draft tokens, too large for model with 40960 maximum context length",
        "The total length of all messages is too long.",
        "The input (1000016 tokens) is longer than the model's context length (131072 tokens).",
        "Sent message larger than max (50000056 vs. 20971520)",
        "too many tokens: size limit exceeded by 713295 tokens. Try using shorter or fewer inputs. The limit for this model is 288000 tokens.",
        "input length and max_tokens exceed context limit: 200043 + 32000 > 204658, decrease input length or max_tokens and try again",
        "Invalid request: Your request exceeded model token limit: 262144 (requested: 263162)",  # kimi
        "Payload Too Large",
        "invalid params, context window exceeds limit (2013)",  # minimax
        "Prompt 262280 > 262144 maximum context length",  # mistral
        "Error code: 400 - {'error': {'code': 400, 'message': 'Input length 264373 exceeds the maximum allowed input length of 262112 tokens.', 'type': 'Bad Request'}}",  # poolside
    ]

    for exception_message in exception_messages:
        query_impl_mock = AsyncMock(side_effect=[Exception(exception_message)])

        mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(MaxContextWindowExceededError) as exc_info:
            await mock_llm.query("Mock Input")

        assert exc_info.value.args[0] == exception_message


@pytest.mark.parametrize(
    "finish_reason,expected_exception",
    [
        (FinishReason.CONTEXT_WINDOW_EXCEEDED, MaxContextWindowExceededError),
        (FinishReason.MAX_TOKENS, MaxOutputTokensExceededError),
        (FinishReason.STOP, ModelNoOutputError),
        (FinishReason.UNKNOWN, ModelNoOutputError),
    ],
)
async def test_handle_empty_response(
    finish_reason: FinishReason,
    expected_exception: type[Exception],
):
    """
    Test that handle_empty_response raises the correct exception for each finish reason
    """
    with pytest.raises(expected_exception):
        handle_empty_response(
            FinishReasonInfo(reason=finish_reason, raw=str(finish_reason.value))
        )


async def test_no_retry_exception_not_retried_despite_retriable_message():
    """
    NoRetryException subclasses should never be retried, even if their
    message contains strings that match RETRIABLE_EXCEPTION_CODES (e.g. "500"
    appearing in a base64 blob or raw API response dump).
    """
    # Message contains multiple retriable code substrings
    poison_message = "Error 500 with timeout and retry and server_error"
    assert is_retriable_error(MaxOutputTokensExceededError(poison_message)) is False
    assert is_retriable_error(MaxContextWindowExceededError(poison_message)) is False
    assert is_retriable_error(ContentFilterError(poison_message)) is False


# Exact 400 body returned by the xiaomi/mimo-v2.5 endpoint when it rejects an
# image payload mid-run.
_MIMO_MULTIMODAL_ERROR_BODY = {
    "error": {
        "code": "400",
        "message": "Request Error",
        "param": "Multimodal data is corrupted or cannot be processed.",
        "type": "",
    }
}
# str(BadRequestError) reproduces the SDK's "Error code: <status> - <body>" form.
_MIMO_MULTIMODAL_ERROR_MESSAGE = (
    "Error code: 400 - {'error': {'code': '400', 'message': 'Request Error', "
    "'param': 'Multimodal data is corrupted or cannot be processed.', 'type': ''}}"
)


def _mimo_multimodal_error() -> BadRequestError:
    """Build the exact openai.BadRequestError mimo raises on a bad image payload."""
    request = httpx.Request("POST", "https://mimo.example/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        _MIMO_MULTIMODAL_ERROR_MESSAGE,
        response=response,
        body=_MIMO_MULTIMODAL_ERROR_BODY,
    )


async def test_mimo_multimodal_400_is_retriable():
    """
    The exact 400 ('Multimodal data is corrupted or cannot be processed') that
    mimo returns must be classified as retriable, even though a plain 400 is not.
    """
    err = _mimo_multimodal_error()
    assert err.status_code == 400
    assert "multimodal data is corrupted" in str(err).lower()
    assert is_retriable_error(err) is True


async def test_query_retries_on_mimo_multimodal_400(mock_llm: LLM):
    """
    End-to-end: a query that hits mimo's multimodal 400 once is retried by the
    backoff retrier and succeeds on the next attempt.
    """
    query_impl_mock = AsyncMock(
        side_effect=[_mimo_multimodal_error(), QueryResult(output_text="success")]
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    result = await mock_llm.query("Mock Input")

    assert result.output_text == "success"
    assert query_impl_mock.call_count == 2
