"""
Unit tests for retry logic.
"""

from typing import Type
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from model_library.base import LLM, QueryResult
from model_library.exceptions import (
    BackoffRetryException,
    BadInputError,
    ImmediateRetryException,
    MaxContextWindowExceededError,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
    RateLimitException,
    RetryException,
    ToolCallingNotSupportedError,
    is_retriable_error,
    jitter,
    retry_llm_call,
)


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep") as mock_sleep:
        yield mock_sleep


def test_jitter():
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
    mock_func = Mock(side_effect=RetryException())

    @retry_llm_call(MagicMock(), max_tries=2, backoff_callback=callback_mock)
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
    mock_func = Mock(side_effect=RetryException())

    @retry_llm_call(MagicMock(), max_tries=3)
    async def func():
        mock_func()

    with pytest.raises(RetryException):
        await func()

    assert mock_func.call_count == 3


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

    @retry_llm_call(MagicMock(), max_tries=5)
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
        (ModelNoOutputError, True),
        (ToolCallingNotSupportedError, False),
        (BadInputError, False),
        (ValueError, False),
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
        side_effect=[exception(), QueryResult(output_text="success")]
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    if retriable:
        await mock_llm.query("Mock Input")
        assert query_impl_mock.call_count == 2
    else:
        with pytest.raises(exception):
            await mock_llm.query("Mock Input")
        assert query_impl_mock.call_count == 1


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
            LLM, "immediate_retry_wrapper", wraps=LLM.immediate_retry_wrapper
        ) as mock_immediate_retry,
        patch.object(
            LLM, "backoff_retry_wrapper", wraps=LLM.backoff_retry_wrapper
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
    Test that ImmediateRetryException reaches limit
    """

    # raise ImmediateRetryException twice, then succeed
    query_impl_mock = AsyncMock(
        side_effect=[
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
            ImmediateRetryException("Immediate retry"),
        ]
    )
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    # track calls
    with (
        patch.object(
            LLM, "immediate_retry_wrapper", wraps=LLM.immediate_retry_wrapper
        ) as mock_immediate_retry,
        patch.object(
            LLM, "backoff_retry_wrapper", wraps=LLM.backoff_retry_wrapper
        ) as mock_backoff_retry,
    ):
        with pytest.raises(Exception):
            await mock_llm.query("Mock Input")

    assert query_impl_mock.call_count == 12
    assert mock_immediate_retry.call_count == 2
    assert mock_backoff_retry.call_count == 1


@pytest.mark.parametrize(
    "exception_class",
    [
        pytest.param("httpx.ReadError", id="httpx_read_error"),
        pytest.param("httpx.ConnectError", id="httpx_connect_error"),
        pytest.param("httpcore.ReadError", id="httpcore_read_error"),
        pytest.param("httpx.RemoteProtocolError", id="httpx_remote_protocol_error"),
    ],
)
def test_httpx_network_errors_are_retriable(exception_class: str):
    """
    Test that httpx/httpcore network errors are retriable
    """
    import httpcore
    import httpx

    exception_map = {
        "httpx.ReadError": httpx.ReadError,
        "httpx.ConnectError": httpx.ConnectError,
        "httpcore.ReadError": httpcore.ReadError,
        "httpx.RemoteProtocolError": httpx.RemoteProtocolError,
    }

    exc_class = exception_map[exception_class]
    exc = exc_class("Network error")
    assert is_retriable_error(exc) is True


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
    ],
)
def test_retry_by_exception_message(
    exception_message: str,
    expected_retriable: bool,
):
    """
    Test retriable exception messages
    """
    exc = ValueError(exception_message)
    assert is_retriable_error(exc) == expected_retriable


async def test_context_window_error_gives_up(mock_llm: LLM):
    """
    Tests against various context window exceeded errors from model providers,
    ensures that we are correctly identifying them and raising with the correct message

    Separate test was created with real api calls to remove the controlled environment aspect
    """

    # List that was generated directly from model providers' error messages
    exception_messages = [
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
        "Payload Too Large",
    ]

    for exception_message in exception_messages:
        query_impl_mock = AsyncMock(side_effect=[Exception(exception_message)])

        mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(MaxContextWindowExceededError) as exc_info:
            await mock_llm.query("Mock Input")

        assert exc_info.value.args[0] == exception_message
