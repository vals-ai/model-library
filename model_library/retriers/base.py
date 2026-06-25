import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Literal, TypeVar

import model_library.telemetry as telemetry
from model_library.base.base import QueryResult
from model_library.exceptions import (
    ImmediateRetryException,
    ImmediateRetryExhaustedError,
    MaxContextWindowExceededError,
    exception_message,
    is_context_window_error,
    is_retriable_error,
)

RetrierType = Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]

R = TypeVar("R")  # wrapper return type


class BaseRetrier(ABC):
    """
    Base class for retry strategies.
    Implements core retry logic and error handling.
    Subclasses should implement strategy-specific wait time calculations.
    """

    # NOTE: for token retrier, the estimate_tokens stays the same because ImmediateRetryException
    # is raised for network errors, where tokens have not been deducted yet

    @staticmethod
    async def immediate_retry_wrapper(
        func: Callable[[], Awaitable[R]],
        logger: logging.Logger,
    ) -> R:
        """
        Retry the query immediately
        """
        MAX_IMMEDIATE_RETRIES = 10
        retries = 0
        while True:
            try:
                result = await func()
                telemetry.set_attributes({"retry.immediate_attempts": retries})
                return result
            except ImmediateRetryException as e:
                if retries >= MAX_IMMEDIATE_RETRIES:
                    telemetry.set_attributes({"retry.immediate_attempts": retries})
                    raise ImmediateRetryExhaustedError(
                        retries, MAX_IMMEDIATE_RETRIES, e
                    ) from e
                retries += 1
                telemetry.add_event(
                    "model_library.immediate_retry",
                    {
                        "retry.attempt": retries,
                        "retry.max_tries": MAX_IMMEDIATE_RETRIES,
                        "exception.type": type(e).__name__,
                    },
                )

                logger.warning(
                    f"[Immediate Retry] | {retries}/{MAX_IMMEDIATE_RETRIES} | Exception {exception_message(e)}"
                )

    def __init__(
        self,
        strategy: Literal["backoff", "token"],
        logger: logging.Logger,
        max_tries: int | None,
        max_time: float | None,
        retry_callback: Callable[[int, Exception | None, float, float], None] | None,
    ):
        self.strategy = strategy
        self.logger = logger
        self.max_tries = max_tries
        self.max_time = max_time
        self.retry_callback = retry_callback

        self.attempts = 0
        self.start_time: float | None = None

    @abstractmethod
    async def _calculate_wait_time(
        self, attempt: int, exception: Exception | None = None
    ) -> float:
        """
        Calculate wait time before retrying

        Args:
            attempt: Current attempt number (0-indexed)
            exception: The exception that triggered the retry

        Returns:
            Wait time in seconds
        """
        ...

    @abstractmethod
    async def _on_retry(
        self, exception: Exception | None, elapsed: float, wait_time: float
    ) -> None:
        """
        Hook called before waiting on retry

        Args:
            exception: The exception that triggered the retry
            elapsed: Time elapsed since start
            wait_time: Wait time
        """
        ...

    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception to evaluate

        Returns:
            True if should retry, False otherwise
        """

        if is_context_window_error(exception):
            return False
        return is_retriable_error(exception)

    def _handle_giveup(self, exception: Exception, reason: str) -> None:
        """
        Handle final exception after all retries exhausted.

        Args:
            exception: The final exception
            reason: Reason for giving up

        Raises:
            MaxContextWindowExceededError: If context window error
            Exception: The original exception otherwise
        """

        telemetry.set_attributes({"retry.attempts": self.attempts})
        telemetry.add_event(
            "model_library.retry_giveup",
            {
                "retry.strategy": self.strategy,
                "retry.reason": reason,
                "retry.attempt": self.attempts,
                "retry.max_tries": self.max_tries,
                "exception.type": type(exception).__name__,
            },
        )
        telemetry.record_exception(
            exception,
            {"retry.strategy": self.strategy, "retry.reason": reason},
        )

        self.logger.error(
            f"[Give up] | {self.strategy} | {reason} | Exception: {exception_message(exception)}"
        )

        # instead of raising the provider exception, raise the custom MaxContextWindowExceededError
        if is_context_window_error(exception):
            message = exception.args[0] if exception.args else str(exception)
            raise MaxContextWindowExceededError(message)

        raise exception

    @abstractmethod
    async def _pre_function(self) -> None:
        """Hook called before the actual function call"""
        ...

    @abstractmethod
    async def _post_function(self, result: tuple[QueryResult, float]) -> None:
        """Hook called after the actual function call"""
        ...

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: If retries exhausted or non-retriable error occurs
        """

        self.attempts = 0
        self.start_time = time.time()

        await self.validate()
        telemetry.set_attributes(
            {
                "retry.strategy": self.strategy,
                "retry.max_tries": self.max_tries,
                "retry.max_time_seconds": self.max_time,
            }
        )

        while True:
            try:
                attempt_attributes = {
                    "retry.strategy": self.strategy,
                    "retry.attempt": self.attempts,
                }
                with telemetry.start_span(
                    "model_library.retry_attempt",
                    attempt_attributes,
                ):
                    telemetry.add_event(
                        "model_library.retry_attempt_start", attempt_attributes
                    )
                    await self._pre_function()
                    result = await func(*args, **kwargs)
                    await self._post_function(result)
                    telemetry.set_attributes({"retry.attempts": self.attempts})
                    telemetry.add_event(
                        "model_library.retry_attempt_success", attempt_attributes
                    )
                    return result

            except Exception as e:
                elapsed = time.time() - self.start_time

                self.attempts += 1

                # check if max_tries exceeded
                if self.max_tries is not None and self.attempts >= self.max_tries:
                    self._handle_giveup(
                        e, f"max_tries exceeded ({self.attempts} >= {self.max_tries})"
                    )

                # check if max_time exceeded
                if self.max_time is not None and elapsed > self.max_time:
                    self._handle_giveup(
                        e, f"max_time exceeded ({elapsed} > {self.max_time}s)"
                    )

                if not self._should_retry(e):
                    self._handle_giveup(e, "not retriable")

                # calculate wait time
                wait_time = await self._calculate_wait_time(self.attempts, e)
                # call pre retry sleep hook
                await self._on_retry(e, elapsed, wait_time)
                sleep_attributes = {
                    "retry.strategy": self.strategy,
                    "retry.attempt": self.attempts,
                    "retry.elapsed_seconds": elapsed,
                    "retry.next_wait_seconds": wait_time,
                    "exception.type": type(e).__name__,
                }
                with telemetry.start_span(
                    "model_library.retry_sleep",
                    sleep_attributes,
                ):
                    telemetry.add_event("model_library.retry_sleep", sleep_attributes)
                    await asyncio.sleep(wait_time)

    async def validate(self) -> None:
        """Validate the retrier"""
        ...


def retry_decorator(retrier: BaseRetrier) -> RetrierType:
    """Create a retry decorator from an initialized retrier"""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retrier.execute(func, *args, **kwargs)

        return wrapper

    return decorator
