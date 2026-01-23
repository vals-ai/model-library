import logging
from typing import Callable

from model_library.base.base import QueryResult
from model_library.exceptions import exception_message
from model_library.retriers.base import BaseRetrier
from model_library.retriers.utils import jitter

RETRY_MAX_TRIES: int = 20
RETRY_INITIAL: float = 10.0
RETRY_EXPO: float = 1.4
RETRY_MAX_BACKOFF_WAIT: float = 240.0


class ExponentialBackoffRetrier(BaseRetrier):
    """
    Exponential backoff retry strategy.
    Uses exponential backoff with jitter for wait times.
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_tries: int = RETRY_MAX_TRIES,
        max_time: float | None = None,
        retry_callback: Callable[[int, Exception | None, float, float], None]
        | None = None,
        *,
        initial: float = RETRY_INITIAL,
        expo: float = RETRY_EXPO,
        max_backoff_wait: float = RETRY_MAX_BACKOFF_WAIT,
    ):
        super().__init__(
            strategy="backoff",
            logger=logger,
            max_tries=max_tries,
            max_time=max_time,
            retry_callback=retry_callback,
        )

        self.initial = initial
        self.expo = expo
        self.max_backoff_wait = max_backoff_wait

    async def _calculate_wait_time(
        self, attempt: int, exception: Exception | None = None
    ) -> float:
        """Calculate exponential backoff wait time with jitter"""

        exponential_wait = self.initial * (self.expo**attempt)
        capped_wait = min(exponential_wait, self.max_backoff_wait)
        return jitter(capped_wait)

    async def _on_retry(
        self, exception: Exception | None, elapsed: float, wait_time: float
    ) -> None:
        """Increment attempt counter and log retry attempt"""

        logger_msg = f"[Retry] | {self.strategy} | Attempt: {self.attempts} | Elapsed: {elapsed:.1f}s | Next wait: {wait_time:.1f}s | Exception: {exception_message(exception)} "

        self.logger.warning(logger_msg)

        if self.retry_callback:
            self.retry_callback(self.attempts, exception, elapsed, wait_time)

    async def _pre_function(self) -> None:
        return

    async def _post_function(self, result: tuple[QueryResult, float]) -> None:
        return

    async def validate(self) -> None:
        return
