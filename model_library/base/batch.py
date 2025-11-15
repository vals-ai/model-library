from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import BaseModel

from model_library.base.input import InputItem
from model_library.base.output import QueryResult


class BatchResult(BaseModel):
    custom_id: str
    output: QueryResult
    error_message: str | None = None


class LLMBatchMixin(ABC):
    @abstractmethod
    async def create_batch_query_request(
        self,
        custom_id: str,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> dict[str, Any]:
        """Return a single query request

        The batch api sends out a batch of query requests to various endpoints.

        For example OpenAI sends can send requests to /v1/responses or /v1/chat/completions endpoints.

        This method creates a query request for methods such methods
        """
        ...

    @abstractmethod
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        """
        Batch query the model
        Returns:
            str: batch_id
        Raises:
            Exception: If failed to batch query
        """
        ...

    @abstractmethod
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """
        Returns results for batch
        Raises:
            Exception: If failed to get results
        """
        ...

    @abstractmethod
    async def get_batch_progress(self, batch_id: str) -> int:
        """
        Returns number of completed requests for batch
        Raises:
            Exception: If failed to get progress
        """
        ...

    @abstractmethod
    async def cancel_batch_request(self, batch_id: str) -> None:
        """
        Cancels batch
        Raises:
            Exception: If failed to cancel
        """
        ...

    @abstractmethod
    async def get_batch_status(
        self,
        batch_id: str,
    ) -> str:
        """
        Returns batch status
        Raises:
            Exception: If failed to get status
        """
        ...

    @classmethod
    @abstractmethod
    def is_batch_status_completed(
        cls,
        batch_status: str,
    ) -> bool:
        """
        Returns if batch status is completed

        A completed state is any state that is final and not in-progress
        Example: failed | cancelled | expired | completed

        An incompleted state is any state that is not completed
        Example: in_progress | pending | running
        """
        ...

    @classmethod
    @abstractmethod
    def is_batch_status_failed(
        cls,
        batch_status: str,
    ) -> bool:
        """Returns if batch status is failed"""
        ...

    @classmethod
    @abstractmethod
    def is_batch_status_cancelled(
        cls,
        batch_status: str,
    ) -> bool:
        """Returns if batch status is cancelled"""
        ...
