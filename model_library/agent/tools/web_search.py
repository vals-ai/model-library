from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import wraps
from typing import Any, Callable, Coroutine

import httpx

from model_library import model_library_settings
from model_library.agent.tool import Tool, ToolOutput

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 30.0
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def retry_on_429(
    func: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Retry on 429 rate limit errors with exponential backoff"""

    MAX_RETRIES = 5
    RETRY_BASE_WAIT = 2

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if attempt == MAX_RETRIES or e.response.status_code != 429:
                    raise
                await asyncio.sleep(RETRY_BASE_WAIT**attempt)

    return wrapper


class TavilyWebSearch(Tool):
    def __init__(self, max_end_date: str, tavily_api_key: str | None = None):
        super().__init__(
            name="web_search",
            description=(
                "Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page."
            ),
            parameters={
                "search_query": {
                    "type": "string",
                    "description": "The query to search for",
                },
                "start_date": {
                    "type": "string",
                    "description": "(optional) The start date for the search range in the format YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "description": f"(optional) The end date for the search range in the format YYYY-MM-DD. If the value is later than {max_end_date}, it will be set to {max_end_date}.",
                },
                "number_of_results": {
                    "type": "integer",
                    "description": "(optional) The number of search results to return.",
                    "maximum": 20,
                    "minimum": 1,
                    "default": 10,
                },
            },
            required=["search_query"],
        )
        self._client = httpx.AsyncClient(timeout=TAVILY_TIMEOUT)
        self._api_key = tavily_api_key or model_library_settings.TAVILY_API_KEY
        self.max_end_date = max_end_date

    @retry_on_429
    async def _execute_search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        if end_date is None:
            end_date = self.max_end_date

        if not DATE_REGEX.match(end_date):
            raise ValueError(
                f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD."
            )
        if end_date > self.max_end_date:
            end_date = self.max_end_date

        body: dict[str, Any] = {
            "api_key": self._api_key,
            "query": search_query,
            "search_depth": "fast",
            "end_date": end_date,
            "max_results": number_of_results,
            "chunks_per_source": 1,
        }

        if start_date:
            if not DATE_REGEX.match(start_date):
                raise ValueError(
                    f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD."
                )
            if start_date > end_date:
                raise ValueError(
                    f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
                )
            body["start_date"] = start_date

        response = await self._client.post(TAVILY_SEARCH_URL, json=body)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {"url": r.get("url"), "title": r.get("title"), "content": r.get("content")}
            for r in results
        ]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            results = await self._execute_search(**args)
            return ToolOutput(output=json.dumps(results, default=str))
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Web search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
