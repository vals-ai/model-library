from __future__ import annotations

import io
import json
import random
import re
import time
from typing import Any, Literal, Sequence, cast

import redis
from redis.client import Redis
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    BatchResult,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMBatchMixin,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
    TextInput,
    ToolDefinition,
)
from model_library.register_models import register_provider
from model_library.utils import truncate_str

FAIL_RATE = 0.1
BATCH_EXP = 60 * 10  # 10 minutes


class DummyAIBatchMixin(LLMBatchMixin):
    def __init__(self, openai: DummyAIModel):
        self._root: DummyAIModel = openai
        self._client: Redis = self._root.get_client()

    @override
    async def create_batch_query_request(
        self,
        custom_id: str,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "",
            "url": "",
            "body": await self._root.create_body(input, tools=[], **kwargs),
        }

    @override
    async def batch_query(self, batch_name: str, requests: list[dict[str, Any]]) -> str:
        """Sends a batch api query and returns batch_id"""
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            raise Exception("Something went wrong in batch query")

        random_id = "".join(random.choices("0123456789", k=8))
        batch_id = f"dumbmar_batch_{random_id}"
        batch_obj = {
            "status": "in_progress",
            "batch_name": batch_name,
            "requests": requests,
        }
        _ = self._client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )
        return batch_id

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            raise Exception("Something went wrong in parsing batch results")

        batch_obj = self._get_batch_obj(batch_id)
        requests: list[dict[str, Any]] = batch_obj["requests"]  # pyright: ignore[reportAny]
        batch_results: list[BatchResult] = []
        for req in requests:
            custom_id = cast(str, req["custom_id"])
            if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        output=QueryResult(),
                        error_message="Dumbmar batch query failed",
                    )
                )
            else:
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        output=QueryResult(output_text="Dumbmar batch query succeeded"),
                    )
                )
        return batch_results

    def _get_batch_obj(self, batch_id: str) -> dict[str, Any]:
        batch_obj: dict[str, Any] = json.loads(  # pyright: ignore[reportAny]
            self._client.get(f"dummy_batch:{batch_id}")  # pyright: ignore[reportArgumentType]
        )
        return batch_obj

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        batch_obj = self._get_batch_obj(batch_id)
        return len(batch_obj["requests"])  # pyright: ignore[reportAny]

    @override
    async def cancel_batch_request(self, batch_id: str):
        batch_obj = self._get_batch_obj(batch_id)
        batch_obj["status"] = "cancelled"
        _ = self._client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        batch_obj = self._get_batch_obj(batch_id)
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            batch_obj["status"] = "failed"
        elif batch_obj["status"] != "cancelled":
            batch_obj["status"] = random.choice(["completed", "in_progress"])

        _ = self._client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )
        return batch_obj["status"]

    @override
    @classmethod
    def is_batch_status_completed(cls, batch_status: str) -> bool:
        return True

    @override
    @classmethod
    def is_batch_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == "cancelled"

    @override
    @classmethod
    def is_batch_status_failed(cls, batch_status: str) -> bool:
        return batch_status == "failed"


@register_provider("vals")
class DummyAIModel(LLM):
    _client: Redis | None = None

    @override
    def get_client(self) -> Redis:
        if not DummyAIModel._client:
            DummyAIModel._client = redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                model_library_settings.REDIS_URL, decode_responses=True
            )
        return DummyAIModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["vals"] = "vals",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.custom_retrier = None
        self.batch: LLMBatchMixin | None = (
            DummyAIBatchMixin(self) if self.supports_batch else None
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> Any:
        new_input: list[dict[str, Any]] = []
        for item in input:
            match item:
                case TextInput():
                    new_input.append({"text": truncate_str(item.text, 16000)})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            new_input.append(await self.parse_image(item))
                        case "file":
                            new_input.append(await self.parse_file(item))
                case _:
                    raise Exception("Unsupported input type")
        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> dict[str, Any]:
        return {"image": image.model_dump()}

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> dict[str, Any]:
        return {"file": file.model_dump()}

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        parsed_tools: list[dict[str, Any]] = []
        for tool in tools:
            parsed_tools.append({"tool": tool.model_dump()})
        return parsed_tools

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise NotImplementedError()

    async def create_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        messages = await self.parse_input(input)
        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "seed": 0,
            "messages": messages,
            "tools": await self.parse_tools(tools),
        }

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        body.update(kwargs)
        return body

    async def mock_response(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body["messages"]
        # TODO: better mock response
        return {
            "id": "mock-id",
            "model": "mock-model",
            "text": "\n".join([str(message) for message in messages]),
            "reasoning": "Reasoning Model" if self.reasoning else "",
            "usage": {
                "in_tokens": 1,
                "out_tokens": 1,
            },
        }

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        body = await self.create_body(input, tools=tools, **kwargs)

        fail_rate = FAIL_RATE

        latency = 0

        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.pop("system_prompt", ""))
            if "fail_rate" in system_prompt:
                match = re.search(r"fail_rate:\s*(\d+\.?\d*)", system_prompt)
                if match:
                    fail_rate = float(match.group(1))
            if "latency" in system_prompt:
                match = re.search(r"latency:\s*(\d+\.?\d*)", system_prompt)
                if match:
                    latency = int(match.group(1))

        for i in range(latency):
            self.logger.info(f"Sleeping... {i + 1} of {latency} seconds remaining")
            time.sleep(1)

        if random.random() < fail_rate and "evaluator" not in self.model_name:
            raise Exception("Dumbmar couldn't retrieve output.")

        response = await self.mock_response(body)

        return QueryResult(
            output_text=response["text"],
            reasoning=response["reasoning"],
            metadata=QueryResultMetadata(
                in_tokens=response["usage"]["in_tokens"],
                out_tokens=response["usage"]["out_tokens"],
            ),
        )
