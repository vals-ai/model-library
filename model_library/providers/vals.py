from __future__ import annotations

import io
import logging
import random
import re
import time
from typing import Any, Literal, Sequence

import redis
from redis.client import Redis
from pydantic import BaseModel
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
    SystemInput,
    TextInput,
    ToolDefinition,
)
from model_library.exceptions import UnexpectedSystemInputError
from model_library.register_models import register_provider
from model_library.utils import truncate_str

FAIL_RATE = 0.1


@register_provider("vals")
class DummyAIModel(LLM):
    _client: Redis | None = None

    def _get_default_api_key(self) -> str:
        return model_library_settings.REDIS_URL

    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Redis:
        if base_url:
            self.instance_logger.warning(
                "custom_endpoint is not supported by this provider and will be ignored"
            )

        if not self.has_client():
            assert api_key
            client = redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                model_library_settings.REDIS_URL, decode_responses=True
            )
            self.assign_client(client)
        return super().get_client()

    def __init__(
        self,
        model_name: str,
        provider: Literal["vals"] = "vals",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.custom_retrier = None

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> Any:
        new_input: list[dict[str, Any]] = []
        for item in input:
            match item:
                case SystemInput():
                    raise UnexpectedSystemInputError()
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

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        if isinstance(input[0], SystemInput):
            input = input[1:]
        messages = await self.parse_input(input)
        body: dict[str, Any] = {
            "model": self.model_name,
            "seed": 0,
            "messages": messages,
            "tools": await self.parse_tools(tools),
        }

        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        body.update(self._sampling_from_config(("temperature", "top_p")))
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
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        fail_rate = FAIL_RATE

        latency = 0

        system_prompt: str | None = None
        if isinstance(input[0], SystemInput):
            system_prompt = input[0].text
        if system_prompt is not None:
            if "fail_rate" in system_prompt:
                match = re.search(r"fail_rate:\s*(\d+\.?\d*)", system_prompt)
                if match:
                    fail_rate = float(match.group(1))
            if "latency" in system_prompt:
                match = re.search(r"latency:\s*(\d+\.?\d*)", system_prompt)
                if match:
                    latency = int(match.group(1))

        for i in range(latency):
            query_logger.info(f"Sleeping... {i + 1} of {latency} seconds remaining")
            time.sleep(1)

        if random.random() < fail_rate and "evaluator" not in self.model_name:
            raise Exception("Dumbmar couldn't retrieve output.")

        response = await self.mock_response(body)

        return QueryResult(
            output_text=response["text"],
            reasoning=response["reasoning"],
            finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
            metadata=QueryResultMetadata(
                in_tokens=response["usage"]["in_tokens"],
                out_tokens=response["usage"]["out_tokens"],
            ),
            extras=QueryResultExtras(
                provider_response_id=response.get("id"),
            ),
        )
