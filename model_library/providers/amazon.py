# pyright: basic
import asyncio
import base64
import io
import json
from typing import Any, Literal, Sequence, cast

import boto3
import botocore
from botocore.client import BaseClient
from typing_extensions import override

from model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.input import FileBase
from model_library.exceptions import (
    BadInputError,
    MaxOutputTokensExceededError,
)
from model_library.model_utils import get_default_budget_tokens
from model_library.register_models import register_provider


@register_provider("amazon")
@register_provider("bedrock")
class AmazonModel(LLM):
    _client: BaseClient | None = None

    @override
    def get_client(self) -> BaseClient:
        if not AmazonModel._client:
            AmazonModel._client = cast(
                BaseClient,
                boto3.client(
                    "bedrock-runtime",
                    # default connection pool is 10
                    config=botocore.config.Config(max_pool_connections=1000),  # pyright: ignore[reportAttributeAccessIssue]
                ),
            )  # pyright: ignore[reportUnknownMemberType]
        return AmazonModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["amazon"] = "amazon",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.supports_cache = "amazon" in self.model_name or "claude" in self.model_name
        self.supports_cache = (
            self.supports_cache and "v2" not in self.model_name
        )  # supported but no access yet
        self.supports_tool_cache = self.supports_cache and "claude" in self.model_name

    cache_control = {"type": "default"}

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        new_input: list[dict[str, Any] | Any] = []
        content_user: list[dict[str, Any]] = []

        for item in input:
            match item:
                case TextInput():
                    content_user.append({"text": item.text})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            content_user.append(await self.parse_image(item))
                        case "file":
                            content_user.append(await self.parse_file(item))
                case _:
                    if content_user:
                        new_input.append({"role": "user", "content": content_user})
                        content_user = []
                    match item:
                        case ToolResult():
                            if not (
                                isinstance(x, dict)
                                and "toolUse" in x
                                and x["toolUse"].get("toolUseId")
                                == item.tool_call.call_id
                                for x in new_input
                            ):
                                raise Exception(
                                    "Tool call result provided with no matching tool call"
                                )
                            new_input.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "toolResult": {
                                                "toolUseId": item.tool_call.id,
                                                "content": [
                                                    {"json": {"result": item.result}}
                                                ],
                                            }
                                        }
                                    ],
                                }
                            )
                        case dict():  # RawInputItem and RawResponse
                            new_input.append(item)

        if content_user:
            if self.supports_cache:
                if not isinstance(input[-1], FileBase):
                    # last item cannot be file
                    content_user.append({"cachePoint": self.cache_control})
            new_input.append({"role": "user", "content": content_user})

        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> dict[str, Any]:
        match image:
            case FileWithBase64():
                image_bytes = base64.b64decode(image.base64)
                return {
                    "image": {
                        "format": image.mime,
                        "source": {"bytes": image_bytes},
                    },
                }
            case _:
                raise BadInputError("Unsupported image type")

    @override
    async def parse_file(
        self,
        file: FileInput,
    ):
        raise NotImplementedError()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        parsed_tools: list[dict[str, Any]] = []
        for tool in tools:
            body = tool.body
            if not isinstance(body, ToolBody):
                parsed_tools.append(body)
                continue
            parsed_tools.append(
                {
                    "toolSpec": {
                        "name": body.name,
                        "description": body.description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": body.properties,
                                "required": body.required,
                            }
                        },
                    }
                }
            )
        if parsed_tools and self.supports_tool_cache:
            parsed_tools.append({"cachePoint": self.cache_control})
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

    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        messages.extend(await self.parse_input(input))

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        body: dict[str, Any] = {"modelId": self.model_name, "messages": messages}
        if tools:
            body["toolConfig"] = {"tools": await self.parse_tools(tools)}

        if "system_prompt" in kwargs:
            body["system"] = [{"text": kwargs.pop("system_prompt")}]
            if self.supports_cache:
                body["system"].append({"cachePoint": self.cache_control})

        if self.reasoning:
            if self.max_tokens < 1024:
                self.max_tokens = 2048
            budget_tokens = kwargs.pop(
                "budget_tokens", get_default_budget_tokens(self.max_tokens)
            )
            body["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
            }

        inference: dict[str, Any] = {
            "maxTokens": self.max_tokens,
        }

        # Only set temperature for models where supports_temperature is True.
        # For example, "thinking" models don't support temperature: https://docs.claude.com/en/docs/build-with-claude/extended-thinking#feature-compatibility
        if self.supports_temperature:
            if self.temperature is not None:
                inference["temperature"] = self.temperature
            if self.top_p is not None:
                inference["topP"] = self.top_p

        inference.update(
            kwargs
        )  # NOTE: in the future need to change so it updates body, and checks if any fields belong inside inference

        body["inferenceConfig"] = inference
        return body

    async def stream_response(
        self,
        response: Any,
    ):
        text_response = ""
        reasoning_content = ""
        reasoning_signature = ""
        tool_calls: dict[str, Any] = {}

        messages: dict[str, Any] = {"content": []}
        stop_reason: str = ""
        metadata = QueryResultMetadata()

        for chunk in response["stream"]:
            key = list(chunk.keys())[0]
            value = chunk[key]
            match key:
                case "messageStart":
                    messages["role"] = chunk["messageStart"]["role"]
                case "contentBlockStart":
                    start = value["start"]
                    start_key = list(start.keys())[0]
                    if start_key == "toolUse":
                        tool = start["toolUse"]
                        tool_calls["toolUseId"] = tool["toolUseId"]
                        tool_calls["name"] = tool["name"]

                case "contentBlockDelta":
                    delta = value["delta"]
                    delta_key = list(delta.keys())[0]
                    match delta_key:
                        case "reasoningContent":
                            if "text" in delta["reasoningContent"]:
                                reasoning_content += delta["reasoningContent"]["text"]
                            if "signature" in delta["reasoningContent"]:
                                reasoning_signature = delta["reasoningContent"][
                                    "signature"
                                ]
                        case "text":
                            text_response += delta["text"]
                        case "toolUse":
                            if "input" not in tool_calls:
                                tool_calls["input"] = ""
                            tool_calls["input"] += delta["toolUse"]["input"]

                case "metadata":
                    metadata = QueryResultMetadata(
                        in_tokens=value["usage"]["inputTokens"],
                        out_tokens=value["usage"]["outputTokens"],
                    )
                    metadata.cache_read_tokens = value["usage"].get(
                        "cacheReadInputTokens", None
                    )
                    metadata.cache_write_tokens = value["usage"].get(
                        "cacheWriteInputTokens", None
                    )

                case "contentBlockStop":
                    if tool_calls:
                        tool_calls["input"] = json.loads(tool_calls["input"])
                        messages["content"].append({"toolUse": tool_calls})
                        tool_calls = {}
                    if text_response:
                        messages["content"].append({"text": text_response})
                        text_response = ""
                    if reasoning_content:
                        messages["content"].append(
                            {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": reasoning_content,
                                        "signature": reasoning_signature,
                                    },
                                }
                            }
                        )
                        reasoning_content = ""

                case "messageStop":
                    stop_reason = value["stopReason"]

        return messages, stop_reason, metadata

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#
    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        body = await self.build_body(input, tools=tools, **kwargs)

        response = await asyncio.to_thread(
            self.get_client().converse_stream,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            **body,
        )

        messages, stop_reason, metadata = await self.stream_response(response)

        text = " ".join([i["text"] for i in messages["content"] if "text" in i])
        reasoning = " ".join(
            [
                i["reasoningContent"]["reasoningText"]["text"]
                for i in messages["content"]
                if "reasoningContent" in i
            ]
        )

        tool_calls: list[ToolCall] = []
        if stop_reason:
            match stop_reason:
                case "max_tokens":
                    if not text and not reasoning:
                        raise MaxOutputTokensExceededError()
                case "tool_use":
                    _tool_calls = [
                        i["toolUse"] for i in messages["content"] if "toolUse" in i
                    ]
                    for tool in _tool_calls:
                        tool_calls.append(
                            ToolCall(
                                id=tool["toolUseId"],
                                name=tool["name"],
                                args=tool["input"],
                            )
                        )

        return QueryResult(
            output_text=text,
            reasoning=reasoning,
            metadata=metadata,
            tool_calls=tool_calls,
            history=[*input, messages],
        )
