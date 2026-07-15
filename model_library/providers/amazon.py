# pyright: basic
import asyncio
import base64
import io
import json
import logging
from typing import Any, Literal, Sequence, cast

import boto3
import botocore
from botocore.client import BaseClient
from pydantic import BaseModel
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.exceptions import (
    BadInputError,
    NoMatchingToolCallError,
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.model_utils import get_default_budget_tokens
from model_library.agent.tool import is_native_web_search
from model_library.register_models import register_provider


def map_amazon_finish_reason(
    stop_reason: str | None,
) -> FinishReasonInfo:
    match stop_reason:
        case "end_turn":
            reason = FinishReason.STOP
        case "max_tokens":
            reason = FinishReason.MAX_TOKENS
        case "stop_sequence":
            reason = FinishReason.STOP_SEQUENCE
        case "tool_use":
            reason = FinishReason.TOOL_CALLS
        case "content_filtered":
            reason = FinishReason.CONTENT_FILTER
        case "guardrail_intervened":
            reason = FinishReason.GUARDRAIL
        case _:
            reason = FinishReason.UNKNOWN

    return FinishReasonInfo(reason=reason, raw=stop_reason)


@register_provider("amazon")
@register_provider("bedrock")
class AmazonModel(LLM):
    @override
    def _get_default_api_key(self) -> str:
        if getattr(model_library_settings, "AWS_ACCESS_KEY_ID", None):
            creds: dict[str, str] = {
                "AWS_ACCESS_KEY_ID": model_library_settings.AWS_ACCESS_KEY_ID,
                "AWS_SECRET_ACCESS_KEY": model_library_settings.AWS_SECRET_ACCESS_KEY,
                "AWS_DEFAULT_REGION": model_library_settings.AWS_DEFAULT_REGION,
            }
            session_token = model_library_settings.get("AWS_SESSION_TOKEN")
            if session_token:
                creds["AWS_SESSION_TOKEN"] = session_token
            return json.dumps(creds)
        return "using-environment"

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> BaseClient:
        if base_url:
            self.instance_logger.warning(
                "custom_endpoint is not supported by this provider and will be ignored"
            )

        if not self.has_client():
            assert api_key
            if api_key != "using-environment":
                creds = json.loads(api_key)
                client = cast(
                    BaseClient,
                    boto3.client(
                        "bedrock-runtime",
                        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
                        aws_session_token=creds.get("AWS_SESSION_TOKEN"),
                        region_name=creds["AWS_DEFAULT_REGION"],
                        config=botocore.config.Config(max_pool_connections=1000),  # pyright: ignore[reportAttributeAccessIssue]
                    ),
                )
            else:
                client = cast(
                    BaseClient,
                    boto3.client(
                        "bedrock-runtime",
                        # default connection pool is 10
                        config=botocore.config.Config(max_pool_connections=1000),  # pyright: ignore[reportAttributeAccessIssue]
                    ),
                )

            self.assign_client(client)
        return super().get_client()

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

        if config and config.custom_api_key:
            raise Exception(
                "custom_api_key is not currently supported for Amazon models"
            )

    cache_control = {"type": "default"}

    async def get_tool_call_ids(self, input: Sequence[InputItem]) -> list[str]:
        raw_responses = [x for x in input if isinstance(x, RawResponse)]
        tool_call_ids: list[str] = []

        calls = [
            y["toolUse"]
            for x in raw_responses
            if "content" in x.response
            for y in x.response["content"]
            if "toolUse" in y
        ]
        tool_call_ids.extend([x["toolUseId"] for x in calls])
        return tool_call_ids

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        new_input: list[dict[str, Any] | Any] = []

        content_user: list[dict[str, Any]] = []

        def flush_content_user():
            if content_user:
                # NOTE: must make new object as we clear()
                new_input.append({"role": "user", "content": content_user.copy()})
                content_user.clear()

        tool_call_ids = await self.get_tool_call_ids(input)

        for item in input:
            if isinstance(item, TextInput):
                content_user.append({"text": item.text})
                continue

            if isinstance(item, FileBase):
                match item.type:
                    case "image":
                        parsed = await self.parse_image(item)
                    case "file":
                        parsed = await self.parse_file(item)
                content_user.append(parsed)
                continue

            # non content user item
            flush_content_user()

            match item:
                case ToolResult():
                    if item.tool_call.id not in tool_call_ids:
                        raise NoMatchingToolCallError()

                    new_input.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        "toolUseId": item.tool_call.id,
                                        "content": [{"json": {"result": item.result}}],
                                    }
                                }
                            ],
                        }
                    )
                case RawResponse():
                    new_input.append(item.response)
                case RawInput():
                    new_input.append(item.input)
                case SystemInput():
                    raise UnexpectedSystemInputError()

        if content_user and self.supports_cache:
            if not isinstance(input[-1], FileBase):
                # last item cannot be file
                content_user.append({"cachePoint": self.cache_control})

        flush_content_user()

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
            if is_native_web_search(body):
                parsed_tools.append(self.search_tool)
                continue
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

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        system_text: str | None = None
        if isinstance(input[0], SystemInput):
            system_text = input[0].text
            input = input[1:]

        messages: list[dict[str, Any]] = []
        messages.extend(await self.parse_input(input))

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        body: dict[str, Any] = {"modelId": self.model_name, "messages": messages}
        if tools:
            body["toolConfig"] = {"tools": await self.parse_tools(tools)}

        if system_text is not None:
            system: list[dict[str, Any]] = [{"text": system_text}]
            if self.supports_cache:
                system.append({"cachePoint": self.cache_control})
            body["system"] = system

        if self.reasoning and self.max_tokens:
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

        inference: dict[str, Any] = {}

        if self.max_tokens:
            inference["maxTokens"] = self.max_tokens

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
        result_builder: QueryResultBuilder | None = None,
    ):
        text_response = ""
        reasoning_content = ""
        reasoning_signature = ""
        tool_calls: dict[str, Any] = {}

        messages: dict[str, Any] = {"content": []}
        stop_reason: str = ""
        metadata = QueryResultMetadata()
        result_builder = result_builder or QueryResultBuilder()

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
                        result_builder.start_tool_call_segment().record_tool_call_ready()
                        tool_calls["toolUseId"] = tool["toolUseId"]
                        tool_calls["name"] = tool["name"]

                case "contentBlockDelta":
                    delta = value["delta"]
                    delta_key = list(delta.keys())[0]
                    match delta_key:
                        case "reasoningContent":
                            if "text" in delta["reasoningContent"]:
                                reasoning_delta = delta["reasoningContent"]["text"]
                                reasoning_content += reasoning_delta
                                result_builder.append_reasoning_delta(reasoning_delta)
                            if "signature" in delta["reasoningContent"]:
                                reasoning_signature = delta["reasoningContent"][
                                    "signature"
                                ]
                        case "text":
                            text_delta = delta["text"]
                            text_response += text_delta
                            result_builder.append_content_delta(text_delta)
                        case "toolUse":
                            if "input" not in tool_calls:
                                tool_calls["input"] = ""
                            tool_input_delta = delta["toolUse"]["input"]
                            tool_calls["input"] += tool_input_delta
                            if tool_input_delta:
                                result_builder.record_tool_call_delta()

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
                    result_builder.finish_current_segment()
                    if tool_calls:
                        raw_tool_input = tool_calls.get("input") or "{}"
                        tool_calls["input"] = json.loads(raw_tool_input)
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

        return messages, stop_reason, metadata, result_builder

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#
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

        result_builder = QueryResultBuilder()
        response = await asyncio.to_thread(
            self.get_client().converse_stream,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            **body,
        )

        request_id = None
        if response_metadata := response.get("ResponseMetadata"):
            request_id = response_metadata.get("RequestId")

        messages, stop_reason, metadata, result_builder = await self.stream_response(
            response, result_builder
        )

        text_blocks = [i["text"] for i in messages["content"] if "text" in i]
        text = " ".join(text for text in text_blocks if text)
        reasoning_blocks = [
            i["reasoningContent"]["reasoningText"]["text"]
            for i in messages["content"]
            if "reasoningContent" in i
        ]
        reasoning = " ".join(reasoning_blocks)

        tool_calls: list[ToolCall] = []
        if stop_reason == "tool_use":
            for item in messages["content"]:
                if "toolUse" not in item:
                    continue
                tool = item["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tool["toolUseId"],
                        name=tool["name"],
                        args=tool["input"],
                    )
                )

        mapped_finish_reason = map_amazon_finish_reason(stop_reason)
        no_useful_content = not text_blocks and not reasoning and not tool_calls
        if no_useful_content:
            handle_empty_response(mapped_finish_reason, {"metadata": metadata})

        if text_blocks:
            result_builder.set_output_text(text)
        if reasoning_blocks:
            result_builder.set_reasoning(reasoning)

        return result_builder.build(
            finish_reason=mapped_finish_reason,
            metadata=metadata,
            extras=QueryResultExtras(
                provider_response_id=request_id,
                provider_request_id=request_id,
            ),
            tool_calls=tool_calls,
            history=[*input, RawResponse(response=messages)],
        )
