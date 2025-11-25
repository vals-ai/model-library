import io
from typing import Any, Literal, Sequence, cast

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types.beta.beta_tool_use_block import BetaToolUseBlock
from anthropic.types.message import Message
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
    QueryResultCost,
    QueryResultMetadata,
    RawInputItem,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.exceptions import (
    MaxOutputTokensExceededError,
)
from model_library.model_utils import get_default_budget_tokens
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import (
    create_openai_client_with_defaults,
    default_httpx_client,
    filter_empty_text_blocks,
    normalize_tool_result,
)


class AnthropicBatchMixin(LLMBatchMixin):
    """Batch processing support for Anthropic's Message Batches API."""

    COMPLETED_RESULT_TYPES = ["succeeded", "errored", "canceled", "expired"]

    def __init__(self, model: "AnthropicModel"):
        self._root = model

    @override
    async def create_batch_query_request(
        self,
        custom_id: str,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> dict[str, Any]:
        """Create a single batch request in Anthropic's format.

        Format: {"custom_id": str, "params": {...message params...}}
        """
        # Build the message body using the parent model's create_body method
        tools = cast(list[ToolDefinition], kwargs.pop("tools", []))
        body = await self._root.create_body(input, tools=tools, **kwargs)

        return {
            "custom_id": custom_id,
            "params": body,
        }

    @override
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        """Submit a batch of requests to Anthropic's Message Batches API.

        Returns the batch ID for status tracking.
        """
        client = self._root.get_client()

        # Create the batch using Anthropic's batches API
        batch = await client.messages.batches.create(
            requests=cast(Any, requests),  # Type mismatch in SDK, cast to Any
        )

        self._root.logger.info(
            f"Created Anthropic batch {batch.id} with {len(requests)} requests"
        )

        return batch.id

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """Retrieve results from a completed batch.

        Streams results using the SDK's batches.results() method.
        """
        client = self._root.get_client()

        # Get batch status to verify it's completed
        batch = await client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            raise ValueError(
                f"Batch {batch_id} is not completed yet. Status: {batch.processing_status}"
            )

        # Stream results using the SDK's results method
        batch_results: list[BatchResult] = []
        async for result_item in await client.messages.batches.results(batch_id):
            # result_item is a MessageBatchIndividualResponse - convert to dict
            result_dict = result_item.model_dump()
            custom_id = cast(str, result_dict["custom_id"])
            result_type = cast(str, result_dict["result"]["type"])

            if result_type not in self.COMPLETED_RESULT_TYPES:
                self._root.logger.warning(
                    f"Unknown result type '{result_type}' for request {custom_id}"
                )
                continue

            if result_type == "succeeded":
                # Extract the message from the successful result
                message_data = cast(dict[str, Any], result_dict["result"]["message"])

                # Parse the message content to extract text, reasoning, and tool calls
                text = ""
                reasoning = ""
                tool_calls: list[ToolCall] = []

                for content in message_data.get("content", []):
                    if content.get("type") == "text":
                        text += content.get("text", "")
                    elif content.get("type") == "thinking":
                        reasoning += content.get("thinking", "")
                    elif content.get("type") == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                id=content["id"],
                                name=content["name"],
                                args=content.get("input", {}),
                            )
                        )

                # Extract usage information
                usage = message_data.get("usage", {})
                metadata = QueryResultMetadata(
                    in_tokens=usage.get("input_tokens", 0),
                    out_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                    cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                )

                query_result = QueryResult(
                    output_text=text,
                    reasoning=reasoning,
                    metadata=metadata,
                    tool_calls=tool_calls,
                    history=[],  # History not available in batch results
                )

                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        output=query_result,
                    )
                )

            elif result_type == "errored":
                # Handle errored results
                error = cast(dict[str, Any], result_dict["result"]["error"])
                error_message = f"{error.get('type', 'unknown_error')}: {error.get('message', 'Unknown error')}"
                output = QueryResult(output_text=error_message)
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        output=output,
                        error_message=error_message,
                    )
                )

            elif result_type in ["canceled", "expired"]:
                # Handle canceled/expired results
                error_message = f"Request {result_type}"
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        output=QueryResult(output_text=""),
                        error_message=error_message,
                    )
                )

        return batch_results

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        """Get the number of completed requests in a batch."""
        client = self._root.get_client()
        batch = await client.messages.batches.retrieve(batch_id)

        # Return the number of processed requests
        request_counts = batch.request_counts
        return (
            request_counts.succeeded
            + request_counts.errored
            + request_counts.canceled
            + request_counts.expired
        )

    @override
    async def cancel_batch_request(self, batch_id: str) -> None:
        """Cancel a running batch request."""
        client = self._root.get_client()
        await client.messages.batches.cancel(batch_id)
        self._root.logger.info(f"Canceled Anthropic batch {batch_id}")

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        """Get the current status of a batch."""
        client = self._root.get_client()
        batch = await client.messages.batches.retrieve(batch_id)
        return batch.processing_status

    @classmethod
    def is_batch_status_completed(cls, batch_status: str) -> bool:
        """Check if a batch status indicates completion."""
        return batch_status == "ended"

    @classmethod
    def is_batch_status_failed(cls, batch_status: str) -> bool:
        """Check if a batch status indicates failure."""
        # Anthropic batches can have individual request failures but the batch
        # itself doesn't have a "failed" status - it just ends
        return False

    @classmethod
    def is_batch_status_cancelled(cls, batch_status: str) -> bool:
        """Check if a batch status indicates cancellation."""
        return batch_status == "canceling" or batch_status == "canceled"


@register_provider("anthropic")
class AnthropicModel(LLM):
    _client: AsyncAnthropic | None = None

    @override
    def get_client(self) -> AsyncAnthropic:
        if not AnthropicModel._client:
            headers: dict[str, str] = {}
            AnthropicModel._client = AsyncAnthropic(
                api_key=model_library_settings.ANTHROPIC_API_KEY,
                http_client=default_httpx_client(),
                max_retries=1,
                default_headers=headers,
            )
        return AnthropicModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["anthropic"] = "anthropic",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.anthropic.com/en/api/openai-sdk
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.ANTHROPIC_API_KEY,
                    base_url="https://api.anthropic.com/v1/",
                ),
                use_completions=True,
            )
        )

        # Initialize batch support if enabled
        self.supports_batch: bool = self.supports_batch and self.native
        self.batch: LLMBatchMixin | None = (
            AnthropicBatchMixin(self) if self.supports_batch else None
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any] | Any]:
        new_input: list[dict[str, Any] | Any] = []
        content_user: list[dict[str, Any]] = []

        # First pass: collect all tool calls from Message objects for validation
        tool_calls_in_input: set[str] = set()
        for item in input:
            if hasattr(item, "content") and hasattr(item, "role"):
                content_list = getattr(item, "content", [])
                for content in content_list:
                    # Check for both ToolUseBlock and BetaToolUseBlock
                    if isinstance(content, (ToolUseBlock, BetaToolUseBlock)):
                        tool_calls_in_input.add(content.id)

        for item in input:
            match item:
                case TextInput():
                    if item.text.strip():
                        content_user.append({"type": "text", "text": item.text})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            content_user.append(await self.parse_image(item))
                        case "file":
                            content_user.append(await self.parse_file(item))
                case _:
                    if content_user:
                        filtered = filter_empty_text_blocks(content_user)
                        if filtered:
                            new_input.append({"role": "user", "content": filtered})
                        content_user = []
                    match item:
                        case ToolResult():
                            if item.tool_call.id not in tool_calls_in_input:
                                raise Exception(
                                    "Tool call result provided with no matching tool call"
                                )
                            result_str = normalize_tool_result(item.result)
                            new_input.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": item.tool_call.id,
                                            "content": [
                                                {"type": "text", "text": result_str}
                                            ],
                                        }
                                    ],
                                }
                            )
                        case dict():  # RawInputItem
                            item = cast(RawInputItem, item)
                            new_input.append(item)
                        case _:  # RawResponse
                            item = cast(Message, item)
                            filtered_content = [
                                block
                                for block in item.content
                                if not isinstance(block, TextBlock)
                                or block.text.strip()
                            ]
                            if filtered_content:
                                new_input.append(
                                    {"role": "assistant", "content": filtered_content}
                                )

        if content_user:
            filtered = filter_empty_text_blocks(content_user)
            if filtered:
                new_input.append({"role": "user", "content": filtered})

        if new_input:
            last_msg = new_input[-1]
            if not isinstance(last_msg, dict):
                return new_input

            last_msg_dict: dict[str, Any] = cast(dict[str, Any], last_msg)
            if last_msg_dict.get("role") != "user":
                return new_input

            content = last_msg_dict.get("content")
            if not isinstance(content, list) or not content:
                return new_input

            content_list: list[Any] = cast(list[Any], content)
            last_block = content_list[-1]
            if isinstance(last_block, dict):
                last_block_dict: dict[str, Any] = cast(dict[str, Any], last_block)
                last_block_dict.setdefault("cache_control", self.cache_control)

        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> dict[str, Any]:
        match image:
            case FileWithBase64():
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image.mime}",
                        "data": image.base64,
                    },
                }
            case FileWithUrl():
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image.url,
                    },
                }
            case FileWithId():
                return {
                    "type": "image",
                    "source": {
                        "type": "file",
                        "file_id": image.file_id,
                    },
                }

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> dict[str, Any]:
        match file:
            case FileWithBase64():
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": file.mime,
                        "data": file.base64,
                    },
                }
            case FileWithUrl():
                return {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": file.url,
                    },
                }
            case FileWithId():
                return {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file.file_id,
                    },
                }

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
                    "name": body.name,
                    "description": body.description,
                    "input_schema": {
                        "type": "object",
                        "properties": body.properties,
                        "required": body.required,
                    },
                }
            )
        return parsed_tools

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        file_mime = f"image/{mime}" if type == "image" else mime  # TODO:
        response = await self.get_client().beta.files.upload(
            file=(
                name,
                bytes,
                file_mime,
            ),
        )

        return FileWithId(
            type=type,
            file_id=response.id,
            name=response.filename,
            mime=mime,
        )

    cache_control = {"type": "ephemeral"}  # 5 min cache

    async def create_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model": self.model_name,
            "messages": await self.parse_input(input),
        }

        if "system_prompt" in kwargs:
            body["system"] = [
                {
                    "type": "text",
                    "text": kwargs.pop("system_prompt"),
                    "cache_control": self.cache_control,
                }
            ]

        if self.reasoning:
            budget_tokens = kwargs.pop(
                "budget_tokens", get_default_budget_tokens(self.max_tokens)
            )
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        # Thinking models don't support temperature: https://docs.claude.com/en/docs/build-with-claude/extended-thinking#feature-compatibility
        if self.supports_temperature and not self.reasoning:
            if self.temperature is not None:
                body["temperature"] = self.temperature

        parsed_tools = await self.parse_tools(tools)
        if parsed_tools:
            if "system" not in body:
                parsed_tools[-1]["cache_control"] = self.cache_control
            body["tools"] = parsed_tools

        body.update(kwargs)

        return body

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate_query(input, tools=tools, **kwargs)

        body = await self.create_body(input, tools=tools, **kwargs)

        betas = ["files-api-2025-04-14", "interleaved-thinking-2025-05-14"]
        if "sonnet-4-5" in self.model_name:
            betas.append("context-1m-2025-08-07")

        async with self.get_client().beta.messages.stream(
            **body,
            betas=betas,
        ) as stream:  # pyright: ignore[reportAny]
            message = await stream.get_final_message()
        self.logger.info(f"Anthropic Response finished: {message.id}")

        text = ""
        reasoning = ""
        tool_calls: list[ToolCall] = []
        for content in message.content:
            if content.type == "text":
                text += content.text
            if content.type == "thinking":
                reasoning += content.thinking
            if content.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=content.id,
                        name=content.name,
                        args=cast(Any, content.input),
                    )
                )

        if message.stop_reason == "max_tokens" and not text and not reasoning:
            raise MaxOutputTokensExceededError()

        return QueryResult(
            output_text=text,
            reasoning=reasoning,
            metadata=QueryResultMetadata(
                # see _calculate_cost
                in_tokens=message.usage.input_tokens,
                out_tokens=message.usage.output_tokens,
                cache_read_tokens=message.usage.cache_read_input_tokens,
                cache_write_tokens=message.usage.cache_creation_input_tokens,
            ),
            tool_calls=tool_calls,
            history=[*input, message],
        )

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """
        Future Cost considerations
        Per 1000 calls:
        - Web Search
        Free:
        - Web Fetch
        """
        # prompt caching manually enabled
        # assumed that cache tokens are ephemeral_5m_input_tokens
        return await super()._calculate_cost(metadata, batch, bill_reasoning=False)
