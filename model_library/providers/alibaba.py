import io
from typing import Any, Literal, Sequence

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    ToolDefinition,
)
from model_library.providers.openai import OpenAIModel
from model_library.utils import create_openai_client_with_defaults


class AlibabaModel(LLM):
    @override
    def get_client(self) -> None:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        model_name: str,
        provider: Literal["alibaba"] = "alibaba",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.native: bool = False

        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.DASHSCOPE_API_KEY,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                ),
                use_completions=True,
            )
        )

    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """
        Calculate cost for Qwen models using tiered pricing.
        Pricing is based on total input tokens:
        - 0-32k: $1.2/M input, $6/M output
        - 32k-128k: $2.4/M input, $12/M output
        - 128k+: $3/M input, $15/M output
        Cached tokens are billed at 20% of regular cost.
        """
        MILLION = 1_000_000
        CACHE_DISCOUNT = 0.20
        # Calculate total input tokens (including cached tokens)
        total_input_tokens = metadata.in_tokens + (metadata.cache_read_tokens or 0)

        # Determine pricing tier based on total input tokens
        if total_input_tokens <= 32_000:
            input_price = 1.2
            output_price = 6.0
        elif total_input_tokens <= 128_000:
            input_price = 2.4
            output_price = 12.0
        else:
            input_price = 3.0
            output_price = 15.0

        # Calculate cache costs (20% of regular price)
        cache_read_cost = input_price * CACHE_DISCOUNT

        # Calculate actual costs
        return QueryResultCost(
            input=input_price * metadata.in_tokens / MILLION,
            output=output_price * metadata.out_tokens / MILLION,
            reasoning=output_price * metadata.reasoning_tokens / MILLION
            if metadata.reasoning_tokens is not None and bill_reasoning
            else None,
            cache_read=cache_read_cost * metadata.cache_read_tokens / MILLION
            if metadata.cache_read_tokens is not None
            else None,
            cache_write=None,
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError()

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Any:
        raise NotImplementedError()

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Any:
        raise NotImplementedError()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> Any:
        raise NotImplementedError()

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
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate_query(input, tools=tools, **kwargs)
        raise NotImplementedError()
