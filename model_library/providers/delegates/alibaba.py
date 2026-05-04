import re
from typing import Any, Literal, Sequence

from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
    ToolDefinition,
)
from model_library.register_models import register_provider


class AlibabaConfig(ProviderConfig):
    """Configuration for Alibaba (Qwen) models.

    Attributes:
        preserve_thinking: When enabled on Qwen 3.6+ reasoning models, previous
            reasoning content is preserved in context across turns instead of
            being stripped and re-serialized. This improves KV cache utilization
            and decision consistency in agentic workflows.
            See: https://qwen.ai/blog?id=qwen3.6-27b
    """

    preserve_thinking: bool = False


@register_provider("alibaba")
class AlibabaModel(DelegateOnly):
    provider_config = AlibabaConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["alibaba"] = "alibaba",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.preserve_thinking = (
            self.provider_config.preserve_thinking and self._is_qwen_36_or_later()
        )

        # https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.DASHSCOPE_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    def _fix_content_null_in_messages(self, messages: list[Any]) -> list[Any]:
        """Set content to \"\" for assistant messages with content=None so Qwen API accepts the request."""
        fixed: list[Any] = []
        for msg in messages:
            if isinstance(msg, ChatCompletionMessage) and msg.content is None:
                fixed.append(msg.model_copy(update={"content": ""}))
            else:
                fixed.append(msg)
        return fixed

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        body = await super().build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )
        if "messages" in body:
            body["messages"] = self._fix_content_null_in_messages(body["messages"])
        return body

    def _is_qwen_36_or_later(self) -> bool:
        """Check if the model is Qwen 3.6 or later based on the model name."""
        match = re.search(r"qwen(\d+(?:\.\d+)?)", self.model_name)
        if not match:
            return False
        return float(match.group(1)) >= 3.6

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """Build extra body parameters for Qwen-specific features."""
        extra: dict[str, Any] = {}
        # Enable thinking mode for Qwen3 reasoning models
        # https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api
        if self.reasoning:
            extra["enable_thinking"] = True
            if self.preserve_thinking:
                extra["preserve_thinking"] = True
        return extra

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        # qwen3-max and qwen3-vl-plus use hardcoded tiered pricing
        if "qwen3-max" not in self.model_name and "qwen3-vl" not in self.model_name:
            return await super()._calculate_cost(metadata, batch, bill_reasoning)

        # Hardcoded tiered pricing for qwen3-max models
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
