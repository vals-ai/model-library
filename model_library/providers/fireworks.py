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
    ProviderConfig,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    ToolDefinition,
)
from model_library.providers.openai import OpenAIModel
from model_library.utils import create_openai_client_with_defaults


class FireworksConfig(ProviderConfig):
    serverless: bool = True


class FireworksModel(LLM):
    provider_config = FireworksConfig()

    @override
    def get_client(self) -> None:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        model_name: str,
        provider: Literal["fireworks"] = "fireworks",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        if self.provider_config.serverless:
            self.model_name = "accounts/fireworks/models/" + self.model_name
        else:
            self.model_name = "accounts/rayan-936e28/deployedModels/" + self.model_name

        # not using Fireworks SDK
        self.native: bool = False

        # https://docs.fireworks.ai/tools-sdks/openai-compatibility
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.FIREWORKS_API_KEY,
                    base_url="https://api.fireworks.ai/inference/v1",
                ),
                use_completions=True,
            )
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

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        # https://docs.fireworks.ai/guides/prompt-caching
        # Prompt caching is enabled by default for all Fireworks models and deployments.

        # Discounts for prompt caching are available for enterprise deployments. Contact us to learn more.

        # https://docs.fireworks.ai/faq-new/billing-pricing/is-prompt-caching-billed-differently
        # prompt caching does not affect billing for serverless models

        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)
