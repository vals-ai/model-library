import io
from typing import Any, Literal, Sequence

from openai import AsyncOpenAI
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
    InputItem,
    LLMConfig,
    QueryResult,
    ToolDefinition,
)
from model_library.providers.openai import OpenAIModel
from model_library.utils import create_openai_client_with_defaults


class PerplexityModel(LLM):
    """Perplexity Sonar models via OpenAI-compatible API."""

    @override
    def get_client(self) -> None:  # pragma: no cover - delegate is used instead
        raise NotImplementedError("Perplexity models are accessed via delegate client")

    def __init__(
        self,
        model_name: str,
        provider: Literal["perplexity"] = "perplexity",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.native = False
        client: AsyncOpenAI = create_openai_client_with_defaults(
            api_key=model_library_settings.PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        )

        self.delegate = OpenAIModel(
            model_name=model_name,
            provider=provider,
            config=config,
            custom_client=client,
            use_completions=True,
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: object,
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
        return await self.delegate_query(
            input,
            tools=tools,
            **kwargs,
        )
