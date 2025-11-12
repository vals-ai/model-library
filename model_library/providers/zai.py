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
    ToolDefinition,
)
from model_library.providers.openai import OpenAIModel
from model_library.utils import create_openai_client_with_defaults


class ZAIModel(LLM):
    @override
    def get_client(self) -> None:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        model_name: str,
        provider: Literal["zai"] = "zai",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.model_name: str = model_name
        self.native: bool = False

        # https://docs.z.ai/
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.ZAI_API_KEY,
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
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
        # relies on oAI delegate
        if self.delegate:
            return await self.delegate_query(input, tools=tools, **kwargs)
        raise NotImplementedError()
