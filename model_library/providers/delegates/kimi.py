import asyncio
import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    FileWithId,
    InputItem,
    LLMConfig,
    ProviderConfig,
    RawInput,
    QueryResult,
    ToolDefinition,
)
from model_library.providers.openai import OpenAIConfig
from model_library.register_models import register_provider


class KimiConfig(ProviderConfig):
    parallel_tool_calls: bool | None = None


@register_provider("kimi")
class KimiModel(DelegateOnly):
    provider_config = KimiConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["kimi"] = "kimi",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility
        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint
                or "https://api.moonshot.ai/v1/",
                "custom_api_key": config.custom_api_key
                or SecretStr(model_library_settings.KIMI_API_KEY),
                "provider_config": OpenAIConfig(
                    parallel_tool_calls=self.provider_config.parallel_tool_calls
                ),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """
        Build extra body parameters for Kimi-specific features.
        see https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart#parameters-differences-in-request-body
        """
        return {"thinking": {"type": "enabled" if self.reasoning else "disabled"}}

    async def _preprocess_files(self, input: Sequence[InputItem]) -> list[InputItem]:
        """Replace file items with TextInput containing extracted text.

        Moonshot doesn't support OpenAI-style file content blocks. Files are
        uploaded via the files API, text is extracted, and injected as plain
        text so the delegate never sees file items.
        """

        async def preprocess_item(item: InputItem) -> InputItem:
            if not isinstance(item, FileWithId):
                return item

            assert self.delegate
            response = await self.delegate.get_client().files.content(
                file_id=item.file_id
            )
            return RawInput(
                input={
                    "role": "system",
                    "content": response.text,
                }
            )

        tasks = [asyncio.create_task(preprocess_item(item)) for item in input]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        preprocessed = await self._preprocess_files(input)
        return await super().delegate_query(
            preprocessed,
            tools=tools,
            query_logger=query_logger,
            extra_body=self._get_extra_body(),
            output_schema=output_schema,
            **kwargs,
        )
