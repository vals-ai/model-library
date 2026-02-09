import logging
from typing import Any, Literal, Sequence

from typing_extensions import override

from pydantic import SecretStr

from model_library import model_library_settings
from model_library.base import (
    DelegateConfig,
    DelegateOnly,
    FileWithId,
    InputItem,
    LLMConfig,
    RawInput,
    QueryResult,
    ToolDefinition,
)
from model_library.register_models import register_provider


@register_provider("kimi")
class KimiModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["kimi"] = "kimi",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility
        self.init_delegate(
            config=config,
            delegate_config=DelegateConfig(
                base_url="https://api.moonshot.ai/v1/",
                api_key=SecretStr(model_library_settings.KIMI_API_KEY),
            ),
            use_completions=True,
            delegate_provider="openai",
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
        preprocessed: list[InputItem] = []
        for item in input:
            if isinstance(item, FileWithId):
                assert self.delegate
                response = await self.delegate.get_client().files.content(
                    file_id=item.file_id
                )
                preprocessed.append(
                    RawInput(
                        input={
                            "role": "system",
                            "content": response.text,
                        }
                    )
                )
            else:
                preprocessed.append(item)
        return preprocessed

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        **kwargs: object,
    ) -> QueryResult:
        preprocessed = await self._preprocess_files(input)
        return await super().delegate_query(
            preprocessed,
            tools=tools,
            query_logger=query_logger,
            extra_body=self._get_extra_body(),
            **kwargs,
        )
