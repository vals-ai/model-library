import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    QueryResult,
    ToolDefinition,
)
from model_library.exceptions import MaxContextWindowExceededError
from model_library.register_models import register_provider


@register_provider("stepfun")
class StepFunModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["stepfun"] = "stepfun",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://platform.stepfun.ai/docs
        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint
                or "https://api.stepfun.ai/v1",
                "custom_api_key": config.custom_api_key
                or SecretStr(model_library_settings.STEPFUN_API_KEY),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=True,
        )

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
        try:
            return await super()._query_impl(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )
        except Exception as e:
            if "The input you provided is invalid" in str(e):
                raise MaxContextWindowExceededError(str(e)) from e
            raise
