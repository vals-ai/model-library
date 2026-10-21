from __future__ import annotations

import io
import logging
from collections.abc import Generator, Sequence
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import BaseModel
from typing_extensions import deprecated, override

from model_library import model_library_settings
from model_library.base import (
    FileInput,
    FileWithId,
    InputItem,
    LLM,
    LLMConfig,
    ProviderConfig,
    PydanticT,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    ToolDefinition,
    TranscriptionResult,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.rate_limits import RateLimit
from model_library.exceptions import (
    ImmediateRetryException,
    register_retryable_exceptions,
)
from model_library.base.transcription import TranscriptionRequest
from model_library.providers.openai import voice
from model_library.register_models import register_provider
from model_library.retriers.base import BaseRetrier
from model_library.utils import create_openai_client_with_defaults

from . import completions, responses

register_retryable_exceptions(
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    UnprocessableEntityError,
    APIConnectionError,
)


OpenAIToolCallMode = Literal["default", "auto", "code_mode"]


class OpenAIConfig(ProviderConfig):
    use_completions: bool = False
    deep_research: bool = False
    tool_call_mode: OpenAIToolCallMode = "default"
    verbosity: Literal["low", "medium", "high"] | None = None
    prompt_cache_retention: Literal["24h", "in_memory"] | None = None
    prompt_cache_key: PromptCacheKeyMode | None = None
    supports_prompt_cache_options: bool = False
    reasoning_context: Literal["current_turn", "all_turns"] | None = None
    parallel_tool_calls: bool | None = None
    service_tier: Literal["default", "flex"] | None = None
    access_programs: dict[str, str] | None = None
    # TODO: move to LLMConfig so OpenAI-compatible delegate providers can configure it.
    stream_completions: bool = True


@register_provider("openai")
class OpenAIModel(LLM):
    api_key_setting = "OPENAI_API_KEY"

    provider_config = OpenAIConfig()

    @override
    def _get_default_api_key(self) -> str:
        return self._api_key_from(model_library_settings)

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncOpenAI:
        if not self.has_client():
            assert api_key
            dns_resolve: dict[str, str] | None = None
            client = create_openai_client_with_defaults(
                base_url=base_url,
                api_key=api_key,
                dns_resolve=dns_resolve,
            )
            self.assign_client(client)
        return super().get_client()

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        *,
        config: LLMConfig | None = None,
        use_completions: bool | None = None,
        normalize_null_assistant_history_fields: bool = False,
    ):
        self._api = completions if use_completions else responses
        self.normalize_null_assistant_history_fields = (
            normalize_null_assistant_history_fields
        )
        # Set by router delegates: capture the served model and router-reported cost.
        self.router_mode: bool = False

        super().__init__(model_name, provider, config=config)
        if use_completions is None and self.provider_config.use_completions:
            self._api = completions

        self.deep_research = self.provider_config.deep_research
        self.tool_call_mode = self.provider_config.tool_call_mode
        self.verbosity = self.provider_config.verbosity
        self.prompt_cache_retention = self.provider_config.prompt_cache_retention
        self.prompt_cache_key_mode: PromptCacheKeyMode | None = (
            self.provider_config.prompt_cache_key
        )
        self.supports_prompt_cache_options = (
            self.provider_config.supports_prompt_cache_options
        )
        self.reasoning_context = self.provider_config.reasoning_context
        self.parallel_tool_calls = self.provider_config.parallel_tool_calls
        self.service_tier = self.provider_config.service_tier
        self.access_programs = self.provider_config.access_programs
        self.stream_completions = self.provider_config.stream_completions

    @override
    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__rich_repr__():
            if name == "_api":
                yield "use_completions", self._api is completions
            else:
                yield name, value

    async def get_tool_call_ids(self, input: Sequence[InputItem]) -> list[str]:
        return await self._api.get_tool_call_ids(self, input)

    @override
    async def parse_input(
        self, input: Sequence[InputItem], **kwargs: Any
    ) -> list[dict[str, Any] | Any]:
        return await self._api.parse_input(self, input, **kwargs)

    @override
    async def parse_image(self, image: FileInput) -> dict[str, Any]:
        return await self._api.parse_image(self, image)

    @override
    async def parse_file(self, file: FileInput) -> dict[str, Any]:
        return await self._api.parse_file(self, file)

    @property
    @override
    def search_tool(self) -> dict[str, str]:
        return responses.search_tool(self)

    @override
    async def parse_tools(self, tools: Sequence[ToolDefinition]) -> list[Any]:
        return await self._api.parse_tools(self, tools)

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        response = await self.get_client().files.create(
            file=(name, bytes, mime),
            purpose="file-extract"  # type: ignore[reportArgumentType]
            if self.provider == "kimi" and type == "file"
            else "assistants",
        )

        return FileWithId(
            type=type,
            name=response.filename,
            mime=mime,
            file_id=response.id,
        )

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        supported_parameters = (
            ("temperature", "top_p", "top_k")
            if self._api is completions
            else ("temperature", "top_p")
        )
        sampling = self._sampling_from_config(supported_parameters)
        return await self._api.build_body(
            self,
            input,
            tools=tools,
            output_schema=output_schema,
            sampling=sampling,
            **kwargs,
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
        return await self._api.query(
            self,
            input,
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        return await self._api.get_rate_limit(self)

    @deprecated("Use query(output_schema=...) instead")
    @override
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> PydanticT:
        return await responses.query_json(
            self, input, pydantic_model, output_schema=output_schema, **kwargs
        )

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Query OpenAI's Embedding endpoint"""

        async def _get_embedding() -> list[float]:
            try:
                response: CreateEmbeddingResponse = (
                    await self.get_client().embeddings.create(
                        input=text,
                        model=model,
                    )
                )
            except APIConnectionError:
                raise ImmediateRetryException("Failed to connect to OpenAI")
            except Exception as e:
                raise Exception("Failed to query OpenAI's Embedding endpoint") from e

            if not response.data:
                raise Exception("No embeddings returned from OpenAI")

            return response.data[0].embedding

        return await BaseRetrier.immediate_retry_wrapper(
            func=_get_embedding, logger=self.instance_logger
        )

    async def transcribe_file_audio(
        self,
        *,
        name: str,
        mime: str,
        audio: bytes,
        language: str | None,
    ) -> TranscriptionResult:
        return await voice.transcribe_file_audio(
            self, name=name, mime=mime, audio=audio, language=language
        )

    @override
    async def _transcribe_audio(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return await voice.transcribe_audio(self, request)

    async def moderate_content(self, text: str) -> ModerationCreateResponse:
        """Query OpenAI's Moderation endpoint"""

        async def _moderate_content() -> ModerationCreateResponse:
            try:
                return await self.get_client().moderations.create(input=text)
            except APIConnectionError:
                raise ImmediateRetryException("Failed to connect to OpenAI")
            except Exception as e:
                raise Exception("Failed to query OpenAI's Moderation endpoint") from e

        return await BaseRetrier.immediate_retry_wrapper(
            func=_moderate_content, logger=self.instance_logger
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
        Per session:
        - Code Interpreter
        Per day:
        - File Search Storage
        Per 1000:
        - File Search Tool Call (responses api)
        - Web Search Tool Call
        """

        # Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it
        # cache tokens are included in input tokens

        # reasoning tokens are included in output tokens

        return await super()._calculate_cost(
            metadata, batch or self.service_tier == "flex", bill_reasoning=True
        )
