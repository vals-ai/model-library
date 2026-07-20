import base64
import hashlib
import hmac
import io
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Generator, Mapping
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    TYPE_CHECKING,
    TypedDict,
    TypeVar,
    cast,
)

import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_serializer,
)

from rich.pretty import pretty_repr
from tiktoken.core import Encoding
from typing_extensions import deprecated

import model_library.base.serialize as init_serialize_opts
import model_library.telemetry as telemetry
from model_library.base.batch import (
    LLMBatchMixin,
)
from model_library.base.input import (
    FileInput,
    FileWithId,
    InputItem,
    RawInput,
    RawResponse,
    SystemInput,
    ToolDefinition,
    normalize_query_input,
)
from model_library.base.output import (
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    RateLimit,
)
from model_library.base.query_ids import resolve_query_ids, scoped_query_ids
from model_library.base.query_logging import (
    log_query_completed,
    log_query_started,
    scoped_query_logger,
)
from model_library.base.utils import serialize_for_tokenizing
from model_library.exceptions import InvalidStructuredOutputError
from model_library.retriers.backoff import ExponentialBackoffRetrier
from model_library.retriers.base import BaseRetrier, R, RetrierType, retry_decorator
from model_library.utils import (
    ValsModel,
    round_to_milliseconds,
)

if TYPE_CHECKING:
    from model_library.register_models import ModelConfig

_ = init_serialize_opts

PydanticT = TypeVar("PydanticT", bound=BaseModel)


class ProviderConfig(BaseModel):
    """Base class for provider-specific configs. Do not use directly."""

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="plain")
    def serialize_actual(self):
        return self.__dict__


class TokenRetryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_modifier: float
    output_modifier: float
    use_dynamic_estimate: bool = True
    limit: int | None = Field(default=None, gt=0)
    limit_refresh_seconds: Literal[60] = 60


class ResolvedTokenRetryParams(BaseModel):
    input_modifier: float
    output_modifier: float
    use_dynamic_estimate: bool
    limit: int
    limit_refresh_seconds: Literal[60] = 60


def resolve_token_retry_params(
    token_retry_params: TokenRetryParams,
    effective_token_limit: int | None,
) -> ResolvedTokenRetryParams:
    if effective_token_limit is None:
        raise ValueError(
            "Token retry requires an explicit limit when no configured provider "
            "default is available"
        )

    return ResolvedTokenRetryParams(
        input_modifier=token_retry_params.input_modifier,
        output_modifier=token_retry_params.output_modifier,
        use_dynamic_estimate=token_retry_params.use_dynamic_estimate,
        limit=effective_token_limit,
        limit_refresh_seconds=token_retry_params.limit_refresh_seconds,
    )


class LLMConfig(ValsModel):
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    reasoning: bool = False
    reasoning_effort: str | bool | None = None
    compute_effort: str | int | None = None
    supports_images: bool = False
    supports_files: bool = False
    supports_audio: bool = False
    supports_videos: bool = False
    supports_batch: bool = False
    supports_temperature: bool = True
    supports_tools: bool = False
    supports_output_schema: bool = False
    native: bool = True
    provider_config: dict[str, Any] | ProviderConfig | None = None
    registry_key: str | None = None
    custom_api_key: SecretStr | None = None
    custom_endpoint: str | None = None


def dump_llm_config(config: LLMConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    data = config.model_dump(exclude_none=True, exclude_unset=True, mode="json")
    if config.custom_api_key is not None:
        data["custom_api_key"] = config.custom_api_key.get_secret_value()
    return data


def dump_gateway_config(
    config: LLMConfig | None,
    *,
    mode: Literal["json", "python"] = "json",
    exclude_none: bool = True,
    exclude_unset: bool = True,
) -> dict[str, Any]:
    if config is None:
        return {}

    return config.model_dump(
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        mode=mode,
    )


def normalize_llm_config_for_model(
    model: str,
    config: LLMConfig | Mapping[str, Any] | None,
    *,
    reject_unknown_fields: bool = False,
    require_custom_key_for_endpoint: bool = True,
) -> LLMConfig | None:
    if config is None:
        return None
    data = dump_llm_config(config) if isinstance(config, LLMConfig) else dict(config)
    if not data:
        return None

    unknown_fields = set(data) - set(LLMConfig.model_fields)
    if reject_unknown_fields and unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown LLM config field(s): {fields}")

    if (
        require_custom_key_for_endpoint
        and data.get("custom_endpoint")
        and not data.get("custom_api_key")
    ):
        raise ValueError("custom_endpoint requires custom_api_key")

    provider_config = data.get("provider_config")
    provider_name, separator, _model_name = model.partition("/")
    if isinstance(provider_config, dict) and separator:
        from model_library.registry_utils import get_provider_registry

        model_class = get_provider_registry().get(provider_name)
        if model_class is None:
            raise ValueError(
                f"Model {model} not found in registry: unknown provider {provider_name}"
            )
        provider_config_template = getattr(model_class, "provider_config", None)
        if isinstance(provider_config_template, BaseModel):
            provider_config_class: type[BaseModel] = provider_config_template.__class__
            data["provider_config"] = provider_config_class.model_validate(
                provider_config
            )

    return LLMConfig(**data)


# shared across all subclasses and instances
# hash(provider + api_key) -> client
client_registry_lock = threading.Lock()
client_registry: dict[tuple[str, str], Any] = {}


class SignedPickle(TypedDict):
    """HMAC-signed pickle blob for non-JSON-serializable fields."""

    pickle: str
    hmac: str


class LLM(ABC):
    """
    Base class for all LLMs
    LLM call errors should be raised as exceptions
    """

    gateway_mode: bool = False

    @property
    def _client_registry_key(self) -> tuple[str, str]:
        if self._own_registry_key is not None:
            return self._own_registry_key
        if self.delegate is not None:
            return self.delegate._client_registry_key
        raise AttributeError(
            f"'{type(self).__name__}' has no _client_registry_key "
            "(non-native model without delegate)"
        )

    @_client_registry_key.setter
    def _client_registry_key(self, value: tuple[str, str]) -> None:
        self._own_registry_key = value

    @property
    def _client_registry_key_model_specific(self) -> tuple[str, str]:
        if self._own_registry_key_model_specific is not None:
            return self._own_registry_key_model_specific
        if self.delegate is not None:
            return self.delegate._client_registry_key_model_specific
        raise AttributeError(
            f"'{type(self).__name__}' has no _client_registry_key_model_specific "
            "(non-native model without delegate)"
        )

    @_client_registry_key_model_specific.setter
    def _client_registry_key_model_specific(self, value: tuple[str, str]) -> None:
        self._own_registry_key_model_specific = value

    @abstractmethod
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Any:
        """
        Returns the cached instance of the appropriate SDK client.
        Sublasses should implement this method and:
        - if api_key is provided, initialize their client and call assing_client(client).
        - else return super().get_client()
        """
        global client_registry
        return client_registry[self._client_registry_key]

    def assign_client(self, client: object) -> None:
        """Thread-safe assignment to the client registry"""
        global client_registry

        if self._client_registry_key not in client_registry:
            with client_registry_lock:
                if self._client_registry_key not in client_registry:
                    client_registry[self._client_registry_key] = client

    def has_client(self) -> bool:
        return self._client_registry_key in client_registry

    @abstractmethod
    def _get_default_api_key(self) -> str:
        """Return the api key from model_library.settings"""
        ...

    def _client_initialization(
        self, config: LLMConfig
    ) -> tuple[str, str | None] | None:
        """Return API key and base URL for client initialization."""
        if not config.native:
            return None

        raw_key = (
            config.custom_api_key.get_secret_value()
            if config.custom_api_key
            else self._get_default_api_key()
        )
        return raw_key, config.custom_endpoint

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig | None = None,
    ):
        self.provider: str = provider
        self.model_name: str = model_name

        config = config or LLMConfig()
        self._registry_key = config.registry_key
        self._metadata: ModelConfig | None = None

        self.max_tokens: int | None = config.max_tokens
        self.temperature: float | None = config.temperature
        self.top_p: float | None = config.top_p
        self.top_k: int | None = config.top_k

        self.reasoning: bool = config.reasoning
        self.reasoning_effort: str | bool | None = config.reasoning_effort
        self.compute_effort: str | int | None = config.compute_effort

        self.supports_files: bool = config.supports_files
        self.supports_audio: bool = config.supports_audio
        self.supports_videos: bool = config.supports_videos
        self.supports_images: bool = config.supports_images
        self.supports_batch: bool = config.supports_batch
        self.supports_temperature: bool = config.supports_temperature
        self.supports_tools: bool = config.supports_tools
        self.supports_output_schema: bool = config.supports_output_schema

        self.native: bool = config.native
        self.delegate: "LLM | None" = None
        self.batch: LLMBatchMixin | None = None
        self.custom_endpoint = config.custom_endpoint

        current_provider_config = getattr(self, "provider_config", None)
        if config.provider_config and isinstance(
            config.provider_config, type(current_provider_config)
        ):
            self.provider_config = config.provider_config

        self.instance_logger: logging.Logger = logging.getLogger("llm").getChild(
            f"{provider}.{model_name}"
        )
        self.custom_retrier: RetrierType | None = None

        self.token_retry_params: TokenRetryParams | None = None
        self._resolved_token_retry_params: ResolvedTokenRetryParams | None = None
        self._own_registry_key: tuple[str, str] | None = None
        self._own_registry_key_model_specific: tuple[str, str] | None = None
        # set _client_registry_key after initializing delegate
        client_initialization = self._client_initialization(config)
        if client_initialization is None:
            return

        raw_key, base_url = client_initialization
        hash_material = raw_key if base_url is None else raw_key + base_url
        key_hash = hashlib.sha256(hash_material.encode()).hexdigest()
        self._client_registry_key = (self.provider, key_hash)
        self._client_registry_key_model_specific = (
            f"{self.provider}.{self.model_name}",
            key_hash,
        )
        self.get_client(api_key=raw_key, base_url=base_url)

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        attrs = vars(self).copy()
        attrs.pop("_metadata", None)
        attrs.pop("custom_retrier", None)
        attrs.pop("instance_logger", None)
        yield from attrs.items()

    def __repr__(self) -> str:
        return pretty_repr(self)

    __str__ = __repr__

    @property
    def metadata(self) -> "ModelConfig | None":
        return self._metadata

    @property
    def input_context_window(self) -> int | None:
        if self.metadata is None:
            return None

        from model_library.registry_utils import get_input_context_window_from_config

        return get_input_context_window_from_config(self.metadata)

    @staticmethod
    async def timer_wrapper(func: Callable[[], Awaitable[R]]) -> tuple[R, float]:
        """
        Time the query
        """
        start = time.perf_counter()
        result = await func()
        return result, time.perf_counter() - start

    async def delegate_query(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition] = [],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if not self.delegate:
            raise Exception("Delegate not set")
        return await self.delegate._query_impl(  # pyright: ignore[reportPrivateUsage]
            input,
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    async def query(
        self,
        input: Sequence[InputItem] | str,
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        logger: logging.Logger | None = None,
        run_id: str | None = None,
        question_id: str | None = None,
        in_agent: bool = False,
        **kwargs: object,
    ) -> QueryResult:
        """
        Query the model
        Join input with history
        Log, Time, and Retry
        """

        if output_schema is not None and not self.supports_output_schema:
            model_name = self._registry_key or f"{self.provider}/{self.model_name}"
            raise Exception(f"{model_name} does not support structured outputs")

        run_id, question_id, query_id = resolve_query_ids(
            run_id=run_id,
            question_id=question_id,
            query_id=kwargs.pop("query_id", None),
        )
        telemetry.set_attributes(
            {
                "run_id": run_id,
                "question_id": question_id,
                "query_id": query_id,
                "model.provider": self.provider,
                "model.name": self.model_name,
                "model.registry_key": self._registry_key,
                "llm.in_agent": in_agent,
                "llm.in_agent.mode": telemetry.mode_attribute(in_agent),
                "llm.tool.count": len(tools),
                "llm.output_schema.mode": telemetry.mode_attribute(
                    output_schema is not None
                ),
                "retry_queue.mode": telemetry.mode_attribute(
                    self.token_retry_params is not None
                ),
            }
        )
        telemetry.add_event(
            "model_library.query.start",
            {
                "run_id": run_id,
                "question_id": question_id,
                "query_id": query_id,
            },
        )

        _base_logger = logger or self.instance_logger.getChild(f"<run={run_id}>")

        query_logger = scoped_query_logger(
            _base_logger,
            question_id=question_id,
            query_id=query_id,
            in_agent=in_agent,
        )
        info_enabled = query_logger.isEnabledFor(logging.INFO)
        debug_enabled = query_logger.isEnabledFor(logging.DEBUG)

        current_input = normalize_query_input(input, kwargs=kwargs)

        # join input with history and validate SystemInput placement
        input = normalize_query_input(current_input, history=history)

        log_query_started(
            query_logger,
            input=current_input,
            all_input=input,
            history=history,
            tools=tools,
            kwargs=kwargs,
            info_enabled=info_enabled,
            debug_enabled=debug_enabled,
        )

        async def query_func() -> QueryResult:
            with telemetry.start_span(
                "model_library.provider_query",
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "query_id": query_id,
                    "model.provider": self.provider,
                    "model.name": self.model_name,
                    "model.registry_key": self._registry_key,
                    "llm.in_agent": in_agent,
                    "llm.in_agent.mode": telemetry.mode_attribute(in_agent),
                    "llm.output_schema.mode": telemetry.mode_attribute(
                        output_schema is not None
                    ),
                    "retry_queue.mode": telemetry.mode_attribute(
                        self.token_retry_params is not None
                    ),
                },
                kind="client",
            ):
                with scoped_query_ids(
                    run_id=run_id,
                    question_id=question_id,
                    query_id=query_id,
                ):
                    return await self._query_impl(
                        input,
                        tools=tools,
                        query_logger=query_logger,
                        output_schema=output_schema,
                        **kwargs,
                    )

        async def timed_query() -> tuple[QueryResult, float]:
            return await LLM.timer_wrapper(query_func)

        async def immediate_retry() -> tuple[QueryResult, float]:
            return await BaseRetrier.immediate_retry_wrapper(timed_query, query_logger)

        async def default_retry() -> tuple[QueryResult, float]:
            if self._resolved_token_retry_params:
                from model_library.retriers.token.token import TokenRetrier

                (
                    estimate_input_tokens,
                    estimate_output_tokens,
                ) = await self.estimate_query_tokens(
                    input,
                    tools=tools,
                    **kwargs,
                )
                retrier = TokenRetrier(
                    logger=query_logger,
                    client_registry_key=self._client_registry_key_model_specific,
                    run_id=run_id,
                    question_id=question_id,
                    estimate_input_tokens=estimate_input_tokens,
                    estimate_output_tokens=estimate_output_tokens,
                    use_dynamic_estimate=self._resolved_token_retry_params.use_dynamic_estimate,
                )
            else:
                retrier = ExponentialBackoffRetrier(logger=query_logger)
            return await retry_decorator(retrier)(immediate_retry)()

        run_with_retry = (
            default_retry
            if not self.custom_retrier
            else self.custom_retrier(immediate_retry)
        )

        telemetry.add_event(
            "model_library.query.retry_wrapper_start",
            {
                "run_id": run_id,
                "question_id": question_id,
                "query_id": query_id,
                "retry_queue.mode": telemetry.mode_attribute(
                    self.token_retry_params is not None
                ),
            },
        )
        output, duration = await run_with_retry()
        telemetry.add_event(
            "model_library.query.retry_wrapper_done",
            {
                "run_id": run_id,
                "question_id": question_id,
                "query_id": query_id,
                "llm.duration_seconds": duration,
            },
        )
        output.metadata.duration_seconds = round_to_milliseconds(duration)
        output.metadata.cost = await self._calculate_cost(output.metadata)

        if output_schema is not None and output.output_text:
            parser_error_type: str | None = None
            try:
                if isinstance(output_schema, dict):
                    output.output_parsed = json.loads(output.output_text)
                else:
                    output.output_parsed = output_schema.model_validate_json(
                        output.output_text
                    )
            except (json.JSONDecodeError, ValidationError) as exc:
                parser_error_type = type(exc).__name__
            if parser_error_type is not None:
                raise InvalidStructuredOutputError(parser_error_type=parser_error_type)

        log_query_completed(
            query_logger,
            output,
            info_enabled=info_enabled,
            debug_enabled=debug_enabled,
        )

        telemetry.add_event(
            "model_library.query.done",
            {
                "run_id": run_id,
                "question_id": question_id,
                "query_id": query_id,
                "gen_ai.usage.input_tokens": output.metadata.total_input_tokens,
                "gen_ai.usage.output_tokens": output.metadata.total_output_tokens,
                "llm.duration_seconds": duration,
            },
        )
        return output

    async def init_token_retry(self, token_retry_params: TokenRetryParams) -> None:
        effective_token_limit = token_retry_params.limit
        await self._init_resolved_token_retry(
            token_retry_params,
            resolve_token_retry_params(
                token_retry_params,
                effective_token_limit,
            ),
        )

    async def ensure_resolved_token_retry(
        self,
        token_retry_params: TokenRetryParams,
        resolved_token_retry_params: ResolvedTokenRetryParams,
    ) -> None:
        if self._resolved_token_retry_params != resolved_token_retry_params:
            await self._init_resolved_token_retry(
                token_retry_params,
                resolved_token_retry_params,
            )

    async def _init_resolved_token_retry(
        self,
        token_retry_params: TokenRetryParams,
        resolved_token_retry_params: ResolvedTokenRetryParams,
    ) -> None:
        from model_library.retriers.token.token import TokenRetrier

        self.token_retry_params = token_retry_params
        self._resolved_token_retry_params = resolved_token_retry_params
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=self._client_registry_key_model_specific,
            limit=resolved_token_retry_params.limit,
            limit_refresh_seconds=resolved_token_retry_params.limit_refresh_seconds,
            get_rate_limit_func=self.get_rate_limit,
            logger=self.instance_logger,
        )

    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """Calculate cost for a query"""
        from model_library.registry_utils import compute_model_cost

        if not self._registry_key:
            self.instance_logger.warning(
                "Model has no registry key, skipping cost calculation"
            )
            return None

        return compute_model_cost(
            self._registry_key,
            metadata,
            batch=batch,
            bill_reasoning=bill_reasoning,
        )

    @abstractmethod
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        """
        Query the model with input
        Input can consist on text, images, files, or model specific raw responses
        Optionally pass in tools
        Kwargs will be passed to the model call (apart from exceptions like system_prompt)
        Note: system_prompt kwarg is deprecated -- pass SystemInput as the first input item instead
        Images and files should be preprocessed according to what the model supports:
            - base64
            - url
            - file_id
        """
        ...

    @abstractmethod
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """
        Builds the body of the request to the model provider
        Calls parse_input
        """
        ...

    @abstractmethod
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> Any:
        """
        Parses input into the appropriate format for the model
        Handles prompts, images, and files
        Handles history and tool call results
        Calls
            - parse_image
            - parse_file
        """
        ...

    @abstractmethod
    async def parse_image(self, image: FileInput) -> Any:
        """Parse an image into the appropriate format for the model"""
        ...

    @abstractmethod
    async def parse_file(self, file: FileInput) -> Any:
        """Parse a file into the appropriate format for the model"""
        ...

    @property
    def search_tool(self) -> Any:
        raise NotImplementedError(
            f"Native web search is not supported by {type(self).__name__}"
        )

    @abstractmethod
    async def parse_tools(self, tools: list[ToolDefinition]) -> Any:
        """Parse tools into the appropriate format for the model"""
        ...

    @abstractmethod
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        """Upload a file to the model provider"""
        ...

    async def get_rate_limit(self) -> RateLimit | None:
        """Get the rate limit for the model provider"""
        return None

    async def estimate_query_tokens(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> tuple[int, int]:
        """Pessimistically estimate the number of tokens required for a query"""
        assert self._resolved_token_retry_params

        # TODO: when passing in images and files, we really need to take that into account when calculating the output tokens!!

        input_tokens = (
            await self.count_tokens(input, history=[], tools=tools, **kwargs)
            * self._resolved_token_retry_params.input_modifier
        )

        output_tokens = input_tokens * self._resolved_token_retry_params.output_modifier
        return ceil(input_tokens), ceil(output_tokens)

    async def get_encoding(self) -> Encoding:
        """Get the appropriate tokenizer"""

        model = self.model_name.lower()

        if any(x in model for x in ["gpt-4o", "o1", "o3", "gpt-4.1", "gpt-5"]):
            return tiktoken.get_encoding("o200k_base")
        elif "gpt-4" in model or "gpt-3.5" in model:
            try:
                return tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")
        elif "claude" in model:
            return tiktoken.get_encoding("cl100k_base")
        elif "gemini" in model:
            return tiktoken.get_encoding("o200k_base")
        elif "llama" in model or "mistral" in model:
            return tiktoken.get_encoding("cl100k_base")
        else:
            return tiktoken.get_encoding("cl100k_base")

    async def stringify_input(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> str:
        input = normalize_query_input(input, history=history, kwargs=kwargs)

        system_prompt = ""
        if input and isinstance(input[0], SystemInput):
            system_prompt = input[0].text
            input = input[1:]

        # special case if using a delegate
        # don't inherit method override by default
        parsed_input: object
        if self.delegate:
            parsed_input = (
                await self.delegate.parse_input(input, **kwargs) if input else []
            )
            parsed_tools = await self.delegate.parse_tools(tools)
        else:
            parsed_input = await self.parse_input(input, **kwargs) if input else []
            parsed_tools = await self.parse_tools(tools)

        serialized_input = serialize_for_tokenizing(parsed_input)
        serialized_tools = serialize_for_tokenizing(parsed_tools)

        combined = f"{system_prompt}\n{serialized_input}\n{serialized_tools}"

        return combined

    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        """
        Count the number of tokens for a query.
        Combines parsed input and tools, then tokenizes the result.
        """

        input = normalize_query_input(input, history=history, kwargs=kwargs)
        if not input and not tools:
            return 0

        if self.delegate:
            encoding = await self.delegate.get_encoding()
        else:
            encoding = await self.get_encoding()
        self.instance_logger.debug(f"Token Count Encoding: {encoding}")

        string_input = await self.stringify_input(
            input, history=[], tools=tools, **kwargs
        )

        count = len(encoding.encode(string_input, disallowed_special=()))
        self.instance_logger.debug(f"Combined Token Count Input: {count}")
        return count

    @deprecated("Use query(output_schema=...) instead")
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        **kwargs: object,
    ) -> PydanticT:
        """Query the model with JSON response format using Pydantic model.

        Deprecated: Use query(output_schema=...) instead.
        """
        raise NotImplementedError(
            f"query_json is not implemented for {self.__class__.__name__}. "
            f"Only OpenAI and Google providers currently support this method."
        )

    @staticmethod
    def serialize_input(
        input: Sequence[InputItem], *, secret: bytes | None = None
    ) -> str:
        """Serialize input items to a JSON string.

        RawResponse.response and RawInput.input hold arbitrary provider SDK
        objects that aren't JSON-serializable.  These fields are pickled and
        base64-encoded inline.  When *secret* is provided, each pickled blob
        gets an HMAC-SHA256 tag so the receiver can verify it wasn't tampered
        with before unpickling.
        """
        items: list[dict[str, Any]] = []
        for item in input:
            if isinstance(item, RawResponse):
                d = {
                    "kind": "raw_response",
                    "response": LLM._pickle_field(item.response, secret),
                }
            elif isinstance(item, RawInput):
                d = {
                    "kind": "raw_input",
                    "input": LLM._pickle_field(item.input, secret),
                }
            else:
                d = item.model_dump()
            items.append(d)
        return json.dumps(items, default=str)

    @staticmethod
    def deserialize_input(
        data: str | bytes | Path, *, secret: bytes | None = None
    ) -> list[InputItem]:
        """Deserialize input from a JSON string, bytes, or file path.

        Restores pickled RawResponse.response and RawInput.input fields.
        When *secret* is provided, HMAC tags on pickled blobs are verified
        before unpickling — rejects tampered data.

        WARNING: Uses pickle internally, which can execute arbitrary code.
        Without a secret, data is deserialized without verification — only
        use this on trusted data (e.g. local files you serialized yourself).
        """
        if isinstance(data, Path):
            data = data.read_text()
        elif isinstance(data, bytes):
            data = data.decode()

        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[InputItem])
        items = adapter.validate_json(data)

        LLM.restore_raw_fields(items, secret=secret)
        return items

    @staticmethod
    def restore_raw_fields(
        items: list[InputItem], *, secret: bytes | None = None
    ) -> None:
        """Unpickle RawResponse.response and RawInput.input fields in-place."""
        for item in items:
            if isinstance(item, RawResponse) and isinstance(item.response, (str, dict)):
                item.response = LLM._unpickle_field(
                    cast("str | SignedPickle", item.response), secret
                )
            elif isinstance(item, RawInput) and isinstance(item.input, (str, dict)):
                item.input = LLM._unpickle_field(
                    cast("str | SignedPickle", item.input), secret
                )

    @staticmethod
    def _pickle_field(value: Any, secret: bytes | None) -> str | SignedPickle:
        """Pickle a value to base64. If secret is given, attach an HMAC tag."""
        pickled = base64.b64encode(pickle.dumps(value)).decode()
        if secret:
            tag = hmac.new(secret, pickled.encode(), hashlib.sha256).hexdigest()
            return SignedPickle(pickle=pickled, hmac=tag)
        return pickled

    @staticmethod
    def _unpickle_field(value: str | SignedPickle, secret: bytes | None) -> Any:
        """Unpickle a base64 value. If secret is given, verify the HMAC tag first."""
        if isinstance(value, dict):
            pickled = value["pickle"]
            if secret:
                expected = hmac.new(
                    secret, pickled.encode(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(value.get("hmac", ""), expected):
                    raise ValueError("HMAC verification failed on pickled field")
            return pickle.loads(base64.b64decode(pickled))
        # Plain base64 string — no HMAC
        if secret:
            raise ValueError("Expected HMAC-signed pickle blob but got unsigned string")
        return pickle.loads(base64.b64decode(value))
