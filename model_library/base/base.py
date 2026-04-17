import hashlib
import io
import json
import logging
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Generator
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    TypeVar,
)

import tiktoken
from pydantic import BaseModel, SecretStr, model_serializer
from rich.pretty import pretty_repr
from tiktoken.core import Encoding
from typing_extensions import deprecated

import model_library.base.serialize as init_serialize_opts
from model_library.base.batch import (
    LLMBatchMixin,
)
from model_library.base.input import (
    FileInput,
    FileWithId,
    InputItem,
    SystemInput,
    TextInput,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output import (
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    RateLimit,
)
from model_library.base.utils import (
    get_pretty_input_types,
    serialize_for_tokenizing,
)
from model_library.retriers.backoff import ExponentialBackoffRetrier
from model_library.retriers.base import BaseRetrier, R, RetrierType, retry_decorator
from model_library.retriers.token import TokenRetrier
from model_library.utils import MAX_LOG_HISTORY, PrettyModel, truncate_str

_ = init_serialize_opts

PydanticT = TypeVar("PydanticT", bound=BaseModel)


class ProviderConfig(BaseModel):
    """Base class for provider-specific configs. Do not use directly."""

    @model_serializer(mode="plain")
    def serialize_actual(self):
        return self.__dict__


class TokenRetryParams(BaseModel):
    input_modifier: float
    output_modifier: float

    use_dynamic_estimate: bool = True

    limit: int
    limit_refresh_seconds: Literal[60] = 60


class LLMConfig(PrettyModel):
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    reasoning: bool = False
    reasoning_effort: str | bool | None = None
    compute_effort: str | int | None = None
    supports_images: bool = False
    supports_files: bool = False
    supports_videos: bool = False
    supports_batch: bool = False
    supports_temperature: bool = True
    supports_tools: bool = False
    supports_output_schema: bool = False
    native: bool = True
    provider_config: ProviderConfig | None = None
    registry_key: str | None = None
    custom_api_key: SecretStr | None = None
    custom_endpoint: str | None = None


# shared across all subclasses and instances
# hash(provider + api_key) -> client
client_registry_lock = threading.Lock()
client_registry: dict[tuple[str, str], Any] = {}


class LLM(ABC):
    """
    Base class for all LLMs
    LLM call errors should be raised as exceptions
    """

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

        self.max_tokens: int | None = config.max_tokens
        self.temperature: float | None = config.temperature
        self.top_p: float | None = config.top_p
        self.top_k: int | None = config.top_k

        self.reasoning: bool = config.reasoning
        self.reasoning_effort: str | bool | None = config.reasoning_effort
        self.compute_effort: str | int | None = config.compute_effort

        self.supports_files: bool = config.supports_files
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

        if config.provider_config:
            if isinstance(
                config.provider_config, type(getattr(self, "provider_config"))
            ):
                self.provider_config = config.provider_config

        self.instance_logger: logging.Logger = logging.getLogger("llm").getChild(
            f"{provider}.{model_name}"
        )
        self.custom_retrier: RetrierType | None = None

        self.token_retry_params = None
        # set _client_registry_key after initializing delegate
        if not self.native:
            return

        if config.custom_api_key:
            raw_key = config.custom_api_key.get_secret_value()
        else:
            raw_key = self._get_default_api_key()

        base_url = config.custom_endpoint

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
        attrs.pop("custom_retrier", None)
        attrs.pop("instance_logger", None)
        yield from attrs.items()

    def __repr__(self) -> str:
        return pretty_repr(self)

    __str__ = __repr__

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
        docent_ingest: bool = False,
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

        # represents a run
        run_id_provided = bool(run_id)

        if not run_id:
            run_id = uuid.uuid4().hex[:8]

        # represents a question (can have multiple queries if agentic)
        if not question_id:
            question_id = uuid.uuid4().hex[:14]

        # represents a query
        query_id = uuid.uuid4().hex[:14]

        _base_logger = logger or self.instance_logger.getChild(f"<run={run_id}>")

        # verbose on debug
        child_name = (
            f"<query={query_id}>"
            if in_agent
            else f"<question={question_id}><query={query_id}>"
        )
        query_logger = _base_logger.getChild(child_name)
        if in_agent:
            query_logger.setLevel(logging.WARNING)

        if docent_ingest and not run_id_provided:
            query_logger.warning("docent_ingest=True but no run_id provided")

        info_enabled = query_logger.isEnabledFor(logging.INFO)
        debug_enabled = query_logger.isEnabledFor(logging.DEBUG)

        # format str input
        if isinstance(input, str):
            input = [TextInput(text=input)]

        # back-compat: system_prompt kwarg is deprecated, use SystemInput as first input item
        system_prompt_kwarg = kwargs.pop("system_prompt", None)
        if system_prompt_kwarg is not None:
            input = [SystemInput(text=str(system_prompt_kwarg)), *input]

        item_info = ""
        tool_info = ""
        short_kwargs = ""
        if info_enabled:
            # format input info
            item_info = f"--- input ({len(input)}): {get_pretty_input_types(input, debug_enabled)}\n"
            if history:
                logged_history = (
                    history if debug_enabled else history[-MAX_LOG_HISTORY:]
                )
                item_info += f"--- history({len(history)}): {get_pretty_input_types(logged_history, debug_enabled)}\n"

            # format tool info
            tool_results = [t for t in input if isinstance(t, ToolResult)]
            tool_names = [tool.name for tool in tools or []]

            tool_info = (
                f"--- tools ({len(tools)}): {tool_names}\n"
                + f"--- tool results ({len(tool_results)}): "
                + f"{[{tool.tool_call.name: truncate_str(str(tool.result))} for tool in tool_results]}\n"
                if tools
                else ""
            )

            short_kwargs = {k: truncate_str(repr(v)) for k, v in kwargs.items()}

        # join input with history
        input = [*history, *input]

        # validate SystemInput: at most one, must be first
        system_inputs = [
            i for i, item in enumerate(input) if isinstance(item, SystemInput)
        ]
        if len(system_inputs) > 1:
            raise ValueError("At most one SystemInput is allowed per query")
        if system_inputs and system_inputs[0] != 0:
            raise ValueError("SystemInput must be the first item in the input sequence")

        if info_enabled:
            query_logger.info(
                "Query started:\n"
                + item_info
                + tool_info
                + f"--- kwargs: {short_kwargs}\n"
            )
            query_logger.debug([repr(item) for item in input])

        async def query_func() -> QueryResult:
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
            if self.token_retry_params:
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
                    use_dynamic_estimate=self.token_retry_params.use_dynamic_estimate,
                )
            else:
                retrier = ExponentialBackoffRetrier(logger=query_logger)
            return await retry_decorator(retrier)(immediate_retry)()

        run_with_retry = (
            default_retry
            if not self.custom_retrier
            else self.custom_retrier(immediate_retry)
        )

        output, duration = await run_with_retry()
        output.metadata.duration_seconds = duration
        output.metadata.cost = await self._calculate_cost(output.metadata)

        if output_schema is not None and output.output_text:
            if isinstance(output_schema, dict):
                output.output_parsed = json.loads(output.output_text)
            else:
                output.output_parsed = output_schema.model_validate_json(
                    output.output_text
                )

        if info_enabled:
            max_string = None if debug_enabled else 400
            query_logger.info(
                f"Query completed: {pretty_repr(output, max_string=max_string)}"
            )
        if debug_enabled:
            query_logger.debug(repr(output))

        # Skip ingestion when in_agent — the Agent handles its own ingestion
        # after the full loop so that all turns are in one Docent agent run.
        if docent_ingest and not in_agent and run_id_provided:
            try:
                from model_library.docent import (
                    ingest,
                    query_result_to_docent_agent_run,
                )

                ingest(
                    run_id,
                    query_result_to_docent_agent_run(
                        input,
                        output,
                        question_id,
                    ),
                )
            except Exception:
                query_logger.warning("Docent ingestion failed", exc_info=True)

        return output

    async def init_token_retry(self, token_retry_params: TokenRetryParams) -> None:
        self.token_retry_params = token_retry_params
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=self._client_registry_key_model_specific,
            limit=self.token_retry_params.limit,
            limit_refresh_seconds=self.token_retry_params.limit_refresh_seconds,
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
        from model_library.registry_utils import get_model_cost

        if not self._registry_key:
            self.instance_logger.warning(
                "Model has no registry key, skipping cost calculation"
            )
            return None

        costs = get_model_cost(self._registry_key)
        if not costs:
            return None

        MILLION = 1_000_000

        input_cost = costs.input
        output_cost = costs.output

        # apply fixed values or discounts/markup
        # applied before other price changes
        cache_read_cost, cache_write_cost = None, None
        if metadata.cache_read_tokens or metadata.cache_write_tokens:
            if not costs.cache:
                raise Exception("Cache costs not set")
            cache_read_cost, cache_write_cost = costs.cache.get_costs(input_cost)

        # costs for long context
        total_in = metadata.total_input_tokens
        if costs.context and total_in > costs.context.threshold:
            input_cost, output_cost = costs.context.get_costs(
                input_cost,
                output_cost,
                total_in,
            )
            if costs.context.cache:
                cache_read_cost, cache_write_cost = costs.context.cache.get_costs(
                    input_cost
                )

        # costs for batching
        if batch:
            if not costs.batch:
                raise Exception("Batch costs not set")
            input_cost, output_cost = costs.batch.get_costs(input_cost, output_cost)

        return QueryResultCost(
            input=input_cost * metadata.in_tokens / MILLION,
            output=output_cost * metadata.out_tokens / MILLION,
            reasoning=output_cost * metadata.reasoning_tokens / MILLION
            if metadata.reasoning_tokens is not None and bill_reasoning
            else None,
            cache_read=cache_read_cost * metadata.cache_read_tokens / MILLION
            if metadata.cache_read_tokens is not None and cache_read_cost
            else None,
            cache_write=cache_write_cost * metadata.cache_write_tokens / MILLION
            if metadata.cache_write_tokens is not None and cache_write_cost
            else None,
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
        assert self.token_retry_params

        # TODO: when passing in images and files, we really need to take that into account when calculating the output tokens!!

        input_tokens = (
            await self.count_tokens(input, history=[], tools=tools, **kwargs)
            * self.token_retry_params.input_modifier
        )

        output_tokens = input_tokens * self.token_retry_params.output_modifier
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
        input = [*history, *input]

        # back-compat: system_prompt kwarg is deprecated, use SystemInput as first input item
        system_prompt_kwarg = kwargs.pop("system_prompt", None)
        if system_prompt_kwarg is not None:
            input = [SystemInput(text=str(system_prompt_kwarg)), *input]

        system_prompt = ""
        if isinstance(input[0], SystemInput):
            system_prompt = input[0].text
            input = input[1:]

        # special case if using a delegate
        # don't inherit method override by default
        if self.delegate:
            parsed_input = await self.delegate.parse_input(input, **kwargs)
            parsed_tools = await self.delegate.parse_tools(tools)
        else:
            parsed_input = await self.parse_input(input, **kwargs)
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

        if not input and not history:
            return 0

        if self.delegate:
            encoding = await self.delegate.get_encoding()
        else:
            encoding = await self.get_encoding()
        self.instance_logger.debug(f"Token Count Encoding: {encoding}")

        string_input = await self.stringify_input(
            input, history=history, tools=tools, **kwargs
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
    def serialize_input(input: Sequence[InputItem]) -> bytes:
        return pickle.dumps(input)

    @staticmethod
    def deserialize_input(data: bytes | Path):
        """
        Deserialize input from bytes or a file path.

        WARNING: Uses pickle which can execute arbitrary code. Only deserialize
        data from trusted sources.

        Save if you serialize_input() -> nothing happens to that data -> deserialize_input()
        Unsafe if you serialize_input() -> send over a network for example -> send back

        If deserializing from untrusted sources, add HMAC verification.
        """
        if isinstance(data, Path):
            with open(data, "rb") as f:
                data = f.read()
        return pickle.loads(data)
