import io
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
    TypeVar,
)

from pydantic import model_serializer
from pydantic.main import BaseModel
from typing_extensions import override

from model_library.base.batch import (
    LLMBatchMixin,
)
from model_library.base.input import (
    FileInput,
    FileWithId,
    InputItem,
    TextInput,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output import (
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.base.utils import (
    get_pretty_input_types,
)
from model_library.exceptions import (
    ImmediateRetryException,
    retry_llm_call,
)
from model_library.utils import truncate_str

if TYPE_CHECKING:
    from model_library.providers.openai import OpenAIModel

PydanticT = TypeVar("PydanticT", bound=BaseModel)


class ProviderConfig(BaseModel):
    """Base class for provider-specific configs. Do not use directly."""

    @model_serializer(mode="plain")
    def serialize_actual(self):
        return self.__dict__


DEFAULT_MAX_TOKENS = 2048


class LLMConfig(BaseModel):
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    reasoning: bool = False
    reasoning_effort: str | None = None
    supports_images: bool = False
    supports_files: bool = False
    supports_videos: bool = False
    supports_batch: bool = False
    supports_temperature: bool = True
    supports_tools: bool = False
    native: bool = True
    provider_config: ProviderConfig | None = None
    registry_key: str | None = None


RetrierType = Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]

R = TypeVar("R")  # return type


class LLM(ABC):
    """
    Base class for all LLMs
    LLM call errors should be raised as exceptions
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig | None = None,
    ):
        self.instance_id = uuid.uuid4().hex[:8]

        self.provider: str = provider
        self.model_name: str = model_name

        config = config or LLMConfig()
        self._registry_key = config.registry_key

        self.max_tokens: int = config.max_tokens
        self.temperature: float | None = config.temperature
        self.top_p: float | None = config.top_p
        self.top_k: int | None = config.top_k

        self.reasoning: bool = config.reasoning
        self.reasoning_effort: str | None = config.reasoning_effort

        self.supports_files: bool = config.supports_files
        self.supports_videos: bool = config.supports_videos
        self.supports_images: bool = config.supports_images
        self.supports_batch: bool = config.supports_batch
        self.supports_temperature: bool = config.supports_temperature
        self.supports_tools: bool = config.supports_tools

        self.native: bool = config.native
        self.delegate: "OpenAIModel | None" = None
        self.batch: LLMBatchMixin | None = None

        if config.provider_config:
            if isinstance(
                config.provider_config, type(getattr(self, "provider_config"))
            ):
                self.provider_config = config.provider_config

        self.logger: logging.Logger = logging.getLogger(
            f"llm.{provider}.{model_name}<instance={self.instance_id}>"
        )
        self.custom_retrier: Callable[..., RetrierType] | None = retry_llm_call

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        attrs.pop("logger", None)
        attrs.pop("custom_retrier", None)
        attrs.pop("_key", None)
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2, sort_dicts=False)}\n)"

    @abstractmethod
    def get_client(self) -> object:
        """Return the instance of the appropriate SDK client."""
        ...

    @staticmethod
    async def timer_wrapper(func: Callable[[], Awaitable[R]]) -> tuple[R, float]:
        """
        Time the query
        """
        start = time.perf_counter()
        result = await func()
        return result, round(time.perf_counter() - start, 4)

    @staticmethod
    async def immediate_retry_wrapper(
        func: Callable[[], Awaitable[R]],
        logger: logging.Logger,
    ) -> R:
        """
        Retry the query immediately
        """
        MAX_IMMEDIATE_RETRIES = 10
        retries = 0
        while True:
            try:
                return await func()
            except ImmediateRetryException as e:
                if retries >= MAX_IMMEDIATE_RETRIES:
                    logger.error(f"Query reached max immediate retries {retries}: {e}")
                    raise Exception(
                        f"Query reached max immediate retries {retries}: {e}"
                    ) from e
                retries += 1

                logger.warning(
                    f"Query retried immediately {retries}/{MAX_IMMEDIATE_RETRIES}: {e}"
                )

    @staticmethod
    async def backoff_retry_wrapper(
        func: Callable[..., Awaitable[R]],
        backoff_retrier: RetrierType | None,
    ) -> R:
        """
        Retry the query with backoff
        """
        if not backoff_retrier:
            return await func()
        return await backoff_retrier(func)()

    async def delegate_query(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> QueryResult:
        if not self.delegate:
            raise Exception("Delegate not set")
        return await self.delegate._query_impl(input, tools=tools, **kwargs)  # pyright: ignore[reportPrivateUsage]

    async def query(
        self,
        input: Sequence[InputItem] | str,
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        # for backwards compatibility
        files: list[FileInput] = [],
        images: list[FileInput] = [],
        **kwargs: object,
    ) -> QueryResult:
        """
        Query the model
        Join input with history
        Log, Time, and Retry
        """

        # verbose on debug
        verbose = self.logger.isEnabledFor(logging.DEBUG)

        # format str input
        if isinstance(input, str):
            input = [TextInput(text=input)]

        # prepends files and images to input
        input = [*files, *images, *input]

        # format input info
        item_info = (
            f"--- input ({len(input)}): {get_pretty_input_types(input, verbose)}\n"
        )
        if history:
            item_info += f"--- history({len(history)}): {get_pretty_input_types(history, verbose)}\n"

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

        # unique logger for the query
        query_id = uuid.uuid4().hex[:14]
        query_logger = self.logger.getChild(f"query={query_id}")

        query_logger.info(
            "Query started:\n" + item_info + tool_info + f"--- kwargs: {short_kwargs}\n"
        )

        async def query_func() -> QueryResult:
            return await self._query_impl(input, tools=tools, **kwargs)

        async def timed_query() -> tuple[QueryResult, float]:
            return await LLM.timer_wrapper(query_func)

        async def immediate_retry() -> tuple[QueryResult, float]:
            return await LLM.immediate_retry_wrapper(timed_query, query_logger)

        async def backoff_retry() -> tuple[QueryResult, float]:
            backoff_retrier = (
                self.custom_retrier(query_logger) if self.custom_retrier else None
            )
            return await LLM.backoff_retry_wrapper(immediate_retry, backoff_retrier)

        output, duration = await backoff_retry()
        output.metadata.duration_seconds = duration
        output.metadata.cost = await self._calculate_cost(output.metadata)

        query_logger.info(f"Query completed: {repr(output)}")
        query_logger.debug(output.model_dump(exclude={"history", "raw"}))

        return output

    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """Calculate cost for a query"""
        from model_library.registry_utils import get_model_cost

        if not self._registry_key:
            self.logger.warning("Model has no registry key, skipping cost calculation")
            return None

        costs = get_model_cost(self._registry_key)
        if not costs:
            return None

        MILLION = 1_000_000

        # base input and output
        if costs.input is None or costs.output is None:
            raise Exception("Base costs not set")
        input_cost = costs.input
        output_cost = costs.output

        # apply fixed values or discounts/markup
        # applied before other price changes
        cache_read_cost, cache_write_cost = None, None
        if metadata.cache_read_tokens or metadata.cache_write_tokens:
            if not costs.cache:
                raise Exception("Cache costs not set")
            cache_read_cost, cache_write_cost = costs.cache.get_costs(
                input_cost, output_cost
            )

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
                    input_cost, output_cost
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
        **kwargs: object,  # TODO: pass in query logger
    ) -> QueryResult:
        """
        Query the model with input
        Input can consist on text, images, files, or model specific raw responses
        Optionally pass in tools
        Kwargs will be passed to the model call (apart from exceptions like system_prompt)
        Images and files should be preprocessed according to what the model supports:
            - base64
            - url
            - file_id
        """
        ...

    @abstractmethod
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
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

    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        **kwargs: object,
    ) -> PydanticT:
        """Query the model with JSON response format using Pydantic model.

        This is a convenience method that is not implemented for all providers.
        Only OpenAI and Google providers currently support this method.

        Args:
            input: Input items (text, files, etc.)
            pydantic_model: Pydantic model class defining the expected response structure
            **kwargs: Additional arguments passed to the query method

        Returns:
            Instance of the pydantic_model with the model's response

        Raises:
            NotImplementedError: If the provider does not support structured JSON output
        """
        raise NotImplementedError(
            f"query_json is not implemented for {self.__class__.__name__}. "
            f"Only OpenAI and Google providers currently support this method."
        )
