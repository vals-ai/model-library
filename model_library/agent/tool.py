import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

from pydantic import computed_field
from typing_extensions import override

from model_library.base.input import ToolBody, ToolDefinition
from model_library.base.output import QueryResultMetadata
from model_library.utils import ValsModel


class ToolOutput(ValsModel):
    """Result of a tool execution

    output: text sent back to the LLM as the tool result
    error: internal error tracking (set to mark the call as failed)
    metadata: token/cost metadata if the tool made its own LLM calls
    done: signal the agent to stop after this tool call
    """

    output: str
    error: str | None = None
    metadata: QueryResultMetadata | None = None
    done: bool = False

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None


class Tool(ABC):
    """Abstract base class for agent tools

    - The LLM only sees ToolOutput.output, even on error
    - Set error to mark the call as failed (for internal tracking)
    - Handle errors internally, don't raise from execute()
    - Unhandled exceptions are caught but sent to the LLM as raw strings

    Subclasses must define: name, description, parameters, and required (as class attributes or via __init__).
    If "required" is not specified, all parameters are assumed to be required.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    execution_type: Literal["local", "provider"] = "local"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        if getattr(cls, "execution_type", "local") == "provider":
            return
        for attr in ("name", "description", "parameters"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if not hasattr(cls, "required"):
            cls.required = list(cls.parameters.keys())

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        required: list[str] | None = None,
    ):
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if required is not None:
            self.required = required

    @abstractmethod
    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput: ...

    @property
    def definition(self) -> ToolDefinition:
        # verifies that required parameters are also parameters
        for required_param in self.required:
            if required_param not in self.parameters:
                raise ValueError(
                    f"Required parameter '{required_param}' not found in parameters"
                )

        return ToolDefinition(
            name=self.name,
            body=ToolBody(
                name=self.name,
                description=self.description,
                properties=self.parameters,
                required=self.required,
            ),
        )


class NativeWebSearch(Tool):
    """Provider-agnostic native web search tool.

    Each provider maps this to its native implementation via search_tool.
    Raises NotImplementedError for providers that don't support native search.
    """

    execution_type = "provider"
    name = "web_search"
    description = ""
    parameters: dict[str, Any] = {}
    required: list[str] = []

    @property
    @override
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, body=_NATIVE_WEB_SEARCH_SENTINEL)

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        raise RuntimeError("NativeWebSearch is executed by the provider, not the agent")


class NativeWebSearchSentinel(ValsModel):
    type: Literal["native_web_search"] = "native_web_search"


_NATIVE_WEB_SEARCH_SENTINEL = NativeWebSearchSentinel()


def is_native_web_search(body: Any) -> bool:
    """Return True if body represents a NativeWebSearch tool.

    Matches both the live instance (direct path) and the JSON sentinel
    (gateway path, where model_dump() has flattened the body to a dict).
    """
    return isinstance(body, (NativeWebSearch, NativeWebSearchSentinel)) or (
        isinstance(body, dict)
        and cast("dict[str, Any]", body).get("type") == "native_web_search"
    )


class ProviderTool(Tool):
    """A tool executed by the provider server-side, not the local agent.

    Pass a raw tool body (e.g. {"type": "web_search"}) and the provider
    handles execution internally. The agent records the event via
    provider_tool_events on QueryResult but never calls execute().
    """

    parameters: dict[str, Any] = {}
    required: list[str] = []
    execution_type = "provider"

    def __init__(self, name: str, body: Any, description: str = ""):
        self.name = name
        self.description = description
        self.body = body

    @property
    @override
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, body=self.body)

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        raise RuntimeError("ProviderTool is executed by the provider, not the agent")
