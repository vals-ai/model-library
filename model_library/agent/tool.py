import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import computed_field

from model_library.base.input import ToolBody, ToolDefinition
from model_library.base.output import QueryResultMetadata
from model_library.utils import PrettyModel


class ToolOutput(PrettyModel):
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
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
