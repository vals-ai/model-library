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
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        required: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or list(parameters.keys())

    @abstractmethod
    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput: ...

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            body=ToolBody(
                name=self.name,
                description=self.description,
                properties=self.parameters,
                required=self.required,
            ),
        )
