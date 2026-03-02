import traceback
from typing import Any

from pydantic import computed_field, field_validator

from model_library.agent.tool import ToolOutput
from model_library.base.input import ToolCall
from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.utils import PrettyModel


class SerializableException(PrettyModel):
    """Serializable representation of an exception"""

    type: str
    message: str
    traceback: str | None = None
    context: dict[str, Any] = {}

    @classmethod
    def from_exception(cls, e: Exception, **context: Any) -> "SerializableException":
        tb = "".join(traceback.format_exception(e))
        return cls(
            type=type(e).__name__,
            message=str(e),
            traceback=tb,
            context=context,
        )


class ToolCallRecord(PrettyModel):
    """Record of a single tool call execution"""

    tool_call: ToolCall
    tool_output: ToolOutput
    duration_seconds: float  # rounded to ms
    error: SerializableException | None = None

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None


class ErrorTurn(PrettyModel):
    """Failed LLM query that was not recoverable by the retrier or before_query hook"""

    error: SerializableException


class AgentTurn(PrettyModel):
    """Successful LLM query + tool execution results"""

    query_result: QueryResult
    tool_call_records: list[ToolCallRecord] = []

    @computed_field
    @property
    def combined_metadata(self) -> QueryResultMetadata:
        """LLM query tokens/cost/duration + sub-LLM calls from tools

        Does not include tool execution time (see ToolCallRecord.duration_seconds)
        """
        result = self.query_result.metadata
        for tc in self.tool_call_records:
            if tc.tool_output.metadata:
                result = result + tc.tool_output.metadata
        return result
