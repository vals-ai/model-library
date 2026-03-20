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
    """Failed LLM query that was not recoverable by the retrier or before_query hook

    - duration_seconds: wall clock for the turn (hooks + failed query including retries)
    """

    error: SerializableException
    duration_seconds: float

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)


class AgentTurn(PrettyModel):
    """Successful LLM query + tool execution results

    - duration_seconds: wall clock for the entire turn (hooks + query + retries + tool execution)
    - retry_overhead_seconds: portion of duration spent in retries/backoff, computed as
      query_wall_time - query_duration (from LLM metadata). Always >= 0.
    - query_result.metadata.duration_seconds: LLM query time only (excludes retries)
    - tool_call_records[i].duration_seconds: individual tool execution time
    """

    query_result: QueryResult
    tool_call_records: list[ToolCallRecord] = []
    duration_seconds: float
    retry_overhead_seconds: float = 0.0

    @field_validator("duration_seconds", "retry_overhead_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def effective_duration_seconds(self) -> float:
        """Wall clock minus retry overhead (duration_seconds - retry_overhead_seconds)"""
        return round(self.duration_seconds - self.retry_overhead_seconds, 3)

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
