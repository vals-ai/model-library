import traceback
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, computed_field, field_validator

from model_library.agent.tool import ToolOutput
from model_library.base.input import ToolCall
from model_library.base.output import FinishReasonInfo, QueryResult, QueryResultMetadata
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


class ToolCallSummary(PrettyModel):
    """Lean summary of a tool call — metadata preserved, content replaced with lengths."""

    tool_name: str
    tool_call_id: str
    args_lengths: dict[str, int]
    output_length: int
    success: bool
    done: bool
    error: SerializableException | None = None
    duration_seconds: float
    metadata: QueryResultMetadata | None = None

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)


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

    def to_summary(self) -> ToolCallSummary:
        args = self.tool_call.parsed_args or {}
        return ToolCallSummary(
            tool_name=self.tool_call.name,
            tool_call_id=self.tool_call.id,
            args_lengths={k: len(str(v)) for k, v in args.items()},
            output_length=len(self.tool_output.output),
            success=self.success,
            done=self.tool_output.done,
            error=self.error,
            duration_seconds=self.duration_seconds,
            metadata=self.tool_output.metadata,
        )


class ErrorTurn(PrettyModel):
    """Failed LLM query that was not recoverable by the retrier or before_query hook

    - duration_seconds: wall clock for the turn (hooks + failed query including retries)
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: SerializableException
    duration_seconds: float

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)


class TurnSummary(PrettyModel):
    """Lean summary of a turn — all metadata preserved, content replaced with lengths."""

    output_text_length: int
    reasoning_length: int
    finish_reason: FinishReasonInfo
    metadata: QueryResultMetadata
    tool_calls: list[ToolCallSummary] = []
    duration_seconds: float
    retry_overhead_seconds: float = 0.0

    @field_validator("duration_seconds", "retry_overhead_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def effective_duration_seconds(self) -> float:
        return round(self.duration_seconds - self.retry_overhead_seconds, 3)


class AgentTurn(PrettyModel):
    """Successful LLM query + tool execution results (raw, full data)

    - duration_seconds: wall clock for the entire turn (hooks + query + retries + tool execution)
    - retry_overhead_seconds: portion of duration spent in retries/backoff, computed as
      query_wall_time - query_duration (from LLM metadata). Always >= 0.
    - query_result.metadata.duration_seconds: LLM query time only (excludes retries)
    - tool_call_records[i].duration_seconds: individual tool execution time
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
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

    def to_summary(self) -> TurnSummary:
        return TurnSummary(
            output_text_length=len(self.query_result.output_text or ""),
            reasoning_length=len(self.query_result.reasoning or ""),
            finish_reason=self.query_result.finish_reason,
            metadata=self.query_result.metadata,
            tool_calls=[r.to_summary() for r in self.tool_call_records],
            duration_seconds=self.duration_seconds,
            retry_overhead_seconds=self.retry_overhead_seconds,
        )
