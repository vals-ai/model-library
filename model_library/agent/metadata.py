import traceback
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, computed_field

from model_library.agent.tool import ToolOutput
from model_library.base.input import ToolCall
from model_library.base.output import FinishReasonInfo, QueryResult, QueryResultMetadata
from model_library.base.output.result import ProviderToolEvent
from model_library.utils import SecondsMetric, ValsModel


class SerializableException(ValsModel):
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


class ToolCallSummary(ValsModel):
    """Lean summary of a tool call — metadata preserved, content replaced with lengths."""

    tool_name: str
    tool_call_id: str
    args_lengths: dict[str, int]
    output_length: int
    success: bool
    done: bool
    error: SerializableException | None = None
    duration_seconds: SecondsMetric
    metadata: QueryResultMetadata | None = None


class ToolCallRecord(ValsModel):
    """Record of a single tool call execution"""

    tool_call: ToolCall
    tool_output: ToolOutput
    duration_seconds: SecondsMetric  # rounded to ms
    error: SerializableException | None = None

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


class ErrorTurn(ValsModel):
    """Failed LLM query that was not recoverable by the retrier or before_query hook

    - duration_seconds: wall clock for the turn (hooks + failed query including retries)
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: SerializableException
    duration_seconds: SecondsMetric


class CompactionSummary(ValsModel):
    """Record of a history compaction attempt."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_number: int | None = Field(
        default=None,
        description=(
            "Agent loop turn during which the compaction ran — i.e. the turn "
            "whose LLM query consumed the freshly-compacted history. Compaction "
            "itself does not increment the turn counter. None if the failure "
            "happened before the turn number was known."
        ),
    )
    input_token_estimate: int | None = Field(
        default=None,
        description=(
            "Estimated input-token size of the next turn's query that triggered "
            "this compaction. This is the value that crossed `threshold_tokens`."
        ),
    )
    threshold_tokens: int | None = Field(
        default=None,
        description=(
            "Compaction threshold (input tokens) active when this attempt fired. "
            "Captured per-summary so post-hoc analysis doesn't need the config."
        ),
    )
    summary: str | None = Field(
        default=None,
        description="Compaction summary text returned by the LLM (None on failure).",
    )
    artifacts_subdir: str | None = Field(
        default=None,
        description=(
            "Relative path under `AgentResult.output_dir` where the pre-compaction "
            "history was serialized (`previous_history.bin`). None on failure."
        ),
    )
    metadata: QueryResultMetadata | None = Field(
        default=None,
        description=(
            "Token usage / cost / duration of the compaction LLM call itself. "
            "Set whenever the LLM call returned a response, even if the result "
            "was rejected (e.g. empty summary)."
        ),
    )
    error: SerializableException | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None


class TurnSummary(ValsModel):
    """Lean summary of a turn — all metadata preserved, content replaced with lengths."""

    output_text_length: int
    reasoning_length: int
    finish_reason: FinishReasonInfo
    metadata: QueryResultMetadata
    tool_calls: list[ToolCallSummary] = []
    provider_tool_events: list[ProviderToolEvent] = []
    duration_seconds: SecondsMetric
    retry_overhead_seconds: SecondsMetric = 0.0

    @computed_field
    @property
    def effective_duration_seconds(self) -> float:
        return round(self.duration_seconds - self.retry_overhead_seconds, 3)


class AgentTurn(ValsModel):
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
    duration_seconds: SecondsMetric
    retry_overhead_seconds: SecondsMetric = 0.0

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
            provider_tool_events=self.query_result.provider_tool_events,
            duration_seconds=self.duration_seconds,
            retry_overhead_seconds=self.retry_overhead_seconds,
        )
