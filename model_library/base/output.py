"""
--- OUTPUT ---
"""

from pprint import pformat
from typing import Any, Mapping, Sequence, cast

from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import override

from model_library.base.input import InputItem, ToolCall
from model_library.base.utils import add_optional
from model_library.utils import truncate_str


class Citation(BaseModel):
    type: str | None = None
    title: str | None = None
    url: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    file_id: str | None = None
    filename: str | None = None
    index: int | None = None
    container_id: str | None = None


class QueryResultExtras(BaseModel):
    citations: list[Citation] = Field(default_factory=list)


class QueryResultCost(BaseModel):
    """
    Cost information for a query
    Includes total cost and a structured breakdown.
    """

    input: float
    output: float
    reasoning: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    total_override: float | None = None

    @computed_field
    @property
    def total(self) -> float:
        if self.total_override is not None:
            return self.total_override

        return sum(
            filter(
                None,
                [
                    self.input,
                    self.output,
                    self.reasoning,
                    self.cache_read,
                    self.cache_write,
                ],
            )
        )

    @computed_field
    @property
    def total_input(self) -> float:
        return sum(
            filter(
                None,
                [
                    self.input,
                    self.cache_read,
                    self.cache_write,
                ],
            )
        )

    @computed_field
    @property
    def total_output(self) -> float:
        return sum(
            filter(
                None,
                [
                    self.output,
                    self.reasoning,
                ],
            )
        )

    def __add__(self, other: "QueryResultCost") -> "QueryResultCost":
        return QueryResultCost(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=add_optional(self.reasoning, other.reasoning),
            cache_read=add_optional(self.cache_read, other.cache_read),
            cache_write=add_optional(self.cache_write, other.cache_write),
            total_override=add_optional(self.total_override, other.total_override),
        )

    @override
    def __repr__(self):
        use_cents = self.total < 1

        def format_cost(value: float | None):
            if value is None:
                return None
            return f"{value * 100:.3f} cents" if use_cents else f"${value:.2f}"

        return (
            f"{format_cost(self.total)} "
            + f"(uncached input: {format_cost(self.input)} | output: {format_cost(self.output)} | reasoning: {format_cost(self.reasoning)} | cache_read: {format_cost(self.cache_read)} | cache_write: {format_cost(self.cache_write)})"
        )


class QueryResultMetadata(BaseModel):
    """
    Metadata for a query: token usage and timing.

    """

    cost: QueryResultCost | None = None  # set post query
    duration_seconds: float | None = None  # set post query
    in_tokens: int = 0
    out_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    @property
    def default_duration_seconds(self) -> float:
        return self.duration_seconds or 0

    @computed_field
    @property
    def total_input_tokens(self) -> int:
        return sum(
            filter(
                None,
                [
                    self.in_tokens,
                    self.cache_read_tokens,
                    self.cache_write_tokens,
                ],
            )
        )

    @computed_field
    @property
    def total_output_tokens(self) -> int:
        return sum(
            filter(
                None,
                [
                    self.out_tokens,
                    self.reasoning_tokens,
                ],
            )
        )

    def __add__(self, other: "QueryResultMetadata") -> "QueryResultMetadata":
        return QueryResultMetadata(
            in_tokens=self.in_tokens + other.in_tokens,
            out_tokens=self.out_tokens + other.out_tokens,
            reasoning_tokens=cast(
                int | None, add_optional(self.reasoning_tokens, other.reasoning_tokens)
            ),
            cache_read_tokens=cast(
                int | None,
                add_optional(self.cache_read_tokens, other.cache_read_tokens),
            ),
            cache_write_tokens=cast(
                int | None,
                add_optional(self.cache_write_tokens, other.cache_write_tokens),
            ),
            duration_seconds=self.default_duration_seconds
            + other.default_duration_seconds,
            cost=cast(QueryResultCost | None, add_optional(self.cost, other.cost)),
        )

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2, sort_dicts=False)}\n)"


class QueryResult(BaseModel):
    """
    Result of a query
    Contains the text, reasoning, metadata, tool calls, and history
    """

    output_text: str | None = None
    reasoning: str | None = None
    metadata: QueryResultMetadata = Field(default_factory=QueryResultMetadata)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    history: list[InputItem] = Field(default_factory=list)
    extras: QueryResultExtras = Field(default_factory=QueryResultExtras)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def output_text_str(self) -> str:
        return self.output_text or ""

    @field_validator("reasoning", mode="before")
    def default_reasoning(cls, v: str | None):
        return None if not v else v  # make reasoning None if empty

    @property
    def search_results(self) -> Any | None:
        """Expose provider-supplied search metadata without additional processing."""
        raw_dict = cast(dict[str, Any], getattr(self, "raw", {}))
        raw_candidate = raw_dict.get("search_results")
        if raw_candidate is not None:
            return raw_candidate

        return _get_from_history(self.history, "search_results")

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        ordered_attrs = {
            "output_text": truncate_str(attrs.pop("output_text", None), 400),
            "reasoning": truncate_str(attrs.pop("reasoning", None), 400),
            "metadata": attrs.pop("metadata", None),
        }
        if self.tool_calls:
            ordered_attrs["tool_calls"] = self.tool_calls
        return f"{self.__class__.__name__}(\n{pformat(ordered_attrs, indent=2, sort_dicts=False)}\n)"


def _get_from_history(history: Sequence[InputItem], key: str) -> Any | None:
    for item in reversed(history):
        value = getattr(item, key, None)
        if value is not None:
            return value

        extra = getattr(item, "model_extra", None)
        if isinstance(extra, Mapping):
            value = cast(Mapping[str, Any], extra).get(key)
            if value is not None:
                return value

    return None
