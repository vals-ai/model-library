"""Rate-limit models and provider-header decoding."""

import datetime
import time
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple, cast

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from model_library.utils import ValsModel

__all__ = ["RateLimit", "RateLimitScope"]


RateLimitScope = Literal["api_key", "shared"]


class _HeaderPair(NamedTuple):
    limit: str
    remaining: str


# Keep these families exact: providers reuse similar names for different
# capacities, and pairing fallback names would corrupt their semantics.
_REQUEST_HEADER_FAMILIES = (
    _HeaderPair("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"),
    _HeaderPair("x-ratelimit-limit-req-minute", "x-ratelimit-remaining-req-minute"),
    _HeaderPair("x-ratelimit-limit", "x-ratelimit-remaining"),
    _HeaderPair("x-ratelimit-limit-dynamic", "x-ratelimit-remaining-dynamic"),
)
_TOKEN_HEADER_FAMILIES = (
    _HeaderPair("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"),
    _HeaderPair(
        "x-ratelimit-limit-tokens-minute", "x-ratelimit-remaining-tokens-minute"
    ),
    _HeaderPair("x-tokenlimit-limit", "x-tokenlimit-remaining"),
    _HeaderPair("x-tokenlimit-limit-dynamic", "x-tokenlimit-remaining-dynamic"),
)
_INPUT_TOKEN_HEADER_FAMILIES = (
    _HeaderPair(
        "x-ratelimit-limit-tokens-prompt", "x-ratelimit-remaining-tokens-prompt"
    ),
)
_UNCACHED_INPUT_TOKEN_HEADER_FAMILIES = (
    _HeaderPair(
        "x-ratelimit-limit-tokens-cache-adjusted-prompt",
        "x-ratelimit-remaining-tokens-cache-adjusted-prompt",
    ),
)
_OUTPUT_TOKEN_HEADER_FAMILIES = (
    _HeaderPair(
        "x-ratelimit-limit-tokens-generated", "x-ratelimit-remaining-tokens-generated"
    ),
)


class RateLimitCapacity(ValsModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    limit: int = Field(ge=0)
    remaining: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_remaining(self) -> Self:
        if self.remaining is not None and self.remaining > self.limit:
            raise ValueError("remaining cannot exceed limit")
        return self


class RequestRateLimit(RateLimitCapacity):
    mode: Literal["sliding_window", "concurrency"] = "sliding_window"


class TokenRateLimit(ValsModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mode: Literal["token_bucket"] = "token_bucket"
    total: RateLimitCapacity | None = None
    input: RateLimitCapacity | None = None
    uncached_input: RateLimitCapacity | None = None
    output: RateLimitCapacity | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.total is not None:
            if any(
                value is not None
                for value in (self.input, self.uncached_input, self.output)
            ):
                raise ValueError(
                    "total cannot be combined with directional token limits"
                )
            return self
        if self.input is None or self.output is None:
            raise ValueError("directional token limits require both input and output")
        return self

    @property
    def limit_total(self) -> int:
        if self.total is not None:
            return self.total.limit
        assert self.input is not None and self.output is not None
        return self.input.limit + self.output.limit

    @property
    def remaining_total(self) -> int | None:
        if self.total is not None:
            return self.total.remaining
        assert self.input is not None and self.output is not None
        if self.input.remaining is None or self.output.remaining is None:
            return None
        return self.input.remaining + self.output.remaining


class RateLimit(ValsModel):
    """Provider-reported request and token capacities."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    requests: tuple[RequestRateLimit, ...] = ()
    tokens: TokenRateLimit | None = None

    @field_validator("requests", mode="before")
    @classmethod
    def deserialize_requests(cls, requests: object) -> object:
        return (
            tuple(cast(list[object], requests))
            if isinstance(requests, list)
            else requests
        )

    scope: RateLimitScope | None = None
    unix_timestamp: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if not self.requests and self.tokens is None:
            raise ValueError("at least one rate limit is required")
        modes = [request.mode for request in self.requests]
        if len(modes) != len(set(modes)):
            raise ValueError("only one request limit per mode is allowed")
        return self

    @property
    def token_limit_total(self) -> int | None:
        return self.tokens.limit_total if self.tokens is not None else None

    @property
    def token_remaining_total(self) -> int | None:
        return self.tokens.remaining_total if self.tokens is not None else None


def rate_limit_header_int(headers: Mapping[str, str], *names: str) -> int | None:
    """Read the first header present; absence is not zero capacity."""
    for name in names:
        value = headers.get(name)
        if value is not None:
            parsed = int(value)
            if parsed < 0:
                raise ValueError("Rate-limit header value cannot be negative")
            return parsed
    return None


def rate_limit_timestamp_from_headers(headers: Mapping[str, str]) -> float:
    """Use provider server time when available, otherwise local receipt time."""
    server_time = headers.get("date")
    if server_time is None:
        return time.time()
    return (
        datetime.datetime.strptime(server_time, "%a, %d %b %Y %H:%M:%S GMT")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def _first_complete_family(
    headers: Mapping[str, str], families: Sequence[_HeaderPair]
) -> tuple[int | None, int | None]:
    for family in families:
        limit = rate_limit_header_int(headers, family.limit)
        if limit is not None:
            return limit, rate_limit_header_int(headers, family.remaining)
    return None, None


def rate_limit_from_headers(
    headers: Mapping[str, str], *, default_scope: RateLimitScope | None = None
) -> RateLimit | None:
    """Read fixed rate-limit capacity from OpenAI-compatible headers."""
    request_limit, request_remaining = _first_complete_family(
        headers, _REQUEST_HEADER_FAMILIES
    )
    total_limit, total_remaining = _first_complete_family(
        headers, _TOKEN_HEADER_FAMILIES
    )
    input_limit, input_remaining = _first_complete_family(
        headers, _INPUT_TOKEN_HEADER_FAMILIES
    )
    uncached_input_limit, uncached_input_remaining = _first_complete_family(
        headers, _UNCACHED_INPUT_TOKEN_HEADER_FAMILIES
    )
    output_limit, output_remaining = _first_complete_family(
        headers, _OUTPUT_TOKEN_HEADER_FAMILIES
    )

    requests = (
        ()
        if request_limit is None
        else (RequestRateLimit(limit=request_limit, remaining=request_remaining),)
    )
    if total_limit is not None and input_limit is not None and output_limit is not None:
        total_limit = total_remaining = None
    tokens = None
    if total_limit is not None:
        tokens = TokenRateLimit(
            total=RateLimitCapacity(limit=total_limit, remaining=total_remaining)
        )
    elif input_limit is not None and output_limit is not None:
        tokens = TokenRateLimit(
            input=RateLimitCapacity(limit=input_limit, remaining=input_remaining),
            uncached_input=(
                RateLimitCapacity(
                    limit=uncached_input_limit,
                    remaining=uncached_input_remaining,
                )
                if uncached_input_limit is not None
                else None
            ),
            output=RateLimitCapacity(limit=output_limit, remaining=output_remaining),
        )

    if not requests and tokens is None:
        return None
    return RateLimit(
        requests=requests,
        tokens=tokens,
        scope=default_scope,
        unix_timestamp=rate_limit_timestamp_from_headers(headers),
    )
