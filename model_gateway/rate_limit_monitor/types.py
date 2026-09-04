"""Strict contracts for the shared rate-limit monitor."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from model_library.rate_limits import RateLimit

MonitorSourceName = Annotated[
    str,
    StringConstraints(pattern=r"^(default|pool_[1-9][0-9]*)$"),
]
MonitorSourceStatus = Literal["starting", "ok", "stale", "unsupported", "error"]
MonitorErrorCode = Literal["provider_error", "unsupported"]
_NonBlankModel = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
_Generation = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{32}$"),
]


class MonitorContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MonitorActivationRequest(MonitorContract):
    model: _NonBlankModel


class MonitorSourceState(MonitorContract):
    source: MonitorSourceName
    status: MonitorSourceStatus
    last_attempt_at: float | None = Field(default=None, ge=0)
    last_success_at: float | None = Field(default=None, ge=0)
    rate_limit: RateLimit | None = None
    error_code: MonitorErrorCode | None = None

    @model_validator(mode="after")
    def _require_consistent_state(self) -> Self:
        if self.status == "starting":
            if any(
                value is not None
                for value in (
                    self.last_attempt_at,
                    self.last_success_at,
                    self.rate_limit,
                    self.error_code,
                )
            ):
                raise ValueError("starting source fields must be null")
            return self

        if self.last_attempt_at is None:
            raise ValueError(f"{self.status} source last_attempt_at is required")

        has_last_good = self.last_success_at is not None and self.rate_limit is not None
        if self.status == "ok":
            if not has_last_good:
                raise ValueError("ok source last-good values are required")
            if self.error_code is not None:
                raise ValueError("ok source error_code must be null")
            return self

        if self.status == "stale":
            if not has_last_good:
                raise ValueError("stale source last-good values are required")
            return self

        if self.last_success_at is not None or self.rate_limit is not None:
            raise ValueError(f"{self.status} source last-good values must be null")
        if self.status == "unsupported" and self.error_code != "unsupported":
            raise ValueError("unsupported source error_code must be unsupported")
        if self.status == "error" and self.error_code != "provider_error":
            raise ValueError("error source error_code is required")
        return self


def derive_monitor_status(sources: list[MonitorSourceState]) -> MonitorSourceStatus:
    if sources and all(source.status == "ok" for source in sources):
        return "ok"
    if any(source.status == "stale" for source in sources):
        return "stale"
    has_last_good = any(source.last_success_at is not None for source in sources)
    has_completed_failure = any(
        source.status in {"error", "unsupported"} for source in sources
    )
    if has_last_good and has_completed_failure:
        return "stale"
    if sources and all(source.status == "unsupported" for source in sources):
        return "unsupported"
    if any(source.status == "error" for source in sources):
        return "error"
    return "starting"


class MonitorSourceFacts(MonitorContract):
    source: MonitorSourceName
    last_attempt_at: float | None = Field(default=None, ge=0)
    last_attempt_generation: _Generation | None = None
    last_success_at: float | None = Field(default=None, ge=0)
    rate_limit: RateLimit | None = None
    last_error_code: MonitorErrorCode | None = None

    @model_validator(mode="after")
    def _require_consistent_facts(self) -> Self:
        has_attempt = self.last_attempt_at is not None
        if has_attempt != (self.last_attempt_generation is not None):
            raise ValueError("attempt time and generation must appear together")

        has_last_good = self.last_success_at is not None
        if has_last_good != (self.rate_limit is not None):
            raise ValueError("last-success time and rate limit must appear together")
        if not has_attempt:
            if has_last_good or self.last_error_code is not None:
                raise ValueError("unattempted source facts must be empty")
            return self

        assert self.last_attempt_at is not None
        if self.last_success_at is not None:
            if self.last_success_at > self.last_attempt_at:
                raise ValueError("last success cannot follow the last attempt")
            if (
                self.last_error_code is None
                and self.last_success_at != self.last_attempt_at
            ):
                raise ValueError("successful attempt time must match last success")
        elif self.last_error_code is None:
            raise ValueError("attempted source requires success or error facts")
        return self


class MonitorFacts(MonitorContract):
    generation: _Generation
    model: _NonBlankModel
    sources: list[MonitorSourceFacts] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_complete_sources(self) -> Self:
        source_names = tuple(source.source for source in self.sources)
        expected_managed_sources = tuple(
            f"pool_{index}" for index in range(1, len(source_names) + 1)
        )
        if source_names != ("default",) and source_names != expected_managed_sources:
            raise ValueError("sources must be default or contiguous managed pools")
        return self


class MonitorState(MonitorContract):
    model: _NonBlankModel
    active: bool
    active_until: float = Field(ge=0)
    retention_until: float = Field(ge=0)
    status: MonitorSourceStatus
    sources: list[MonitorSourceState] = Field(min_length=1)


class MonitorListResponse(MonitorContract):
    server_time: float = Field(ge=0)
    states: list[MonitorState]


class MonitorActivationResponse(MonitorContract):
    server_time: float = Field(ge=0)
    state: MonitorState
