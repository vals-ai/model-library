"""Wire contracts and coordinator errors for benchmark admission."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

from model_gateway.types import GatewayRequestBase
from model_library.base import TokenRetryParams

MAX_BENCHMARK_WAIT_SECONDS = 30.0
DEFAULT_BENCHMARK_WAIT_SECONDS = 25.0
BenchmarkAdmissionState = Literal["waiting", "acquired", "released"]
BenchmarkAdmissionOutcome = Literal["finished", "cancelled", "failed"]
_NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BenchmarkCoordinatorError(RuntimeError):
    pass


class BenchmarkAdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: _NonBlankString
    run_id: _NonBlankString


class BenchmarkAcquireRequest(GatewayRequestBase):
    model: _NonBlankString
    run_id: _NonBlankString
    token_retry_params: TokenRetryParams
    total_requests: StrictInt | None = Field(default=None, ge=0)
    early_release: bool = True
    immediate_queue_release: bool = False


class BenchmarkWaitRequest(BenchmarkAdmissionRequest):
    timeout_seconds: StrictFloat = Field(
        default=DEFAULT_BENCHMARK_WAIT_SECONDS,
        ge=0,
        le=MAX_BENCHMARK_WAIT_SECONDS,
    )


class BenchmarkReleaseRequest(BenchmarkAdmissionRequest):
    outcome: BenchmarkAdmissionOutcome


class BenchmarkAdmissionResponse(BenchmarkAdmissionRequest):
    state: BenchmarkAdmissionState
    effective_token_limit: StrictInt = Field(gt=0)
    outcome: BenchmarkAdmissionOutcome | None = None

    @model_validator(mode="after")
    def _require_outcome_only_when_released(self) -> Self:
        if self.state == "released" and self.outcome is None:
            raise ValueError("released admission outcome is required")
        if self.state != "released" and self.outcome is not None:
            raise ValueError(f"{self.state} admission outcome must be null")
        return self
