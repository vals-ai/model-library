from typing import Literal

import pytest
from pydantic import ValidationError

from model_gateway.benchmark_admission_types import (
    BenchmarkAcquireRequest,
    BenchmarkAdmissionResponse,
)


def _acquire_payload() -> dict[str, object]:
    return {
        "model": "openai/gpt-4o",
        "run_id": "run-123",
        "token_retry_params": {
            "input_modifier": 1.0,
            "output_modifier": 1.0,
            "use_dynamic_estimate": True,
            "limit": 10_000,
            "limit_refresh_seconds": 60,
        },
    }


@pytest.mark.parametrize("field", ["model", "run_id"])
def test_benchmark_requests_reject_blank_identity(field: str) -> None:
    payload = _acquire_payload()
    payload[field] = "  "

    with pytest.raises(ValidationError):
        BenchmarkAcquireRequest.model_validate(payload)


def test_benchmark_acquire_request_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        BenchmarkAcquireRequest.model_validate(
            {**_acquire_payload(), "total_requests": -1}
        )


def test_benchmark_acquire_requires_token_retry_params() -> None:
    payload = _acquire_payload()
    del payload["token_retry_params"]

    with pytest.raises(ValidationError):
        BenchmarkAcquireRequest.model_validate(payload)


def test_benchmark_acquire_rejects_obsolete_wait_timeout() -> None:
    with pytest.raises(ValidationError):
        BenchmarkAcquireRequest.model_validate(
            {**_acquire_payload(), "timeout_seconds": 25.0}
        )


@pytest.mark.parametrize("state", ["waiting", "acquired"])
def test_live_benchmark_admission_response_has_no_terminal_outcome(
    state: Literal["waiting", "acquired"],
) -> None:
    response = BenchmarkAdmissionResponse(
        state=state,
        model="openai/gpt-4o",
        run_id="run-123",
        effective_token_limit=10_000,
        outcome=None,
    )

    assert response.state == state
    assert response.outcome is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "state": "released",
            "model": "openai/gpt-4o",
            "run_id": "run-123",
            "effective_token_limit": 10_000,
            "outcome": None,
        },
        {
            "state": "acquired",
            "model": "openai/gpt-4o",
            "run_id": "run-123",
            "effective_token_limit": 10_000,
            "outcome": "finished",
        },
    ],
)
def test_benchmark_admission_response_enforces_terminal_outcome(payload) -> None:
    with pytest.raises(ValidationError):
        BenchmarkAdmissionResponse.model_validate(payload)


def test_released_benchmark_admission_response_requires_outcome() -> None:
    response = BenchmarkAdmissionResponse(
        state="released",
        model="openai/gpt-4o",
        run_id="run-123",
        effective_token_limit=10_000,
        outcome="cancelled",
    )

    assert response.outcome == "cancelled"
