import os
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from model_library.rate_limits import (
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)

def test_rate_limit_response_allows_data_error_or_no_data_but_not_both():
    from model_gateway.types import ProviderError, RateLimitResponse
    from model_library.rate_limits import RateLimit

    error = ProviderError(code="internal_error", message="provider failed")
    rate_limit = RateLimit(
        requests=(RequestRateLimit(limit=10),),
        unix_timestamp=1.0,
    )

    assert RateLimitResponse().model_dump(exclude_none=True) == {}
    assert RateLimitResponse(rate_limit=rate_limit).model_dump(
        mode="json", exclude_none=True
    ) == {
        "rate_limit": {
            "requests": [{"limit": 10, "mode": "sliding_window"}],
            "unix_timestamp": 1.0,
        }
    }
    assert RateLimitResponse(error=error).model_dump(
        mode="json", exclude_none=True
    ) == {
        "error": {
            "type": "ProviderError",
            "code": "internal_error",
            "message": "provider failed",
        }
    }

    with pytest.raises(ValidationError):
        RateLimitResponse(rate_limit=rate_limit, error=error)

async def test_fixed_rate_limit_round_trips_through_gateway_wire() -> None:
    from model_gateway.types import RateLimitResponse
    from model_library.base.gateway import GatewayLLM
    from model_library.rate_limits import RateLimit

    rate_limit = RateLimit(
        requests=(RequestRateLimit(limit=25, remaining=20, mode="concurrency"),),
        tokens=TokenRateLimit(total=RateLimitCapacity(limit=1_000, remaining=400)),
        scope="shared",
        unix_timestamp=1.0,
    )
    wire = RateLimitResponse(rate_limit=rate_limit).model_dump(
        mode="json",
        exclude_none=True,
    )
    with patch.dict(
        os.environ,
        {
            "MODEL_GATEWAY_URL": "http://localhost:8000",
            "MODEL_GATEWAY_API_KEY": "test-key",
        },
    ):
        llm = GatewayLLM("gpt-4o", "openai")

    with patch.object(
        llm,
        "post_gateway",
        new_callable=AsyncMock,
        return_value=wire,
    ):
        reconstructed = await llm.get_rate_limit()

    assert reconstructed == RateLimit(
        requests=(RequestRateLimit(limit=25, remaining=20, mode="concurrency"),),
        tokens=TokenRateLimit(total=RateLimitCapacity(limit=1_000, remaining=400)),
        scope="shared",
        unix_timestamp=1.0,
    )
    assert "token_limit_total" not in wire["rate_limit"]
    assert "token_remaining_total" not in wire["rate_limit"]

@pytest.mark.parametrize(
    "raw_rate_limit",
    [
        {
            "requests": [{"limit": 1}],
            "unix_timestamp": 1.0,
            "unexpected": True,
        },
        {"requests": [{"limit": -1}], "unix_timestamp": 1.0},
        {"requests": [{"remaining": 1}], "unix_timestamp": 1.0},
        {
            "tokens": {
                "total": {"limit": 1},
                "input": {"limit": 1},
                "output": {"limit": 1},
            },
            "unix_timestamp": 1.0,
        },
        {"requests": [{"limit": 1}], "unix_timestamp": float("nan")},
        {"requests": [{"limit": 1}], "unix_timestamp": float("inf")},
        {"requests": [{"limit": 1}], "unix_timestamp": float("-inf")},
    ],
)
async def test_gateway_rate_limit_rejects_invalid_wire_payload(
    raw_rate_limit: dict[str, object],
) -> None:
    from model_library.base.gateway import GatewayLLM

    with patch.dict(
        os.environ,
        {
            "MODEL_GATEWAY_URL": "http://localhost:8000",
            "MODEL_GATEWAY_API_KEY": "test-key",
        },
    ):
        llm = GatewayLLM("gpt-4o", "openai")

    with (
        patch.object(
            llm,
            "post_gateway",
            new_callable=AsyncMock,
            return_value={"rate_limit": raw_rate_limit},
        ),
        pytest.raises(ValidationError),
    ):
        await llm.get_rate_limit()
