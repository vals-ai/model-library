import json

import pytest

from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)
from model_library.retriers.token.background import _token_correction_observation
from model_library.retriers.token.utils import (
    deserialize_last_header,
    serialize_last_header,
)


def _capacity(limit: int, remaining: int | None = None) -> RateLimitCapacity:
    return RateLimitCapacity(limit=limit, remaining=remaining)


def test_last_header_redis_mapping_round_trips_grouped_observation() -> None:
    rate_limit = RateLimit(
        requests=(
            RequestRateLimit(limit=10, remaining=0, mode="concurrency"),
        ),
        tokens=TokenRateLimit(
            input=_capacity(400, 0),
            output=_capacity(100, 100),
        ),
        scope="shared",
        unix_timestamp=1.0,
    )

    mapping = serialize_last_header(rate_limit)

    assert set(mapping) == {"rate_limit"}
    assert json.loads(mapping["rate_limit"])["tokens"]["input"] == {
        "limit": 400,
        "remaining": 0,
    }
    assert deserialize_last_header(mapping) == RateLimit(
        requests=(RequestRateLimit(limit=10, remaining=0, mode="concurrency"),),
        tokens=TokenRateLimit(input=_capacity(400, 0), output=_capacity(100, 100)),
        scope="shared",
        unix_timestamp=1.0,
    )


def test_total_token_pair_preserves_zero_remaining() -> None:
    rate_limit = RateLimit(
        tokens=TokenRateLimit(total=_capacity(1_000, 0)),
        unix_timestamp=1.0,
    )

    assert _token_correction_observation(rate_limit) == (1_000, 0)


def test_complete_directional_pairs_form_one_correction_pair() -> None:
    rate_limit = RateLimit(
        tokens=TokenRateLimit(input=_capacity(600, 300), output=_capacity(400, 0)),
        unix_timestamp=1.0,
    )

    assert _token_correction_observation(rate_limit) == (1_000, 300)


@pytest.mark.parametrize(
    "rate_limit",
    [
        RateLimit(tokens=TokenRateLimit(total=_capacity(1_000)), unix_timestamp=1.0),
        RateLimit(
            tokens=TokenRateLimit(input=_capacity(600), output=_capacity(400)),
            unix_timestamp=1.0,
        ),
        RateLimit(
            tokens=TokenRateLimit(input=_capacity(600, 300), output=_capacity(400)),
            unix_timestamp=1.0,
        ),
        RateLimit(requests=(RequestRateLimit(limit=100),), unix_timestamp=1.0),
    ],
)
def test_limit_only_or_partial_data_does_not_drive_correction(
    rate_limit: RateLimit,
) -> None:
    assert _token_correction_observation(rate_limit) is None
