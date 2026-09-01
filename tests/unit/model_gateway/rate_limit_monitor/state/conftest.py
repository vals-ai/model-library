from typing import cast

import pytest
from fakeredis import aioredis

from model_gateway.rate_limit_monitor.state import MonitorRedis, RateLimitMonitorStore


@pytest.fixture
def redis() -> aioredis.FakeRedis:
    return aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis: aioredis.FakeRedis) -> RateLimitMonitorStore:
    return RateLimitMonitorStore(cast(MonitorRedis, redis))
