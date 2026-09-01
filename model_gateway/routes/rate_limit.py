"""Provider rate-limit route registration."""

import asyncio
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from model_library.base import LLMConfig, dump_llm_config

from model_gateway import model_helpers
from model_gateway.cache import ModelCache
from model_gateway.metrics import model_dimensions
from model_gateway.route_helpers import GatewayOperation, ok_response
from model_gateway.telemetry_helpers import dimension_telemetry_attributes
from model_gateway.types import ProviderError, RateLimitRequest, RateLimitResponse

RATE_LIMIT_CACHE_SECONDS = 60.0
RATE_LIMIT_CACHE_MAXSIZE = 128
RATE_LIMIT_MAX_IN_FLIGHT_PROBES = 64
RATE_LIMIT_PROBE_TIMEOUT_SECONDS = 30.0


class RateLimitProbeService:
    def __init__(self, cache: ModelCache) -> None:
        self._cache = cache
        self._rate_limits: OrderedDict[str, tuple[float, RateLimitResponse]] = (
            OrderedDict()
        )
        self._probes: dict[str, asyncio.Task[JSONResponse]] = {}

    async def _probe(
        self,
        body: RateLimitRequest,
        key: str,
        config: dict[str, Any],
    ) -> JSONResponse:
        try:
            start = time.perf_counter()
            dimensions = model_dimensions(
                operation="rate_limit",
                model=body.model,
                config=config,
            )
            operation = GatewayOperation(
                operation="rate_limit",
                dimensions=dimensions,
                start=start,
                provider=model_helpers.provider_from_model(body.model),
            )
            attrs: dict[str, object | None] = {
                **telemetry.model_attributes(operation="rate_limit", model=body.model),
                **dimension_telemetry_attributes(dimensions),
            }
            operation.start_event(attrs)
            try:
                operation.add_event("model_cache_lookup")
                llm = model_helpers.get_cached_llm(
                    self._cache,
                    body,
                    config=config,
                    model_config=config,
                )
                result = await operation.provider_call(
                    asyncio.wait_for(
                        llm.get_rate_limit(), timeout=RATE_LIMIT_PROBE_TIMEOUT_SECONDS
                    ),
                    span_attrs=attrs,
                )
                if isinstance(result, ProviderError):
                    safe_error = ProviderError(
                        message="Provider rate-limit probe failed",
                        provider=operation.provider,
                        exception_type=result.exception_type,
                        status_code=result.status_code,
                    )
                    return ok_response(RateLimitResponse(error=safe_error))
                operation.add_event("provider_call_done")
                response = RateLimitResponse(rate_limit=result)
                now = time.monotonic()
                for expired in [
                    cache_key
                    for cache_key, (expiry, _) in self._rate_limits.items()
                    if now >= expiry
                ]:
                    del self._rate_limits[expired]
                self._rate_limits[key] = (now + RATE_LIMIT_CACHE_SECONDS, response)
                while len(self._rate_limits) > RATE_LIMIT_CACHE_MAXSIZE:
                    self._rate_limits.popitem(last=False)
                return operation.success(response)
            except Exception as exc:
                return operation.error(exc, phase="rate_limit")
        finally:
            self._probes.pop(key, None)

    async def get_rate_limit(self, body: RateLimitRequest) -> JSONResponse:
        if (
            body.config.custom_api_key is not None
            or body.config.custom_endpoint is not None
        ):
            return ok_response(RateLimitResponse())

        config = dump_llm_config(LLMConfig())
        key = self._cache.make_key(body.model, config)
        cached = self._rate_limits.get(key)
        if cached is not None and time.monotonic() < cached[0]:
            return ok_response(cached[1])

        shared_probe = self._probes.get(key)
        if shared_probe is None:
            if len(self._probes) >= RATE_LIMIT_MAX_IN_FLIGHT_PROBES:
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": "gateway_overloaded",
                        "message": "Gateway rate-limit probe capacity is full",
                    },
                )
            shared_probe = self._probes[key] = asyncio.create_task(
                self._probe(body, key, config)
            )
        # A cancelled waiter must not cancel the shared provider probe.
        return await asyncio.shield(shared_probe)

    async def close(self) -> None:
        probes = tuple(self._probes.values())
        self._probes.clear()
        for probe in probes:
            probe.cancel()
        if probes:
            await asyncio.gather(*probes, return_exceptions=True)


def register_rate_limit_route(
    app: FastAPI, *, cache: ModelCache
) -> RateLimitProbeService:
    service = RateLimitProbeService(cache)

    @app.post("/rate-limit")
    async def rate_limit(body: RateLimitRequest):
        return await service.get_rate_limit(body)

    return service
