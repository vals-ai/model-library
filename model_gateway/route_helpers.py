"""Shared route helpers for gateway endpoints."""

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
import time
from typing import Any, Protocol, TypeVar, cast

from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel

import model_library.telemetry as telemetry

from model_gateway.errors import map_exception_to_error
from model_gateway.metrics import (
    MetricSpec,
    emit_model_error,
    emit_model_success,
    record_gateway_phase,
)
from model_gateway.telemetry_helpers import error_telemetry_attributes, latency_ms
from model_gateway.types import ProviderError
from model_library.exceptions import exception_to_provider_error

T = TypeVar("T")


class EmbeddingModel(Protocol):
    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]: ...


class ModerationModel(Protocol):
    async def moderate_content(self, text: str) -> object: ...


def dump_pydantic_response(response: object) -> dict[str, Any]:
    if isinstance(response, PydanticBaseModel):
        return response.model_dump(mode="json", by_alias=True)
    if isinstance(response, dict):
        return cast(dict[str, Any], response)
    raise TypeError(f"Unsupported response type: {type(response).__name__}")


def ok_response(body: PydanticBaseModel | dict[str, Any]) -> JSONResponse:
    content = (
        body.model_dump(mode="json", exclude_none=True)
        if isinstance(body, PydanticBaseModel)
        else body
    )
    return JSONResponse(status_code=200, content=content)


class ProviderCallTelemetryError(Exception):
    """Sanitized provider-call failure used for exception telemetry."""


def provider_error_from_exception(
    exc: Exception, *, provider: str | None = None
) -> ProviderError:
    return ProviderError.model_validate(
        {**exception_to_provider_error(exc), "provider": provider}
    )


def provider_error_telemetry_attributes(
    err: ProviderError,
) -> dict[str, object | None]:
    attrs: dict[str, object | None] = {
        "gateway.error.code": "provider_error",
        "gateway.error.provider": err.provider,
        "gateway.error.phase": "provider_call",
        "gateway.status_code": 200,
        "http.status_code": 200,
        "http.response.status_code": 200,
        "gateway.provider_error.exception_type": err.exception_type,
    }
    if err.code is not None:
        attrs["gateway.provider_error.code"] = err.code
    if err.status_code is not None:
        attrs["gateway.provider_error.status_code"] = err.status_code
    return attrs


@dataclass
class GatewayOperation:
    operation: str
    dimensions: Mapping[str, str]
    start: float
    provider: str | None

    def start_event(self, attrs: Mapping[str, object | None]) -> None:
        telemetry.set_attributes(attrs)
        self.add_event("start")

    def add_event(
        self, suffix: str, attrs: Mapping[str, object | None] | None = None
    ) -> None:
        name = f"gateway.{self.operation}.{suffix}"
        if attrs is None:
            telemetry.add_event(name)
            return
        telemetry.add_event(name, attrs)

    def record_phase(
        self,
        phase: str,
        *,
        outcome: str,
        start: float,
    ) -> None:
        record_gateway_phase(
            operation=self.operation,
            provider=self.dimensions.get("Provider") or self.provider,
            phase=phase,
            outcome=outcome,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def provider_call(
        self,
        awaitable: Awaitable[T],
        *,
        span_attrs: Mapping[str, object | None],
    ) -> T | ProviderError:
        self.add_event("provider_call_start")
        phase_start = time.perf_counter()
        with telemetry.start_span(
            f"gateway.{self.operation}.provider_call",
            span_attrs,
            kind="client",
        ):
            try:
                result = await awaitable
            except Exception as exc:
                self.record_phase("provider_call", outcome="error", start=phase_start)
                err = provider_error_from_exception(exc, provider=self.provider)
                error_attrs = provider_error_telemetry_attributes(err)
                error_code = str(error_attrs["gateway.error.code"])
                telemetry.record_exception(
                    ProviderCallTelemetryError("Provider call failed"),
                    error_attrs,
                )
                telemetry.set_status_error(error_code)
                self.add_event("error", error_attrs)
                emit_model_error(
                    self.dimensions,
                    error_code=error_code,
                    latency_ms=latency_ms(self.start),
                )
                return err
            if isinstance(result, ProviderError):
                return result
            self.record_phase("provider_call", outcome="success", start=phase_start)
            return result

    def success(
        self,
        body: PydanticBaseModel | dict[str, Any],
        *,
        extra_metrics: Mapping[str, MetricSpec] | None = None,
        attrs: Mapping[str, object | None] | None = None,
    ) -> JSONResponse:
        emit_model_success(
            self.dimensions,
            latency_ms=latency_ms(self.start),
            extra_metrics=extra_metrics,
        )
        telemetry.set_attributes(
            {"gateway.latency_ms": latency_ms(self.start), **(attrs or {})}
        )
        telemetry.set_status_ok()
        self.add_event("success")
        return ok_response(body)

    def error(self, exc: Exception, *, phase: str | None = None) -> JSONResponse:
        err = map_exception_to_error(exc, provider=self.provider)
        error_attrs = error_telemetry_attributes(err, phase=phase)
        telemetry.record_exception(exc, error_attrs)
        telemetry.set_status_error(err.body.code)
        self.add_event("error", error_attrs)
        emit_model_error(
            self.dimensions,
            error_code=err.body.code,
            latency_ms=latency_ms(self.start),
        )
        return JSONResponse(status_code=err.status_code, content=err.body.model_dump())
