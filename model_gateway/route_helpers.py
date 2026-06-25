"""Shared route helpers for gateway endpoints."""

from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, TypeVar, cast

from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel

import model_library.telemetry as telemetry

from model_gateway.errors import map_exception_to_error
from model_gateway.metrics import emit_model_error
from model_gateway.telemetry_helpers import (
    latency_ms,
    provider_error_telemetry_attributes,
)
from model_gateway.types import ProviderError

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


async def provider_call_or_error(
    awaitable: Awaitable[T],
    *,
    provider: str | None,
    operation: str,
    dimensions: Mapping[str, str],
    start: float,
) -> T | ProviderError:
    try:
        return await awaitable
    except Exception as exc:
        err = map_exception_to_error(exc, provider=provider)
        error_attrs = provider_error_telemetry_attributes(err)
        telemetry.record_exception(exc, error_attrs)
        telemetry.set_status_error(err.body.code)
        telemetry.add_event(f"gateway.{operation}.error", error_attrs)
        emit_model_error(
            dimensions,
            error_code=err.body.code,
            latency_ms=latency_ms(start),
        )
        return ProviderError(
            code=err.body.code,
            message=err.body.message,
            provider=err.body.provider,
        )
