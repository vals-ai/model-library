"""Telemetry helper functions for gateway routes."""

import time
from collections.abc import Mapping
from typing import Any

import model_library.telemetry as telemetry

from model_gateway.errors import ErrorResponse
from model_gateway.metrics import param_group, provider_endpoint_bucket
from model_gateway.types import QueryRequest


def latency_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def query_config_params(config: Mapping[str, Any]) -> dict[str, object]:
    return {key: value for key, value in config.items() if value is not None}


def llm_config_telemetry_attributes(
    config: Mapping[str, Any],
) -> dict[str, object | None]:
    return {
        f"llm.config.{key}": telemetry.json_attribute(value)
        for key, value in config.items()
        if value is not None
    }


def query_telemetry_attributes(
    body: QueryRequest,
    config: Mapping[str, Any],
    *,
    config_query_params: Mapping[str, object | None],
    token_retry_params: object | None,
) -> dict[str, object | None]:
    attrs: dict[str, object | None] = {
        f"gen_ai.request.{key}": telemetry.json_attribute(value)
        for key, value in config_query_params.items()
    }
    attrs.update(llm_config_telemetry_attributes(config))
    attrs.update(
        {
            "gateway.input.count": len(body.inputs),
            "gateway.tool.count": len(body.tools),
            "gateway.output_schema.mode": telemetry.mode_attribute(
                body.output_schema is not None
            ),
            "gateway.retry_queue.mode": telemetry.mode_attribute(
                body.token_retry_params is not None
            ),
            "model.registry_key": body.model,
            "model.provider_endpoint": provider_endpoint_bucket(config),
            "model.param_group": param_group(
                config,
                config_query_params,
                token_retry_params,
            ),
        }
    )
    if body.token_retry_params is not None:
        attrs.update(
            {
                "retry_queue.mode": "enabled",
                "retry_queue.limit": body.token_retry_params.limit,
                "retry_queue.input_modifier": body.token_retry_params.input_modifier,
                "retry_queue.output_modifier": body.token_retry_params.output_modifier,
                "retry_queue.dynamic_estimate.mode": telemetry.mode_attribute(
                    body.token_retry_params.use_dynamic_estimate
                ),
            }
        )
    else:
        attrs["retry_queue.mode"] = "disabled"
    return attrs


def dimension_telemetry_attributes(
    dimensions: Mapping[str, str],
) -> dict[str, object | None]:
    return {
        "model.provider_endpoint": dimensions["ProviderEndpoint"],
        "model.param_group": dimensions["ParamGroup"],
        "gateway.operation": dimensions["Operation"],
    }


def error_telemetry_attributes(
    err: ErrorResponse, *, phase: str | None = None
) -> dict[str, object | None]:
    attrs: dict[str, object | None] = {
        "gateway.error.code": err.body.code,
        "gateway.error.provider": err.body.provider,
        "gateway.status_code": err.status_code,
        "http.status_code": err.status_code,
        "http.response.status_code": err.status_code,
    }
    if phase is not None:
        attrs["gateway.error.phase"] = phase
    return attrs


def provider_error_telemetry_attributes(
    err: ErrorResponse,
) -> dict[str, object | None]:
    attrs = error_telemetry_attributes(err, phase="provider_call")
    attrs["gateway.provider_error.status_code"] = err.status_code
    attrs["gateway.status_code"] = 200
    attrs["http.status_code"] = 200
    attrs["http.response.status_code"] = 200
    return attrs
