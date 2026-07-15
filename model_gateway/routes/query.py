"""Query route registration."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import FastAPI, Request

import model_library.telemetry as telemetry
from model_gateway.cache import ModelCache
from model_gateway.history import sign_history
from model_gateway.metrics import model_dimensions
from model_gateway.model_helpers import (
    get_query_llm,
    provider_from_model,
    require_raw_input_secret,
)
from model_gateway.route_helpers import GatewayOperation, ok_response
from model_gateway.telemetry_helpers import (
    query_config_params,
    query_telemetry_attributes,
)
from model_gateway.types import (
    GatewayResponse,
    ProviderError,
    QueryRequest,
    query_result_response_body,
)
from model_gateway.usage_ledger.store import build_success_usage_event
from model_library.base import LLM, dump_gateway_config, dump_llm_config

_T = TypeVar("_T")


def _is_local_startup_canary(request: Request, body: QueryRequest) -> bool:
    client = request.client
    return (
        body.run_id == "gateway-startup-canary"
        and client is not None
        and client.host in {"127.0.0.1", "::1", "localhost"}
    )


def register_query_routes(app: FastAPI, *, cache: ModelCache) -> None:
    @app.post("/query")
    async def query(request: Request, body: QueryRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        display_config = dump_gateway_config(body.config)
        query_params = {
            "run_id": body.run_id,
            "question_id": body.question_id,
            "query_id": body.query_id,
            "identity": body.identity,
            "in_agent": body.in_agent,
        }
        config_query_params = query_config_params(display_config)
        token_retry_params = (
            body.token_retry_params.model_dump(mode="json")
            if body.token_retry_params is not None
            else None
        )
        dimensions = model_dimensions(
            operation="query",
            model=body.model,
            config=display_config,
            params=config_query_params,
            token_retry_params=token_retry_params,
        )
        operation = GatewayOperation(
            operation="query",
            dimensions=dimensions,
            start=start,
            provider=provider_from_model(body.model),
        )
        config_attrs: dict[str, object] = {}
        config_hash: str | None = None
        redacted_config: dict[str, object] | None = None
        if telemetry.is_recording():
            config_hash, redacted_config = telemetry.config_fingerprint(
                body.model,
                display_config,
                config_query_params,
                token_retry_params,
            )
            config_attrs["model.config_hash"] = config_hash
        query_attrs = {
            **telemetry.model_attributes(operation="query", model=body.model),
            **telemetry.run_attributes(query_params),
            **query_telemetry_attributes(
                body,
                display_config,
                config_query_params=config_query_params,
                token_retry_params=token_retry_params,
            ),
            **config_attrs,
        }
        operation.start_event(query_attrs)
        if config_hash is not None and redacted_config is not None:
            telemetry.record_config_seen(config_hash, redacted_config)
        error_phase: str | None = None

        def record_phase_error(phase: str, exc: Exception) -> None:
            error_attrs = {
                "gateway.error.phase": phase,
                "exception.type": type(exc).__name__,
            }
            telemetry.set_attributes(error_attrs)
            operation.add_event(f"{phase}_error", error_attrs)

        async def run_phase(
            phase: str,
            action: Callable[[], Awaitable[_T]],
            *,
            handled_exceptions: tuple[type[Exception], ...] = (Exception,),
            map_error: Callable[[Exception], Exception] | None = None,
        ) -> _T:
            nonlocal error_phase
            phase_start = time.perf_counter()
            try:
                result = await action()
            except handled_exceptions as exc:
                operation.record_phase(phase, outcome="error", start=phase_start)
                error_phase = phase
                record_phase_error(phase, exc)
                if map_error is not None:
                    raise map_error(exc) from exc
                raise
            operation.record_phase(phase, outcome="success", start=phase_start)
            return result

        try:
            operation.add_event("model_cache_lookup")
            llm = await run_phase(
                "model_cache_lookup", lambda: get_query_llm(cache, body)
            )

            # Restored provider objects can be much larger than signed JSON blobs.
            # The capacity middleware admits this route before body parsing.
            secret = request.app.state.hmac_secret

            async def restore_history() -> None:
                require_raw_input_secret(body.inputs, secret=secret or None)
                operation.add_event("restore_history_start")
                LLM.restore_raw_fields(body.inputs, secret=secret or None)
                operation.add_event("restore_history_done")

            await run_phase(
                "restore_history",
                restore_history,
                handled_exceptions=(KeyError, TypeError, ValueError),
                map_error=lambda exc: ValueError(
                    f"Invalid signed raw history blob: {exc}"
                ),
            )

            result_or_error = await operation.provider_call(
                llm.query(
                    body.inputs,
                    tools=body.tools,
                    output_schema=body.output_schema,
                    run_id=body.run_id,
                    question_id=body.question_id,
                    in_agent=body.in_agent,
                    query_id=body.query_id,
                ),
                span_attrs=query_attrs,
            )
            if isinstance(result_or_error, ProviderError):
                return ok_response(GatewayResponse(error=result_or_error))
            result = result_or_error
            operation.add_event(
                "provider_call_done",
                {
                    "gen_ai.usage.input_tokens": result.metadata.total_input_tokens,
                    "gen_ai.usage.output_tokens": result.metadata.total_output_tokens,
                    "gateway.tool_call.count": len(result.tool_calls),
                    "gateway.provider_tool_event.count": len(
                        result.provider_tool_events
                    ),
                    "gateway.history_item.count": len(result.history),
                },
            )

            operation.add_event("sign_history_start")

            async def sign_result_history() -> str:
                return sign_history(result.history, secret=secret or None)

            signed_history = await run_phase("sign_history", sign_result_history)
            operation.add_event("sign_history_done")

            if request.app.state.usage_ledger.enabled and not _is_local_startup_canary(
                request, body
            ):

                async def write_usage_event() -> None:
                    usage_event = build_success_usage_event(
                        body=body,
                        config=config,
                        query_params=query_params,
                        dimensions=dimensions,
                        result=result,
                        api_key_fingerprint=getattr(
                            request.state, "gateway_api_key_fingerprint", None
                        ),
                    )
                    telemetry.set_attributes(
                        {"gateway.usage_event_id": usage_event["usage_event_id"]}
                    )
                    await request.app.state.usage_ledger.write_success(usage_event)

                await run_phase("usage_ledger", write_usage_event)

            extra_metrics: dict[str, tuple[int | float, str]] = {
                "InputTokens": (result.metadata.total_input_tokens, "Count"),
                "OutputTokens": (result.metadata.total_output_tokens, "Count"),
                "TotalTokens": (
                    result.metadata.total_input_tokens
                    + result.metadata.total_output_tokens,
                    "Count",
                ),
                "ToolCallCount": (len(result.tool_calls), "Count"),
                "ProviderToolEventCount": (
                    len(result.provider_tool_events),
                    "Count",
                ),
                "HistoryItemCount": (len(result.history), "Count"),
            }
            if result.metadata.cost is not None:
                extra_metrics["CostUsd"] = (result.metadata.cost.total, "None")
            if body.token_retry_params is not None:
                extra_metrics["TokenRetryRequestCount"] = (1, "Count")
            response_attrs: dict[str, object | None] = {
                "gateway.result.finish_reason": result.finish_reason.reason
                if result.finish_reason
                else None,
            }
            # Enable only for controlled deployment-drain tests; keep disabled otherwise.
            if False and not _is_local_startup_canary(request, body):
                _DEPLOY_TEST_RESPONSE_DELAY_SECONDS = 60
                telemetry.add_event(
                    "deploy_test_response_delay_start",
                    {
                        "gateway.deploy_test.delay_seconds": (
                            _DEPLOY_TEST_RESPONSE_DELAY_SECONDS
                        )
                    },
                )
                await asyncio.sleep(_DEPLOY_TEST_RESPONSE_DELAY_SECONDS)
                telemetry.add_event("deploy_test_response_delay_done")

            return operation.success(
                query_result_response_body(result, signed_history=signed_history),
                extra_metrics=extra_metrics,
                attrs=response_attrs,
            )

        except Exception as exc:
            return operation.error(exc, phase=error_phase)
