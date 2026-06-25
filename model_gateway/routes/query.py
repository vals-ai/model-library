"""Query route registration."""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from model_library.base import dump_llm_config
from model_library.base.base import LLM

from model_gateway.cache import ModelCache
from model_gateway.errors import map_exception_to_error
from model_gateway.history import sign_history
from model_gateway.metrics import emit_model_error, emit_model_success, model_dimensions
from model_gateway.model_helpers import get_query_llm, has_serialized_raw_blob
from model_gateway.route_helpers import ok_response, provider_call_or_error
from model_gateway.telemetry_helpers import (
    error_telemetry_attributes,
    latency_ms,
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
        query_params = {
            "run_id": body.run_id,
            "question_id": body.question_id,
            "query_id": body.query_id or uuid.uuid4().hex[:14],
            "identity": body.identity,
            "in_agent": body.in_agent,
        }
        config_query_params = query_config_params(config)
        dimensions = model_dimensions(
            operation="query",
            model=body.model,
            config=config,
            params=config_query_params,
            token_retry_params=body.token_retry_params,
        )
        config_attrs: dict[str, object] = {}
        config_hash: str | None = None
        redacted_config: dict[str, object] | None = None
        if telemetry.is_recording():
            config_hash, redacted_config = telemetry.config_fingerprint(
                body.model,
                config,
                config_query_params,
                body.token_retry_params,
            )
            config_attrs["model.config_hash"] = config_hash
        telemetry.set_attributes(
            {
                **telemetry.model_attributes(operation="query", model=body.model),
                **telemetry.run_attributes(query_params),
                **query_telemetry_attributes(body, config),
                **config_attrs,
            }
        )
        if config_hash is not None and redacted_config is not None:
            telemetry.record_config_seen(config_hash, redacted_config)
        telemetry.add_event("gateway.query.start")
        error_phase: str | None = None
        try:
            telemetry.add_event("gateway.query.model_cache_lookup")
            llm = await get_query_llm(cache, body)

            # Restored provider objects can be much larger than signed JSON blobs.
            # The capacity middleware admits this route before body parsing.
            secret = request.app.state.hmac_secret
            if has_serialized_raw_blob(body.inputs) and not secret:
                raise ValueError(
                    "MODEL_GATEWAY_HMAC_SECRET is required to accept raw history blobs"
                )
            try:
                telemetry.add_event("gateway.query.restore_history_start")
                LLM.restore_raw_fields(body.inputs, secret=secret or None)
                telemetry.add_event("gateway.query.restore_history_done")
            except (KeyError, TypeError, ValueError) as exc:
                error_phase = "restore_history"
                restore_error_attrs = {
                    "gateway.error.phase": error_phase,
                    "exception.type": type(exc).__name__,
                }
                telemetry.set_attributes(restore_error_attrs)
                telemetry.add_event(
                    "gateway.query.restore_history_error", restore_error_attrs
                )
                raise ValueError(f"Invalid signed raw history blob: {exc}") from exc

            telemetry.add_event("gateway.query.provider_call_start")
            with telemetry.start_span(
                "gateway.query.provider_call",
                {
                    **telemetry.model_attributes(operation="query", model=body.model),
                    **telemetry.run_attributes(query_params),
                    **query_telemetry_attributes(body, config),
                    **config_attrs,
                },
                kind="client",
            ):
                result_or_error = await provider_call_or_error(
                    llm.query(
                        body.inputs,
                        tools=body.tools,
                        output_schema=body.output_schema,
                        run_id=body.run_id,
                        question_id=body.question_id,
                        in_agent=body.in_agent,
                        query_id=query_params["query_id"],
                    ),
                    provider=body.model.partition("/")[0] or None,
                    operation="query",
                    dimensions=dimensions,
                    start=start,
                )
            if isinstance(result_or_error, ProviderError):
                return ok_response(GatewayResponse(error=result_or_error))
            result = result_or_error
            telemetry.add_event(
                "gateway.query.provider_call_done",
                {
                    "gen_ai.usage.input_tokens": result.metadata.total_input_tokens,
                    "gen_ai.usage.output_tokens": result.metadata.total_output_tokens,
                    "gateway.tool_call.count": len(result.tool_calls),
                    "gateway.history_item.count": len(result.history),
                },
            )

            telemetry.add_event("gateway.query.sign_history_start")
            try:
                signed_history = sign_history(result.history, secret=secret or None)
            except Exception as exc:
                error_phase = "sign_history"
                sign_error_attrs = {
                    "gateway.error.phase": error_phase,
                    "exception.type": type(exc).__name__,
                }
                telemetry.set_attributes(sign_error_attrs)
                telemetry.add_event(
                    "gateway.query.sign_history_error", sign_error_attrs
                )
                raise
            telemetry.add_event("gateway.query.sign_history_done")

            if request.app.state.usage_ledger.enabled and not _is_local_startup_canary(
                request, body
            ):
                try:
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
                except Exception as exc:
                    error_phase = "usage_ledger"
                    ledger_error_attrs = {
                        "gateway.error.phase": error_phase,
                        "exception.type": type(exc).__name__,
                    }
                    telemetry.set_attributes(ledger_error_attrs)
                    telemetry.add_event(
                        "gateway.query.usage_ledger_error", ledger_error_attrs
                    )
                    raise

            extra_metrics: dict[str, tuple[int | float, str]] = {
                "InputTokens": (result.metadata.total_input_tokens, "Count"),
                "OutputTokens": (result.metadata.total_output_tokens, "Count"),
                "TotalTokens": (
                    result.metadata.total_input_tokens
                    + result.metadata.total_output_tokens,
                    "Count",
                ),
                "ToolCallCount": (len(result.tool_calls), "Count"),
                "HistoryItemCount": (len(result.history), "Count"),
            }
            if result.metadata.cost is not None:
                extra_metrics["CostUsd"] = (result.metadata.cost.total, "None")
            if body.token_retry_params is not None:
                extra_metrics["TokenRetryRequestCount"] = (1, "Count")
            emit_model_success(
                dimensions,
                latency_ms=latency_ms(start),
                extra_metrics=extra_metrics,
            )
            response_attrs: dict[str, object | None] = {
                "gateway.latency_ms": latency_ms(start),
                "gateway.result.finish_reason": result.finish_reason.reason
                if result.finish_reason
                else None,
            }
            telemetry.set_attributes(response_attrs)
            telemetry.set_status_ok()
            telemetry.add_event("gateway.query.success")

            return ok_response(
                query_result_response_body(result, signed_history=signed_history)
            )

        except Exception as exc:
            err = map_exception_to_error(
                exc, provider=body.model.partition("/")[0] or None
            )
            error_attrs = error_telemetry_attributes(err, phase=error_phase)
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error(err.body.code)
            telemetry.add_event("gateway.query.error", error_attrs)
            emit_model_error(
                dimensions,
                error_code=err.body.code,
                latency_ms=latency_ms(start),
            )
            return JSONResponse(
                status_code=err.status_code, content=err.body.model_dump()
            )
