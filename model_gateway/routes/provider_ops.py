"""Provider-operation route registration."""

import base64
import binascii
import io
import time
from typing import cast

from fastapi import FastAPI

import model_library.telemetry as telemetry
from model_library.base import LLM, dump_gateway_config, dump_llm_config

from model_gateway import model_helpers
from model_gateway.cache import ModelCache
from model_gateway.metrics import model_dimensions
from model_gateway.route_helpers import (
    EmbeddingModel,
    GatewayOperation,
    ModerationModel,
    dump_pydantic_response,
    ok_response,
)
from model_gateway.telemetry_helpers import dimension_telemetry_attributes
from model_gateway.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModerationRequest,
    ModerationResponse,
    ProviderError,
    TokenCountRequest,
    TokenCountResponse,
    UploadFileRequest,
    UploadFileResponse,
)


def register_provider_ops_routes(app: FastAPI, *, cache: ModelCache) -> None:
    @app.post("/tokens/count")
    async def count_tokens(body: TokenCountRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        display_config = dump_gateway_config(body.config)
        dimensions = model_dimensions(
            operation="tokens_count",
            model=body.model,
            config=display_config,
        )
        operation = GatewayOperation(
            operation="tokens_count",
            dimensions=dimensions,
            start=start,
            provider=model_helpers.provider_from_model(body.model),
        )
        base_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="tokens_count", model=body.model),
            **dimension_telemetry_attributes(dimensions),
        }
        operation.start_event(
            {
                **base_attrs,
                "gateway.input.count": len(body.inputs),
                "gateway.tool.count": len(body.tools),
            }
        )
        error_phase = "tokens_count"
        try:
            operation.add_event("model_cache_lookup")
            llm = model_helpers.get_cached_llm(cache, body, config=config)

            secret = getattr(app.state, "hmac_secret", b"")
            model_helpers.require_raw_input_secret(body.inputs, secret=secret or None)
            try:
                operation.add_event("restore_history_start")
                LLM.restore_raw_fields(body.inputs, secret=secret or None)
                operation.add_event("restore_history_done")
            except (KeyError, TypeError, ValueError) as exc:
                error_phase = "restore_history"
                raise ValueError(f"Invalid signed raw history blob: {exc}") from exc

            tokens_or_error = await operation.provider_call(
                llm.count_tokens(body.inputs, tools=body.tools),
                span_attrs=base_attrs,
            )
            if isinstance(tokens_or_error, ProviderError):
                return ok_response(TokenCountResponse(error=tokens_or_error))
            tokens = tokens_or_error
            operation.add_event(
                "provider_call_done", {"gen_ai.usage.input_tokens": tokens}
            )
            return operation.success(
                TokenCountResponse(tokens=tokens),
                extra_metrics={"InputTokens": (tokens, "Count")},
            )

        except Exception as exc:
            return operation.error(exc, phase=error_phase)

    @app.post("/files/upload")
    async def upload_file(body: UploadFileRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        display_config = dump_gateway_config(body.config)
        dimensions = model_dimensions(
            operation="files_upload",
            model=body.model,
            config=display_config,
            params={"mime": body.mime, "type": body.type},
        )
        operation = GatewayOperation(
            operation="files_upload",
            dimensions=dimensions,
            start=start,
            provider=model_helpers.provider_from_model(body.model),
        )
        base_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="files_upload", model=body.model),
            **dimension_telemetry_attributes(dimensions),
        }
        operation.start_event(
            {
                **base_attrs,
                "gateway.file.mime": body.mime,
                "gateway.file.type": body.type,
            }
        )
        try:
            operation.add_event("model_cache_lookup")
            llm = model_helpers.get_cached_llm(cache, body, config=config)
            try:
                file_bytes = io.BytesIO(
                    base64.b64decode(body.content_base64, validate=True)
                )
            except binascii.Error as exc:
                raise ValueError("Invalid content_base64") from exc
            uploaded_file_bytes = len(file_bytes.getvalue())
            operation.add_event(
                "decode_done",
                {"gateway.uploaded_file.bytes": uploaded_file_bytes},
            )
            file_or_error = await operation.provider_call(
                llm.upload_file(
                    body.name,
                    body.mime,
                    file_bytes,
                    type=body.type,
                ),
                span_attrs=base_attrs,
            )
            if isinstance(file_or_error, ProviderError):
                return ok_response(UploadFileResponse(error=file_or_error))
            file = file_or_error
            operation.add_event("provider_call_done")
            return operation.success(
                UploadFileResponse(file=file),
                extra_metrics={"UploadedFileBytes": (uploaded_file_bytes, "Bytes")},
            )

        except Exception as exc:
            return operation.error(exc, phase="files_upload")

    @app.post("/embeddings")
    async def embeddings(body: EmbeddingRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        display_config = dump_gateway_config(body.config)
        dimensions = model_dimensions(
            operation="embeddings",
            model=body.model,
            config=display_config,
            params={"embedding_model": body.embedding_model},
        )
        operation = GatewayOperation(
            operation="embeddings",
            dimensions=dimensions,
            start=start,
            provider=model_helpers.provider_from_model(body.model),
        )
        base_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="embeddings", model=body.model),
            **dimension_telemetry_attributes(dimensions),
        }
        operation.start_event(
            {**base_attrs, "gateway.embedding.model": body.embedding_model}
        )
        try:
            operation.add_event("model_cache_lookup")
            llm = model_helpers.get_cached_llm(cache, body, config=config)
            embedding_llm = cast(EmbeddingModel, llm)
            embedding_or_error = await operation.provider_call(
                embedding_llm.get_embedding(body.text, model=body.embedding_model),
                span_attrs=base_attrs,
            )
            if isinstance(embedding_or_error, ProviderError):
                return ok_response(EmbeddingResponse(error=embedding_or_error))
            embedding = embedding_or_error
            operation.add_event(
                "provider_call_done",
                {"gateway.embedding.dimension_count": len(embedding)},
            )
            return operation.success(
                EmbeddingResponse(embedding=embedding),
                extra_metrics={"EmbeddingDimensionCount": (len(embedding), "Count")},
            )

        except Exception as exc:
            return operation.error(exc, phase="embeddings")

    @app.post("/moderation")
    async def moderation(body: ModerationRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        display_config = dump_gateway_config(body.config)
        dimensions = model_dimensions(
            operation="moderation",
            model=body.model,
            config=display_config,
        )
        operation = GatewayOperation(
            operation="moderation",
            dimensions=dimensions,
            start=start,
            provider=model_helpers.provider_from_model(body.model),
        )
        base_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="moderation", model=body.model),
            **dimension_telemetry_attributes(dimensions),
        }
        operation.start_event(base_attrs)
        try:
            operation.add_event("model_cache_lookup")
            llm = model_helpers.get_cached_llm(cache, body, config=config)
            moderation_llm = cast(ModerationModel, llm)
            response_or_error = await operation.provider_call(
                moderation_llm.moderate_content(body.text),
                span_attrs=base_attrs,
            )
            if isinstance(response_or_error, ProviderError):
                return ok_response(ModerationResponse(error=response_or_error))
            response = response_or_error
            operation.add_event("provider_call_done")
            return operation.success(
                ModerationResponse(response=dump_pydantic_response(response))
            )

        except Exception as exc:
            return operation.error(exc, phase="moderation")
