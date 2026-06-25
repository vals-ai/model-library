"""Provider-operation route registration."""

import base64
import binascii
import io
import time
from typing import cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import model_library.telemetry as telemetry
from model_library.base import dump_llm_config

from model_gateway.cache import ModelCache
from model_gateway.errors import map_exception_to_error
from model_gateway import model_helpers
from model_gateway.metrics import emit_model_error, emit_model_success, model_dimensions
from model_gateway.route_helpers import (
    EmbeddingModel,
    ModerationModel,
    dump_pydantic_response,
    ok_response,
    provider_call_or_error,
)
from model_gateway.telemetry_helpers import (
    dimension_telemetry_attributes,
    error_telemetry_attributes,
    latency_ms,
)
from model_gateway.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModerationRequest,
    ModerationResponse,
    ProviderError,
    UploadFileRequest,
    UploadFileResponse,
)


def register_provider_ops_routes(app: FastAPI, *, cache: ModelCache) -> None:
    @app.post("/files/upload")
    async def upload_file(body: UploadFileRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        dimensions = model_dimensions(
            operation="files_upload",
            model=body.model,
            config=config,
            params={"mime": body.mime, "type": body.type},
        )
        upload_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="files_upload", model=body.model),
            **dimension_telemetry_attributes(dimensions),
            "gateway.file.mime": body.mime,
            "gateway.file.type": body.type,
        }
        telemetry.set_attributes(upload_attrs)
        telemetry.add_event("gateway.files_upload.start")
        try:
            telemetry.add_event("gateway.files_upload.model_cache_lookup")
            llm = cache.get_or_create(
                body.model,
                config,
                lambda m, _c: model_helpers.get_registry_model(
                    m, model_helpers.normalize_llm_config_for_model(m, body.config)
                ),
            )
            try:
                file_bytes = io.BytesIO(
                    base64.b64decode(body.content_base64, validate=True)
                )
            except binascii.Error as exc:
                raise ValueError("Invalid content_base64") from exc
            uploaded_file_bytes = len(file_bytes.getvalue())
            telemetry.add_event(
                "gateway.files_upload.decode_done",
                {"gateway.uploaded_file.bytes": uploaded_file_bytes},
            )
            telemetry.add_event("gateway.files_upload.provider_call_start")
            with telemetry.start_span(
                "gateway.files_upload.provider_call",
                {
                    **telemetry.model_attributes(
                        operation="files_upload", model=body.model
                    ),
                    **dimension_telemetry_attributes(dimensions),
                },
                kind="client",
            ):
                file_or_error = await provider_call_or_error(
                    llm.upload_file(
                        body.name,
                        body.mime,
                        file_bytes,
                        type=body.type,
                    ),
                    provider=body.model.partition("/")[0] or None,
                    operation="files_upload",
                    dimensions=dimensions,
                    start=start,
                )
            if isinstance(file_or_error, ProviderError):
                return ok_response(UploadFileResponse(error=file_or_error))
            file = file_or_error
            telemetry.add_event("gateway.files_upload.provider_call_done")
            emit_model_success(
                dimensions,
                latency_ms=latency_ms(start),
                extra_metrics={"UploadedFileBytes": (uploaded_file_bytes, "Bytes")},
            )
            telemetry.set_attributes({"gateway.latency_ms": latency_ms(start)})
            telemetry.set_status_ok()
            telemetry.add_event("gateway.files_upload.success")
            return ok_response(UploadFileResponse(file=file))

        except Exception as exc:
            err = map_exception_to_error(
                exc, provider=body.model.partition("/")[0] or None
            )
            error_attrs = error_telemetry_attributes(err, phase="files_upload")
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error(err.body.code)
            telemetry.add_event("gateway.files_upload.error", error_attrs)
            emit_model_error(
                dimensions,
                error_code=err.body.code,
                latency_ms=latency_ms(start),
            )
            return JSONResponse(
                status_code=err.status_code, content=err.body.model_dump()
            )

    @app.post("/embeddings")
    async def embeddings(body: EmbeddingRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        dimensions = model_dimensions(
            operation="embeddings",
            model=body.model,
            config=config,
            params={"embedding_model": body.embedding_model},
        )
        embedding_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="embeddings", model=body.model),
            **dimension_telemetry_attributes(dimensions),
            "gateway.embedding.model": body.embedding_model,
        }
        telemetry.set_attributes(embedding_attrs)
        telemetry.add_event("gateway.embeddings.start")
        try:
            telemetry.add_event("gateway.embeddings.model_cache_lookup")
            llm = cache.get_or_create(
                body.model,
                config,
                lambda m, _c: model_helpers.get_registry_model(
                    m, model_helpers.normalize_llm_config_for_model(m, body.config)
                ),
            )
            embedding_llm = cast(EmbeddingModel, llm)
            telemetry.add_event("gateway.embeddings.provider_call_start")
            with telemetry.start_span(
                "gateway.embeddings.provider_call",
                {
                    **telemetry.model_attributes(
                        operation="embeddings", model=body.model
                    ),
                    **dimension_telemetry_attributes(dimensions),
                },
                kind="client",
            ):
                embedding_or_error = await provider_call_or_error(
                    embedding_llm.get_embedding(body.text, model=body.embedding_model),
                    provider=body.model.partition("/")[0] or None,
                    operation="embeddings",
                    dimensions=dimensions,
                    start=start,
                )
            if isinstance(embedding_or_error, ProviderError):
                return ok_response(EmbeddingResponse(error=embedding_or_error))
            embedding = embedding_or_error
            telemetry.add_event(
                "gateway.embeddings.provider_call_done",
                {"gateway.embedding.dimension_count": len(embedding)},
            )
            emit_model_success(
                dimensions,
                latency_ms=latency_ms(start),
                extra_metrics={"EmbeddingDimensionCount": (len(embedding), "Count")},
            )
            telemetry.set_attributes({"gateway.latency_ms": latency_ms(start)})
            telemetry.set_status_ok()
            telemetry.add_event("gateway.embeddings.success")
            return ok_response(EmbeddingResponse(embedding=embedding))

        except Exception as exc:
            err = map_exception_to_error(
                exc, provider=body.model.partition("/")[0] or None
            )
            error_attrs = error_telemetry_attributes(err, phase="embeddings")
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error(err.body.code)
            telemetry.add_event("gateway.embeddings.error", error_attrs)
            emit_model_error(
                dimensions,
                error_code=err.body.code,
                latency_ms=latency_ms(start),
            )
            return JSONResponse(
                status_code=err.status_code, content=err.body.model_dump()
            )

    @app.post("/moderation")
    async def moderation(body: ModerationRequest):
        start = time.perf_counter()
        config = dump_llm_config(body.config)
        dimensions = model_dimensions(
            operation="moderation",
            model=body.model,
            config=config,
        )
        moderation_attrs: dict[str, object | None] = {
            **telemetry.model_attributes(operation="moderation", model=body.model),
            **dimension_telemetry_attributes(dimensions),
        }
        telemetry.set_attributes(moderation_attrs)
        telemetry.add_event("gateway.moderation.start")
        try:
            telemetry.add_event("gateway.moderation.model_cache_lookup")
            llm = cache.get_or_create(
                body.model,
                config,
                lambda m, _c: model_helpers.get_registry_model(
                    m, model_helpers.normalize_llm_config_for_model(m, body.config)
                ),
            )
            moderation_llm = cast(ModerationModel, llm)
            telemetry.add_event("gateway.moderation.provider_call_start")
            with telemetry.start_span(
                "gateway.moderation.provider_call",
                {
                    **telemetry.model_attributes(
                        operation="moderation", model=body.model
                    ),
                    **dimension_telemetry_attributes(dimensions),
                },
                kind="client",
            ):
                response_or_error = await provider_call_or_error(
                    moderation_llm.moderate_content(body.text),
                    provider=body.model.partition("/")[0] or None,
                    operation="moderation",
                    dimensions=dimensions,
                    start=start,
                )
            if isinstance(response_or_error, ProviderError):
                return ok_response(ModerationResponse(error=response_or_error))
            response = response_or_error
            telemetry.add_event("gateway.moderation.provider_call_done")
            emit_model_success(dimensions, latency_ms=latency_ms(start))
            moderation_response_attrs: dict[str, object | None] = {
                "gateway.latency_ms": latency_ms(start)
            }
            telemetry.set_attributes(moderation_response_attrs)
            telemetry.set_status_ok()
            telemetry.add_event("gateway.moderation.success")
            return ok_response(
                ModerationResponse(response=dump_pydantic_response(response))
            )

        except Exception as exc:
            err = map_exception_to_error(
                exc, provider=body.model.partition("/")[0] or None
            )
            error_attrs = error_telemetry_attributes(err, phase="moderation")
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error(err.body.code)
            telemetry.add_event("gateway.moderation.error", error_attrs)
            emit_model_error(
                dimensions,
                error_code=err.body.code,
                latency_ms=latency_ms(start),
            )
            return JSONResponse(
                status_code=err.status_code, content=err.body.model_dump()
            )
