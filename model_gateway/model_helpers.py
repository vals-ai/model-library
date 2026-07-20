"""Model lookup and query helper functions for gateway routes."""

from typing import Any

from model_library.base import (
    LLM,
    ResolvedTokenRetryParams,
    TokenRetryParams,
    dump_llm_config,
    normalize_llm_config_for_model,
    resolve_token_retry_params,
)
from model_library.base.input import InputItem, RawInput, RawResponse
from model_library.registry_utils import get_registry_model

from model_gateway.benchmark_admission_types import BenchmarkAcquireRequest
from model_gateway.cache import ModelCache
from model_gateway.types import GatewayRequestBase, QueryRequest


def provider_from_model(model: str) -> str | None:
    provider = model.partition("/")[0]
    return provider or None


def get_cached_llm(
    cache: ModelCache,
    body: GatewayRequestBase,
    *,
    config: dict[str, Any],
) -> LLM:
    return cache.get_or_create(
        body.model,
        config,
        lambda m, _c: get_registry_model(
            m, normalize_llm_config_for_model(m, body.config)
        ),
    )


def has_serialized_raw_blob(inputs: list[InputItem]) -> bool:
    return any(
        (isinstance(item, RawResponse) and isinstance(item.response, (str, dict)))
        or (isinstance(item, RawInput) and isinstance(item.input, (str, dict)))
        for item in inputs
    )


def require_raw_input_secret(inputs: list[InputItem], *, secret: bytes | None) -> None:
    if has_serialized_raw_blob(inputs) and not secret:
        raise ValueError(
            "MODEL_GATEWAY_HMAC_SECRET is required to accept raw history blobs"
        )


def resolve_gateway_token_retry_params(
    model: str,
    token_retry_params: TokenRetryParams,
) -> ResolvedTokenRetryParams:
    effective_token_limit = token_retry_params.limit
    return resolve_token_retry_params(
        token_retry_params,
        effective_token_limit,
    )


async def get_query_llm(
    cache: ModelCache,
    body: QueryRequest | BenchmarkAcquireRequest,
    *,
    resolved_token_retry_params: ResolvedTokenRetryParams | None = None,
) -> LLM:
    token_retry_params = body.token_retry_params
    if token_retry_params is None:
        resolved_token_retry_params = None
    elif resolved_token_retry_params is None:
        resolved_token_retry_params = resolve_gateway_token_retry_params(
            body.model,
            token_retry_params,
        )

    cache_config = dump_llm_config(body.config)
    if resolved_token_retry_params is not None:
        cache_config["__token_retry_params"] = resolved_token_retry_params.model_dump(
            mode="json"
        )

    llm = get_cached_llm(cache, body, config=cache_config)
    if token_retry_params is not None and resolved_token_retry_params is not None:
        await llm.ensure_resolved_token_retry(
            token_retry_params,
            resolved_token_retry_params,
        )
    return llm
