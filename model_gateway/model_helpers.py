"""Model lookup and query helper functions for gateway routes."""

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from model_library import model_library_settings
from model_library.base import (
    LLM,
    ResolvedTokenRetryParams,
    TokenRetryParams,
    dump_llm_config,
    normalize_llm_config_for_model,
    resolve_token_retry_params,
)
from model_library.base.input import FileWithId, InputItem, RawInput, RawResponse
from model_library.registry_utils import get_registry_config, get_registry_model

from model_gateway.benchmark_admission_types import BenchmarkAcquireRequest
from model_gateway.cache import ModelCache
from model_gateway.types import GatewayRequestBase, QueryRequest, TokenCountRequest


def provider_from_model(model: str) -> str | None:
    provider = model.partition("/")[0]
    return provider or None


def get_cached_llm(
    cache: ModelCache,
    body: GatewayRequestBase,
    *,
    config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> LLM:
    effective_config = body.config if model_config is None else model_config
    return cache.get_or_create(
        body.model,
        config,
        lambda m, _c: get_registry_model(
            m, normalize_llm_config_for_model(m, effective_config)
        ),
    )


@dataclass(frozen=True)
class ManagedApiKey:
    source: Literal["pool_1", "pool_2"]
    key_setting: str


def managed_api_keys(model: str) -> tuple[ManagedApiKey, ...]:
    return ()


def _managed_api_key(
    body: QueryRequest | TokenCountRequest | BenchmarkAcquireRequest,
) -> str | None:
    if body.config.custom_api_key is not None or not isinstance(body, QueryRequest):
        return None
    if any(isinstance(item, FileWithId) for item in body.inputs):
        return None

    managed_keys = managed_api_keys(body.model)
    if not managed_keys:
        return None

    keys_by_source = {
        managed_key.source: getattr(model_library_settings, managed_key.key_setting)
        for managed_key in managed_keys
    }
    keys = tuple(keys_by_source.values())
    digest = hashlib.sha256(f"{body.run_id}{body.question_id}".encode()).digest()
    return keys[int.from_bytes(digest, "big") % len(keys)]


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
    effective_requests_per_minute = token_retry_params.requests_per_minute

    # Only consult the registry when a dimension still needs a configured
    # default; explicit params must resolve without a registry entry.
    if effective_token_limit is None or effective_requests_per_minute is None:
        registry_config = get_registry_config(model)
        if registry_config is None:
            raise ValueError(f"Model {model} not found in registry")

        rate_limit = registry_config.rate_limit
        if rate_limit is not None:
            effective_token_limit, effective_requests_per_minute = (
                rate_limit.apply_token_retry_defaults(
                    effective_token_limit,
                    effective_requests_per_minute,
                )
            )

    return resolve_token_retry_params(
        token_retry_params,
        effective_token_limit,
        effective_requests_per_minute,
    )


def get_gateway_llm(
    cache: ModelCache,
    body: QueryRequest | TokenCountRequest | BenchmarkAcquireRequest,
    *,
    resolved_token_retry_params: ResolvedTokenRetryParams | None = None,
) -> LLM:
    """Resolve the canonical gateway model identity without starting token work."""
    model_config = dump_llm_config(body.config)
    managed_api_key = _managed_api_key(body)
    if managed_api_key is not None:
        model_config["custom_api_key"] = managed_api_key

    cache_config = dict(model_config)
    if resolved_token_retry_params is not None:
        cache_config["__token_retry_params"] = resolved_token_retry_params.model_dump(
            mode="json"
        )

    return get_cached_llm(
        cache,
        body,
        config=cache_config,
        model_config=model_config,
    )


async def get_query_llm(
    cache: ModelCache,
    body: QueryRequest | TokenCountRequest | BenchmarkAcquireRequest,
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

    llm = get_gateway_llm(
        cache,
        body,
        resolved_token_retry_params=resolved_token_retry_params,
    )
    if token_retry_params is not None and resolved_token_retry_params is not None:
        await llm.ensure_resolved_token_retry(
            token_retry_params,
            resolved_token_retry_params,
        )
    return llm
