"""Model lookup and query helper functions for gateway routes."""

from typing import Any

from model_library.base import (
    LLM,
    dump_gateway_config,
    dump_llm_config,
    normalize_llm_config_for_model,
)
from model_library.base.input import InputItem, RawInput, RawResponse
from model_library.registry_utils import (
    create_config,
    get_input_context_window_from_config,
    get_registry_config,
    get_registry_model,
)

from model_gateway.cache import ModelCache
from model_gateway.types import (
    GatewayRequestBase,
    ModelResolveRequest,
    ModelResolveResponse,
    QueryRequest,
)


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


def resolve_model(body: ModelResolveRequest) -> ModelResolveResponse:
    registry_config = get_registry_config(body.model)
    if registry_config is None:
        return ModelResolveResponse(exists=False, model=body.model)

    override_config = normalize_llm_config_for_model(body.model, body.config)
    effective_config = create_config(registry_config, override_config)
    return ModelResolveResponse(
        exists=True,
        model=body.model,
        effective_config=dump_gateway_config(effective_config),
        registry_config=registry_config.model_dump(mode="json"),
        input_context_window=get_input_context_window_from_config(registry_config),
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


def query_cache_config(body: QueryRequest) -> dict[str, Any]:
    config = dump_llm_config(body.config)
    if body.token_retry_params is None:
        return config

    return {
        **config,
        "__token_retry_params": body.token_retry_params.model_dump(mode="json"),
    }


async def get_query_llm(cache: ModelCache, body: QueryRequest) -> LLM:
    llm = cache.get_or_create(
        body.model,
        query_cache_config(body),
        lambda m, _c: get_registry_model(
            m, normalize_llm_config_for_model(body.model, body.config)
        ),
    )
    if (
        body.token_retry_params is not None
        and llm.token_retry_params != body.token_retry_params
    ):
        await llm.init_token_retry(body.token_retry_params)
    return llm
