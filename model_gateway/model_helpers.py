"""Model lookup and query helper functions for gateway routes."""

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, TypeGuard

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
from model_library.base.query_ids import prompt_identity
from model_library.registry_utils import (
    get_registry_config,
    get_registry_model,
    get_transcription_model,
)

from model_gateway.benchmark_admission_types import BenchmarkAcquireRequest
from model_gateway.cache import ModelCache
from model_gateway.errors import AssignedKeyUnavailableError
from model_gateway.key_assignments import AssignmentStore, AttemptKey
from model_gateway.key_pools import PoolConfig, keys_for, provider_of
from model_gateway.runtime_config import runtime_cache_namespace, runtime_pool_snapshot
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
    transcription: bool = False,
) -> LLM:
    effective_config = body.config if model_config is None else model_config
    build = get_transcription_model if transcription else get_registry_model
    return cache.get_or_create(
        body.model,
        config,
        lambda m, _c: build(m, normalize_llm_config_for_model(m, effective_config)),
        namespace=runtime_cache_namespace(),
    )


@dataclass(frozen=True)
class ManagedApiKey:
    source: str
    key_setting: str


def current_pool_config() -> PoolConfig:
    """The pool config this request runs with, from the `pools` AppConfig profile.

    Without AppConfig (a local Gateway) there are no pools, and model-library
    resolves every key itself.
    """
    snapshot = runtime_pool_snapshot()
    if snapshot is None:
        return PoolConfig()
    return snapshot.config


def current_pool_revision() -> int | None:
    """The `revision` of the pools document active for this request; None without AppConfig."""
    snapshot = runtime_pool_snapshot()
    return None if snapshot is None else snapshot.document.revision


def managed_api_keys(model: str) -> tuple[ManagedApiKey, ...]:
    """The keys that serve `model`, labelled `pool_1`, `pool_2`, ... by position.

    A model whose matched pool is empty has no managed keys here; its query
    fails in `_managed_api_key`.
    """
    config = current_pool_config()
    try:
        key_settings = keys_for(config, model)
    except KeyError:
        return ()
    return tuple(
        ManagedApiKey(f"pool_{index}", key_setting)
        for index, key_setting in enumerate(key_settings, start=1)
    )


def _uses_managed_key(
    body: QueryRequest | TokenCountRequest | BenchmarkAcquireRequest,
) -> TypeGuard[QueryRequest]:
    return (
        body.config.custom_api_key is None
        and isinstance(body, QueryRequest)
        and not any(isinstance(item, FileWithId) for item in body.inputs)
    )


def _pick_identity(body: QueryRequest) -> str:
    run_id, question_id = prompt_identity(
        model_name=body.model,
        run_id=body.run_id,
        question_id=body.question_id,
        input=body.inputs,
        tools=body.tools,
    )
    if body.attempt_id is None:
        return f"{run_id}{question_id}"
    # A JSON list keeps ("ab", "c", "d") and ("a", "bc", "d") apart.
    parts = [run_id, question_id, body.attempt_id]
    return json.dumps(parts, separators=(",", ":"))


def _pick_managed_key_setting(body: QueryRequest, config: PoolConfig) -> str | None:
    """Hash the request's identity into the model's pool, each key taking its weight's share.

    The identity is run, question and attempt, so a retry draws its key afresh;
    a request with no `attempt_id` hashes run and question as before pools.
    `PoolConfig.weight_bounds` holds each key's share.
    """
    pool = keys_for(config, body.model)
    if not pool:
        return None
    bounds = config.weight_bounds(pool)
    digest = hashlib.sha256(_pick_identity(body).encode()).digest()
    return pool[bisect_right(bounds, int.from_bytes(digest, "big") % bounds[-1])]


def _managed_api_key(
    body: QueryRequest | TokenCountRequest | BenchmarkAcquireRequest,
    *,
    key_setting: str | None = None,
) -> str | None:
    if not _uses_managed_key(body):
        return None
    if key_setting is None:
        key_setting = _pick_managed_key_setting(body, current_pool_config())
    if key_setting is None:
        return None
    return getattr(model_library_settings, key_setting)


def _assignment_key(body: QueryRequest) -> AttemptKey | None:
    """The attempt a request belongs to, scoped by its run.

    Run tokens stamp `run_id`, so a run-token caller reaches only its own run's
    attempts. A request without both `run_id` and `attempt_id` belongs to no
    attempt. The store encodes the parts, so ("ab", "c") and ("a", "bc") stay
    apart.
    """
    if body.run_id is None or body.attempt_id is None:
        return None
    return ("attempt", body.run_id, body.attempt_id)


def _provider_has_pools(model: str, config: PoolConfig) -> bool:
    """Whether the pools document names any pool for `model`'s provider.

    Only these models can hold a saved key, so the rest never read the store.
    """
    prefix = f"{provider_of(model)}/"
    return any(pool.startswith(prefix) for pool in config.pools)


async def assigned_key_setting(
    body: QueryRequest, store: AssignmentStore
) -> str | None:
    """The managed key setting that serves this request.

    The request picks with the same hash a Gateway without the table uses; it saves
    that pick unless it already holds a key, and the held key wins. When no pool
    serves the model any more, an attempt still reads the key it holds. A request
    that names no attempt keeps its pick unsaved. None means no managed key
    applies; a key with no value fails the request.
    """
    if not _uses_managed_key(body):
        return None
    config = current_pool_config()
    picked = _pick_managed_key_setting(body, config)
    attempt_key = _assignment_key(body)
    if picked is None:
        if attempt_key is None or not _provider_has_pools(body.model, config):
            return None
        key_setting = await store.get(attempt_key, body.model)
    elif attempt_key is None:
        key_setting = picked
    else:
        key_setting = await store.put_if_absent(attempt_key, body.model, picked)
    if key_setting is None:
        return None
    if not model_library_settings.get(key_setting):
        raise AssignedKeyUnavailableError(
            key_name=key_setting, attempt_key=attempt_key, model=body.model
        )
    return key_setting


def has_serialized_raw_blob(inputs: list[InputItem]) -> bool:
    return any(
        (isinstance(item, RawResponse) and isinstance(item.response, (str, dict)))
        or (isinstance(item, RawInput) and LLM.is_raw_input_blob(item.input))
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
    managed_key_setting: str | None = None,
) -> LLM:
    """Resolve the canonical gateway model identity without starting token work.

    `managed_key_setting` is an attempt's assigned key; None picks by hash.
    """
    model_config = dump_llm_config(body.config)
    managed_api_key = _managed_api_key(body, key_setting=managed_key_setting)
    if managed_api_key is not None:
        model_config["custom_api_key"] = managed_api_key

    # The cache key adds the retry params only when there are some.
    cache_config = model_config
    if resolved_token_retry_params is not None:
        cache_config = {
            **model_config,
            "__token_retry_params": resolved_token_retry_params.model_dump(mode="json"),
        }

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
    managed_key_setting: str | None = None,
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
        managed_key_setting=managed_key_setting,
    )
    if token_retry_params is not None and resolved_token_retry_params is not None:
        await llm.ensure_resolved_token_retry(
            token_retry_params,
            resolved_token_retry_params,
        )
    return llm
