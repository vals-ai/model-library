"""Checks on operator-supplied model to pool mappings.

Operator mappings are `PoolConfig` rules that list exact model ids in
`models`. Each model must exist in the model registry, map to exactly one pool,
and land on a non-empty pool whose keys are all known keys of that model's
provider. Keys reserved for internal models serve only models whose registry
entry is `metadata.internal_only`; internal models may also use ordinary keys.
Every `<provider>/default` pool in `config.pools` is checked the same way as a
public model's pool, because default pools serve public models, and every
pooled provider must have one.

`PUT /admin/key-pools` runs it on every operator document. The seed document
`scripts/seed_key_pools.py` publishes is proven by its parity tests instead.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from model_gateway.key_pools import ModelRule, PoolConfig, provider_of

MappingErrorReason = Literal[
    "unknown_model",
    "unknown_pool",
    "empty_pool",
    "missing_default_pool",
    "cross_provider_pool",
    "unknown_key",
    "duplicate_mapping",
    "internal_key_for_public_model",
    "unknown_description",
]


class _RegistryMetadata(Protocol):
    @property
    def internal_only(self) -> bool: ...


class RegistryModel(Protocol):
    """The parts of a registry `ModelConfig` the checks read."""

    @property
    def provider_name(self) -> str: ...

    @property
    def metadata(self) -> _RegistryMetadata: ...


@dataclass(frozen=True)
class MappingError:
    """One rejected mapping. `model` is empty for a rule-level error."""

    model: str
    pool: str
    reason: MappingErrorReason


def validate_mapping(
    config: PoolConfig,
    *,
    registry: Mapping[str, RegistryModel],
    known_keys: Mapping[str, frozenset[str]],
    internal_keys: frozenset[str],
    pooled_providers: frozenset[str],
    described: Collection[str] = (),
) -> list[MappingError]:
    """The first problem found for each model and pool; an empty list means valid.

    Each pooled provider must keep a `<provider>/default` pool, so its unmapped
    models never fall back to a raw environment key.

    `known_keys` maps a provider name to the key setting names it owns.
    `internal_keys` holds the key setting names reserved for internal models.
    `pooled_providers` names the providers whose models must keep a
    `<provider>/default` pool, so a replacement cannot drop one silently.
    `described` names the document's descriptions; each must be a pool or a
    pooled key.
    """
    # Only pooled providers need a default pool: an unpooled provider such as
    # Alibaba resolves its own credentials.
    errors: list[MappingError] = [
        MappingError("", f"{provider}/default", "missing_default_pool")
        for provider in sorted(pooled_providers)
        if f"{provider}/default" not in config.pools
    ]
    all_keys = frozenset[str]().union(*known_keys.values())
    for pool in sorted(config.pools):
        if not pool.endswith("/default"):
            continue
        reason = _default_pool_problem(
            config.pools[pool],
            _has_serving_key(config, pool),
            known_keys.get(provider_of(pool), frozenset()),
            all_keys,
            internal_keys,
        )
        if reason is not None:
            errors.append(MappingError("", pool, reason))
    mapped: set[str] = set()
    for rule in config.rules:
        for model in sorted(rule.models):
            if model in mapped:
                errors.append(MappingError(model, rule.pool, "duplicate_mapping"))
                continue
            mapped.add(model)
            reason = _mapping_problem(
                config,
                model,
                rule,
                registry,
                known_keys,
                all_keys,
                internal_keys,
            )
            if reason is not None:
                errors.append(MappingError(model, rule.pool, reason))
    # A pool no rule reaches today still must not name a setting that is not a key.
    reported = {error.pool for error in errors}
    errors += [
        MappingError("", pool, "unknown_key")
        for pool, keys in sorted(config.pools.items())
        if pool not in reported and any(key not in all_keys for key in keys)
    ]
    # A weight may only name a key some pool holds.
    pooled = {key for keys in config.pools.values() for key in keys}
    errors += [
        MappingError("", "", "unknown_key")
        for key in sorted(config.weights)
        if key not in pooled
    ]
    errors += [
        MappingError("", name, "unknown_description")
        for name in sorted(described)
        if name not in pooled and name not in config.pools
    ]
    return errors


def _mapping_problem(
    config: PoolConfig,
    model: str,
    rule: ModelRule,
    registry: Mapping[str, RegistryModel],
    known_keys: Mapping[str, frozenset[str]],
    all_keys: frozenset[str],
    internal_keys: frozenset[str],
) -> MappingErrorReason | None:
    entry = registry.get(model)
    if entry is None:
        return "unknown_model"
    pool = rule.pool
    if pool not in config.pools:
        return "unknown_pool"
    if not _has_serving_key(config, pool):
        return "empty_pool"
    return _key_problem(
        config.pools[pool],
        known_keys.get(entry.provider_name, frozenset()),
        all_keys,
        internal_keys,
        internal_ok=entry.metadata.internal_only,
    )


def _has_serving_key(config: PoolConfig, pool: str) -> bool:
    """Whether a key in the pool can be picked: a weight-0 key takes no new attempts."""
    return any(config.weights.get(key, 1) > 0 for key in config.pools[pool])


def _default_pool_problem(
    keys: tuple[str, ...],
    serving: bool,
    own_keys: frozenset[str],
    all_keys: frozenset[str],
    internal_keys: frozenset[str],
) -> MappingErrorReason | None:
    if not serving:
        return "empty_pool"
    return _key_problem(keys, own_keys, all_keys, internal_keys, internal_ok=False)


def _key_problem(
    keys: tuple[str, ...],
    own_keys: frozenset[str],
    all_keys: frozenset[str],
    internal_keys: frozenset[str],
    *,
    internal_ok: bool,
) -> MappingErrorReason | None:
    foreign = next((key for key in keys if key not in own_keys), None)
    if foreign is not None:
        return "cross_provider_pool" if foreign in all_keys else "unknown_key"
    if not internal_ok and internal_keys.intersection(keys):
        return "internal_key_for_public_model"
    return None
