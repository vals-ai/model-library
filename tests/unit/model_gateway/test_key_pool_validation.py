"""Checks on operator-supplied model to pool mappings."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from model_gateway.key_pool_validation import (
    MappingError,
    RegistryModel,
    validate_mapping,
)
from model_gateway.key_pool_config import key_catalog, pool_config_from_document
from model_gateway.key_pools import (
    ModelRule,
    PoolConfig,
    keys_for,
)
from model_library.register_models import get_model_registry


@dataclass(frozen=True)
class _FakeMetadata:
    internal_only: bool = False


@dataclass(frozen=True)
class _FakeModel:
    provider_name: str
    metadata: _FakeMetadata = field(default_factory=_FakeMetadata)


_REGISTRY = {
    "anthropic/claude-a": _FakeModel("anthropic"),
    "anthropic/claude-b": _FakeModel("anthropic"),
    "meta/llama-a": _FakeModel("meta"),
    "anthropic/claude-internal": _FakeModel(
        "anthropic", _FakeMetadata(internal_only=True)
    ),
}
_KNOWN_KEYS = {
    "anthropic": frozenset(
        {"ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_EAP_9_18"}
    ),
    "meta": frozenset({"META_API_KEY"}),
}
_INTERNAL_KEYS = frozenset({"ANTHROPIC_API_KEY_EAP_9_18"})


_CLEAN_DEFAULTS = {
    "anthropic/default": ("ANTHROPIC_API_KEY",),
    "meta/default": ("META_API_KEY",),
}


def _config(*rules: ModelRule, **pools: tuple[str, ...]) -> PoolConfig:
    """Rules plus pools, with a clean default pool for each provider unless given."""
    return PoolConfig(pools={**_CLEAN_DEFAULTS, **pools}, rules=rules)


def _validate(
    config: PoolConfig, registry: Mapping[str, RegistryModel] = _REGISTRY
) -> list[MappingError]:
    return validate_mapping(
        config,
        registry=registry,
        known_keys=_KNOWN_KEYS,
        internal_keys=_INTERNAL_KEYS,
        pooled_providers=frozenset({"anthropic"}),
    )


def test_unknown_model_is_rejected() -> None:
    config = _config(
        ModelRule("second", models=frozenset({"anthropic/claude-made-up"})),
        second=("ANTHROPIC_API_KEY_2",),
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-made-up", "second", "unknown_model")
    ]


def test_anthropic_model_on_a_meta_key_is_rejected() -> None:
    config = _config(
        ModelRule("mixed", models=frozenset({"anthropic/claude-a"})),
        mixed=("ANTHROPIC_API_KEY", "META_API_KEY"),
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "mixed", "cross_provider_pool")
    ]


def test_pool_naming_an_unknown_key_is_rejected() -> None:
    config = _config(
        ModelRule("typo", models=frozenset({"anthropic/claude-a"})),
        typo=("ANTHROPIC_API_KEY_TYPO",),
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "typo", "unknown_key")
    ]


def test_rule_naming_an_undeclared_pool_is_rejected() -> None:
    config = _config(ModelRule("missing", models=frozenset({"anthropic/claude-a"})))

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "missing", "unknown_pool")
    ]


def test_same_model_mapped_twice_is_rejected() -> None:
    config = _config(
        ModelRule("first", models=frozenset({"anthropic/claude-a"})),
        ModelRule("second", models=frozenset({"anthropic/claude-a"})),
        first=("ANTHROPIC_API_KEY",),
        second=("ANTHROPIC_API_KEY_2",),
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "second", "duplicate_mapping")
    ]


def test_rule_pointing_at_an_empty_pool_is_rejected() -> None:
    config = _config(
        ModelRule("empty", models=frozenset({"anthropic/claude-a"})), empty=()
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "empty", "empty_pool")
    ]


def test_public_model_on_an_internal_key_is_rejected() -> None:
    config = _config(
        ModelRule("eap", models=frozenset({"anthropic/claude-a"})),
        eap=("ANTHROPIC_API_KEY_EAP_9_18",),
    )

    assert _validate(config) == [
        MappingError("anthropic/claude-a", "eap", "internal_key_for_public_model")
    ]


def test_internal_model_on_the_default_pool_passes() -> None:
    config = _config(
        ModelRule("anthropic/default", models=frozenset({"anthropic/claude-internal"})),
        **{"anthropic/default": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2")},
    )

    assert _validate(config) == []


def test_default_pool_with_an_internal_key_is_rejected() -> None:
    config = PoolConfig(
        pools={"anthropic/default": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_EAP_9_18")}
    )

    assert _validate(config) == [
        MappingError("", "anthropic/default", "internal_key_for_public_model")
    ]


def test_default_pool_with_another_providers_key_is_rejected() -> None:
    config = PoolConfig(pools={"anthropic/default": ("ANTHROPIC_API_KEY", "META_API_KEY")})

    assert _validate(config) == [
        MappingError("", "anthropic/default", "cross_provider_pool")
    ]


def test_default_pool_with_an_unknown_key_is_rejected() -> None:
    config = PoolConfig(pools={"anthropic/default": ("ANTHROPIC_API_KEY_TYPO",)})

    assert _validate(config) == [
        MappingError("", "anthropic/default", "unknown_key")
    ]


def test_empty_default_pool_is_rejected() -> None:
    config = PoolConfig(pools={"anthropic/default": ()})

    assert _validate(config) == [MappingError("", "anthropic/default", "empty_pool")]


def test_clean_default_pool_passes() -> None:
    config = PoolConfig(
        pools={
            "anthropic/default": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"),
            "meta/default": ("META_API_KEY",),
        }
    )

    assert _validate(config) == []


_REACHABLE_CONFIGS = {
    "exact": _config(
        ModelRule("second", models=frozenset({"anthropic/claude-a"})),
        second=("ANTHROPIC_API_KEY_2",),
    ),
    "internal": _config(
        ModelRule("eap", models=frozenset({"anthropic/claude-internal"})),
        eap=("ANTHROPIC_API_KEY_EAP_9_18",),
    ),
}


@pytest.mark.parametrize("name", sorted(_REACHABLE_CONFIGS))
def test_accepted_rules_are_the_pool_keys_for_returns(name: str) -> None:
    config = _REACHABLE_CONFIGS[name]
    assert _validate(config) == []

    for rule in config.rules:
        for model in rule.models:
            assert keys_for(config, model) == config.pools[rule.pool]


def test_pooled_provider_without_a_default_pool_is_rejected() -> None:
    config = PoolConfig(pools={"second": ("ANTHROPIC_API_KEY_2",)})

    errors = validate_mapping(
        config,
        registry=_REGISTRY,
        known_keys=_KNOWN_KEYS,
        internal_keys=_INTERNAL_KEYS,
        pooled_providers=frozenset({"anthropic"}),
    )

    assert errors == [MappingError("", "anthropic/default", "missing_default_pool")]


def test_real_registry_accepts_a_real_model_and_rejects_a_made_up_one() -> None:
    config = _config(
        ModelRule(
            "second",
            models=frozenset(
                {"anthropic/claude-haiku-5-5", "anthropic/claude-made-up-0-0"}
            ),
        ),
        second=("ANTHROPIC_API_KEY_2",),
    )

    assert _validate(config, get_model_registry()) == [MappingError("anthropic/claude-made-up-0-0", "second", "unknown_model")]
def test_a_weight_for_a_key_no_pool_holds_is_refused() -> None:
    config = PoolConfig(
        pools=_CLEAN_DEFAULTS, weights={"ANTHROPIC_API_KEY": 1, "META_API_KEY_9": 2}
    )

    assert _validate(config) == [MappingError("", "", "unknown_key")]
