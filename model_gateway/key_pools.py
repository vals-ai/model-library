"""Managed provider key pools: which key settings each model may use.

The pools themselves are data: every task reads them from the `pools` AppConfig
profile (`model_gateway/key_pool_config.py`). This module holds their shape and
the lookup. Reading setting values is the caller's job.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate

from model_library.register_models import get_provider_registry


@dataclass(frozen=True)
class ModelRule:
    """Sends the models listed in `models`, by exact id, to `pool`."""

    pool: str
    models: frozenset[str]

    def matches(self, model: str) -> bool:
        return model in self.models


@dataclass(frozen=True)
class PoolConfig:
    """Key pools and the models routed to them.

    - `pools`: pool name to its ordered key setting names. `<provider>/default`
      holds the keys a provider's models use when no rule names them.
    - `rules`: ordered rules; the first match picks the model's pool, and a
      model no rule matches uses its provider's default pool.
    - `weights`: key setting name to its weight, default 1; a key's share of
      new picks is its weight over the pool's total.
    """

    pools: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict[str, tuple[str, ...]]
    )
    rules: tuple[ModelRule, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict[str, float])
    _bounds: dict[tuple[str, ...], tuple[int, ...]] = field(
        default_factory=dict[tuple[str, ...], tuple[int, ...]],
        init=False,
        repr=False,
        compare=False,
    )

    def weight_bounds(self, pool: tuple[str, ...]) -> tuple[int, ...]:
        """Cumulative whole-number weights of `pool`, worked out once per config.

        Weights are reduced to the smallest whole numbers in the same ratio, so
        equal weights give a `hash % len(pool)` pick. All-zero weights raise
        `KeyError`, like an empty pool.
        """
        if pool not in self._bounds:
            shares = [Fraction(self.weights.get(key, 1)) for key in pool]
            scale = math.lcm(*(share.denominator for share in shares))
            tickets = [int(share * scale) for share in shares]
            common = math.gcd(*tickets)
            if common == 0:
                raise KeyError(pool)
            self._bounds[pool] = tuple(accumulate(t // common for t in tickets))
        return self._bounds[pool]


def provider_of(model: str) -> str:
    return model.partition("/")[0]


def keys_for(config: PoolConfig, model: str) -> tuple[str, ...]:
    """The key settings that serve `model`, in order.

    Raises `KeyError` when a rule matches but its pool has no keys, so the
    query fails instead of silently using another pool.
    """
    for rule in config.rules:
        if rule.matches(model):
            keys = config.pools.get(rule.pool, ())
            if not keys:
                raise KeyError(rule.pool)
            return keys
    return config.pools.get(f"{provider_of(model)}/default", ())


def provider_key_settings() -> dict[str, str]:
    """Each provider with one default key, to that key's setting name."""
    return {
        provider: cls.api_key_setting
        for provider, cls in sorted(get_provider_registry().items())
        if cls.api_key_setting is not None
    }


def key_setting_provider(
    key_setting: str, key_settings: Mapping[str, str]
) -> str | None:
    """The provider a managed key setting belongs to, from its name.

    `key_settings` is `provider_key_settings()`, built once by the caller.
    """
    return None


def internal_key_settings() -> frozenset[str]:
    """Key settings reserved for `internal_only` models; code-owned, not operator-set."""
    return frozenset()
