"""Default key pools built from the key settings a unit test sets."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import model_gateway.model_helpers as model_helpers
from model_gateway.key_pool_config import PublishedPoolDocument
from model_gateway.key_pools import PoolConfig, provider_key_settings

_NUMBERED_PROVIDERS = ("anthropic", "meta")


def settings_pool_config(get_setting: Callable[[str], object]) -> PoolConfig:
    """Default pools for the key settings `get_setting` holds."""
    pools: dict[str, tuple[str, ...]] = {}
    for provider, key_setting in provider_key_settings().items():
        keys = [key_setting] if get_setting(key_setting) else []
        if provider in _NUMBERED_PROVIDERS:
            while keys and get_setting(f"{key_setting}_{len(keys) + 1}"):
                keys.append(f"{key_setting}_{len(keys) + 1}")
        if keys:
            pools[f"{provider}/default"] = tuple(keys)
    return PoolConfig(pools=pools)


def settings_pool_document(
    get_setting: Callable[[str], object], revision: int = 0
) -> PublishedPoolDocument:
    """`settings_pool_config` as a published document at `revision`."""
    pools = settings_pool_config(get_setting).pools
    return PublishedPoolDocument(
        pools={pool: list(keys) for pool, keys in pools.items()},
        model_pools=[],
        weights={},
        revision=revision,
    )


@contextmanager
def settings_pools() -> Iterator[None]:
    """Serve `settings_pool_config` for the settings each call reads."""

    def current() -> PoolConfig:
        return settings_pool_config(model_helpers.model_library_settings.get)

    with patch.object(model_helpers, "current_pool_config", current):
        yield
