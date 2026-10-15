import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_managed_settings_context: ContextVar[
    tuple[Mapping[str, str], frozenset[str]] | None
] = ContextVar("model_library_managed_settings", default=None)


class MissingManagedSettingError(AttributeError):
    """A managed runtime snapshot deliberately omitted a configured field."""


@contextmanager
def managed_settings_snapshot(
    values: Mapping[str, str], managed_fields: frozenset[str]
) -> Generator[None]:
    """Make one credential snapshot authoritative in the current context."""
    token = _managed_settings_context.set((values, managed_fields))
    try:
        yield
    finally:
        _managed_settings_context.reset(token)


class ModelLibrarySettings:
    _key_overrides: dict[str, str]

    def __init__(self):
        self._key_overrides = {}

    def set(self, **keys: str | bool):
        self._key_overrides.update({k.upper(): str(v) for k, v in keys.items()})

    def unset(self, key: str):
        self._key_overrides.pop(key.upper(), None)
        os.environ.pop(key.upper(), None)

    def reset(self):
        self._key_overrides = {}

    def get(self, name: str, default: Any = None) -> str | Any:
        try:
            return self.__getattr__(name)
        except AttributeError:
            return default

    def __getattr__(self, name: str) -> str:
        managed_snapshot = _managed_settings_context.get()
        if managed_snapshot is not None:
            values, managed_fields = managed_snapshot
            if name in managed_fields:
                if name in values:
                    return values[name]
                raise MissingManagedSettingError(f"Missing managed config key: {name}")

        # load key from override
        if name in self._key_overrides:
            return self._key_overrides[name]

        # load key from environment
        if name in os.environ:
            return os.environ[name]

        raise AttributeError(f"Missing config key: {name}")
