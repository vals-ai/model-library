"""Bounded LRU cache for LLM instances with idle TTL expiry."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")
DEFAULT_MODEL_CACHE_TTL_SECONDS = 60 * 60


class ModelCache:
    def __init__(
        self,
        maxsize: int = 128,
        ttl_seconds: float = DEFAULT_MODEL_CACHE_TTL_SECONDS,
    ):
        self._maxsize = maxsize
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[object, float]] = OrderedDict()

    @staticmethod
    def _make_key(model: str, config: dict[str, Any]) -> str:
        config_json = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        return f"{model}:{config_hash}"

    def get_or_create(
        self,
        model: str,
        config: dict[str, Any],
        factory: Callable[[str, dict[str, Any]], T],
    ) -> T:
        now = time.monotonic()
        self._prune_expired(now)
        key = self._make_key(model, config)
        entry = self._cache.get(key)
        if entry is not None:
            instance, _last_used = entry
            self._cache[key] = (instance, now)
            self._cache.move_to_end(key)
            return cast(T, instance)

        new_instance = factory(model, config)
        self._cache[key] = (new_instance, time.monotonic())
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return new_instance

    def _prune_expired(self, now: float) -> None:
        while self._cache:
            _instance, last_used = next(iter(self._cache.values()))
            if now - last_used < self._ttl_seconds:
                break
            self._cache.popitem(last=False)
