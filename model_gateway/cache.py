"""Bounded LRU cache for LLM instances."""

import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ModelCache:
    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, object] = OrderedDict()

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
        key = self._make_key(model, config)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]  # pyright: ignore[reportReturnType]

        instance = factory(model, config)
        self._cache[key] = instance
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return instance
