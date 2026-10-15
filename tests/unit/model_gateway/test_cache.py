import model_gateway.cache as cache_module
from model_gateway.cache import ModelCache


def test_model_cache_reuses_same_config_regardless_of_key_order():
    cache = ModelCache()
    created: list[str] = []

    def factory(model: str, config: dict[str, object]) -> object:
        created.append(model)
        return object()

    first = cache.get_or_create(
        "openai/gpt-4o-mini",
        {"temperature": 0, "max_tokens": 16},
        factory,
        namespace="test",
    )
    second = cache.get_or_create(
        "openai/gpt-4o-mini",
        {"max_tokens": 16, "temperature": 0},
        factory,
        namespace="test",
    )

    assert first is second
    assert created == ["openai/gpt-4o-mini"]


def test_model_cache_separates_different_configs():
    cache = ModelCache()

    first = cache.get_or_create(
        "openai/gpt-4o-mini", {"max_tokens": 16}, lambda *_: object(), namespace="test"
    )
    second = cache.get_or_create(
        "openai/gpt-4o-mini", {"max_tokens": 64}, lambda *_: object(), namespace="test"
    )

    assert first is not second


def test_model_cache_evicts_least_recently_used_entry():
    cache = ModelCache(maxsize=2)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory, namespace="test")
    second = cache.get_or_create("model/b", {}, factory, namespace="test")
    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first

    cache.get_or_create("model/c", {}, factory, namespace="test")
    assert cache.get_or_create("model/b", {}, factory, namespace="test") is not second


def test_model_cache_expires_idle_entries(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now)
    cache = ModelCache(ttl_seconds=60)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory, namespace="test")
    now += 60
    second = cache.get_or_create("model/a", {}, factory, namespace="test")

    assert second is not first
    assert created == [first, second]


def test_model_cache_ttl_extends_on_access(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now)
    cache = ModelCache(ttl_seconds=60)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory, namespace="test")
    now += 59
    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first
    now += 59
    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first

    assert created == [first]


def test_model_cache_starts_ttl_after_factory_returns(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now)
    cache = ModelCache(ttl_seconds=60)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        nonlocal now
        now += 61
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory, namespace="test")

    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first
    assert created == [first]


def test_model_cache_prunes_expired_lru_entries_without_dropping_refreshed_entries(
    monkeypatch,
):
    now = 1000.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now)
    cache = ModelCache(maxsize=3, ttl_seconds=60)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory, namespace="test")
    second = cache.get_or_create("model/b", {}, factory, namespace="test")
    now += 59
    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first

    now += 2
    cache.get_or_create("model/c", {}, factory, namespace="test")

    assert cache.get_or_create("model/a", {}, factory, namespace="test") is first
    assert cache.get_or_create("model/b", {}, factory, namespace="test") is not second
