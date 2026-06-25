from model_gateway.cache import ModelCache


def test_model_cache_reuses_same_config_regardless_of_key_order():
    cache = ModelCache()
    created: list[str] = []

    def factory(model: str, config: dict[str, object]) -> object:
        created.append(model)
        return object()

    first = cache.get_or_create(
        "openai/gpt-4o-mini", {"temperature": 0, "max_tokens": 16}, factory
    )
    second = cache.get_or_create(
        "openai/gpt-4o-mini", {"max_tokens": 16, "temperature": 0}, factory
    )

    assert first is second
    assert created == ["openai/gpt-4o-mini"]


def test_model_cache_separates_different_configs():
    cache = ModelCache()

    first = cache.get_or_create(
        "openai/gpt-4o-mini", {"max_tokens": 16}, lambda *_: object()
    )
    second = cache.get_or_create(
        "openai/gpt-4o-mini", {"max_tokens": 64}, lambda *_: object()
    )

    assert first is not second


def test_model_cache_evicts_least_recently_used_entry():
    cache = ModelCache(maxsize=2)
    created: list[object] = []

    def factory(_model: str, _config: dict[str, object]) -> object:
        instance = object()
        created.append(instance)
        return instance

    first = cache.get_or_create("model/a", {}, factory)
    second = cache.get_or_create("model/b", {}, factory)
    assert cache.get_or_create("model/a", {}, factory) is first

    third = cache.get_or_create("model/c", {}, factory)
    assert third is not second
    assert cache.get_or_create("model/b", {}, factory) is not second
