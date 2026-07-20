from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from inspect import signature
from threading import Event, Lock
from unittest.mock import MagicMock, patch

import pytest
import yaml

import model_library
import model_library.register_models as register_models
from model_library.register_models import ModelRegistry, get_model_registry


@pytest.fixture(autouse=True)
def reset_model_registry_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(register_models, "_model_registry", None)
    monkeypatch.setattr(
        register_models, "_model_registry_refreshed_at", None, raising=False
    )


def _registry() -> dict[str, object]:
    return {}


class LocalSettings:
    def get(self, _name: str, default: str | None = None) -> str | None:
        return default


class GatewaySettings:
    def __init__(self, url: str = "https://gateway.test/") -> None:
        self.gateway_url = url

    @property
    def MODEL_GATEWAY_URL(self) -> str:
        return self.gateway_url

    def get(self, name: str, default: str | None = None) -> str | None:
        match name:
            case "MODEL_GATEWAY_URL":
                return self.gateway_url
            case "MODEL_GATEWAY_API_KEY":
                return "sk-gateway"
            case _:
                return default


def patch_gateway_fetch(monkeypatch: pytest.MonkeyPatch, fetch) -> None:
    monkeypatch.setattr(register_models, "fetch_gateway_model_registry", fetch)


def _refresh_and_get(
    *, refresh_ttl: timedelta, allow_stale_on_error: bool = False
) -> ModelRegistry:
    assert (
        register_models.refresh_model_registry(
            refresh_ttl=refresh_ttl,
            allow_stale_on_error=allow_stale_on_error,
        )
        is None
    )
    return get_model_registry()


def test_no_argument_gateway_registry_remains_process_lifetime_singleton(
    monkeypatch: pytest.MonkeyPatch,
):
    first_registry = _registry()
    second_registry = _registry()
    mock_fetch = MagicMock(side_effect=[first_registry, second_registry])
    patch_gateway_fetch(monkeypatch, mock_fetch)

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert get_model_registry() is first_registry
        assert get_model_registry() is first_registry

    assert signature(get_model_registry).parameters == {}
    assert mock_fetch.call_count == 1


def test_refresh_uses_same_local_loader_as_getter(
    monkeypatch: pytest.MonkeyPatch,
):
    first_registry = _registry()
    refreshed_registry = _registry()
    mock_load = MagicMock(side_effect=[first_registry, refreshed_registry])
    monkeypatch.setattr(register_models, "_register_models", mock_load)

    with patch.object(model_library, "model_library_settings", LocalSettings()):
        assert get_model_registry() is first_registry
        assert (
            register_models.refresh_model_registry(refresh_ttl=timedelta(seconds=30))
            is None
        )
        assert get_model_registry() is refreshed_registry

    assert mock_load.call_count == 2


def test_local_refresh_uses_stale_snapshot_on_yaml_error(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry()
    mock_load = MagicMock(side_effect=[registry, yaml.YAMLError("invalid config")])
    monkeypatch.setattr(register_models, "_register_models", mock_load)

    with patch.object(model_library, "model_library_settings", LocalSettings()):
        assert _refresh_and_get(refresh_ttl=timedelta(0)) is registry
        assert (
            _refresh_and_get(refresh_ttl=timedelta(0), allow_stale_on_error=True)
            is registry
        )

    assert mock_load.call_count == 2


def test_gateway_wrapper_skips_local_provider_initialization(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry()
    initialize_providers = MagicMock(
        side_effect=AssertionError("Gateway registry must not import local providers")
    )

    monkeypatch.setattr(register_models, "get_provider_registry", initialize_providers)
    patch_gateway_fetch(monkeypatch, lambda _gateway_url: registry)

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert (
            register_models.refresh_model_registry(refresh_ttl=timedelta(seconds=30))
            is None
        )
        assert get_model_registry() is registry

    initialize_providers.assert_not_called()


def test_gateway_model_hydration_preserves_raw_provider_properties(
    monkeypatch: pytest.MonkeyPatch,
):
    with patch.object(model_library, "model_library_settings", LocalSettings()):
        payload = get_model_registry()["openai/gpt-4o"].model_dump(mode="json")
    payload["provider_properties"] = {"serverless": False}
    dynamic_properties = MagicMock(
        side_effect=AssertionError(
            "Gateway model hydration must not load provider property types"
        )
    )
    monkeypatch.setattr(
        register_models,
        "get_dynamic_provider_properties_model",
        dynamic_properties,
    )

    model = register_models.model_config_from_json(payload)

    assert model.provider_properties.model_dump() == {"serverless": False}
    dynamic_properties.assert_not_called()


def test_gateway_wrapper_replaces_legacy_initialized_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    legacy_registry = _registry()
    refreshed_registry = _registry()
    mock_fetch = MagicMock(side_effect=[legacy_registry, refreshed_registry])
    patch_gateway_fetch(monkeypatch, mock_fetch)

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert get_model_registry() is legacy_registry
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is refreshed_registry
        assert get_model_registry() is refreshed_registry

    assert mock_fetch.call_count == 2


def test_refresh_ttl_replaces_expired_gateway_registry_atomically(
    monkeypatch: pytest.MonkeyPatch,
):
    first_registry = _registry()
    second_registry = _registry()
    now = [100.0]
    mock_fetch = MagicMock(side_effect=[first_registry, second_registry])
    patch_gateway_fetch(monkeypatch, mock_fetch)
    monkeypatch.setattr(register_models, "monotonic", lambda: now[0])

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is first_registry
        now[0] = 129.999
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is first_registry
        now[0] = 130.0
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is second_registry
        assert get_model_registry() is second_registry

    assert mock_fetch.call_count == 2


def test_gateway_registry_first_fetch_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_gateway_fetch(monkeypatch, MagicMock(side_effect=OSError("gateway down")))

    with (
        patch.object(model_library, "model_library_settings", GatewaySettings()),
        pytest.raises(OSError, match="gateway down"),
    ):
        _refresh_and_get(refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True)

    assert register_models._model_registry is None


def test_gateway_registry_refresh_failure_raises_without_destroying_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry()
    now = [100.0]
    mock_fetch = MagicMock(side_effect=[registry, OSError("gateway down")])
    patch_gateway_fetch(monkeypatch, mock_fetch)
    monkeypatch.setattr(register_models, "monotonic", lambda: now[0])

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is registry
        now[0] = 130.0
        with pytest.raises(OSError, match="gateway down"):
            _refresh_and_get(refresh_ttl=timedelta(seconds=30))
        assert get_model_registry() is registry


def test_gateway_registry_stale_fallback_starts_new_refresh_cooldown(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry()
    now = [100.0]
    mock_fetch = MagicMock(side_effect=[registry, OSError("gateway down"), registry])
    patch_gateway_fetch(monkeypatch, mock_fetch)
    monkeypatch.setattr(register_models, "monotonic", lambda: now[0])

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        now[0] = 130.0
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        now[0] = 159.999
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        now[0] = 160.0
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )

    assert mock_fetch.call_count == 3


def test_stale_fallback_cooldown_starts_after_slow_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry()
    now = [100.0]
    fetch_count = 0

    def fetch(_url: str):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 2:
            now[0] = 165.0
            raise OSError("slow gateway failure")
        return registry

    patch_gateway_fetch(monkeypatch, fetch)
    monkeypatch.setattr(register_models, "monotonic", lambda: now[0])

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        now[0] = 130.0
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        now[0] = 194.999
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )
        assert fetch_count == 2
        now[0] = 195.0
        assert (
            _refresh_and_get(
                refresh_ttl=timedelta(seconds=30), allow_stale_on_error=True
            )
            is registry
        )

    assert fetch_count == 3


def test_concurrent_expired_callers_share_one_gateway_refresh(
    monkeypatch: pytest.MonkeyPatch,
):
    stale_registry = _registry()
    refreshed_registry = _registry()
    now = [100.0]
    load_started = Event()
    allow_load = Event()
    count_lock = Lock()
    load_count = 0

    def fetch(_url: str):
        nonlocal load_count
        with count_lock:
            load_count += 1
            current_count = load_count
        if current_count == 1:
            return stale_registry
        load_started.set()
        assert allow_load.wait(timeout=1)
        return refreshed_registry

    patch_gateway_fetch(monkeypatch, fetch)
    monkeypatch.setattr(register_models, "monotonic", lambda: now[0])

    with (
        patch.object(model_library, "model_library_settings", GatewaySettings()),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        assert _refresh_and_get(refresh_ttl=timedelta(seconds=30)) is stale_registry
        now[0] = 130.0
        first = executor.submit(_refresh_and_get, refresh_ttl=timedelta(seconds=30))
        assert load_started.wait(timeout=1)
        second = executor.submit(_refresh_and_get, refresh_ttl=timedelta(seconds=30))
        allow_load.set()
        assert first.result(timeout=1) is refreshed_registry
        assert second.result(timeout=1) is refreshed_registry

    assert load_count == 2


def test_zero_refresh_ttl_fetches_every_time(monkeypatch: pytest.MonkeyPatch):
    first_registry = _registry()
    second_registry = _registry()
    mock_fetch = MagicMock(side_effect=[first_registry, second_registry])
    patch_gateway_fetch(monkeypatch, mock_fetch)

    with patch.object(model_library, "model_library_settings", GatewaySettings()):
        assert _refresh_and_get(refresh_ttl=timedelta(0)) is first_registry
        assert _refresh_and_get(refresh_ttl=timedelta(0)) is second_registry
