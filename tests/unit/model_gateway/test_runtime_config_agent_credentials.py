import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from model_gateway import runtime_config as runtime_config_module
from model_gateway.observability import worker_id
from model_gateway.runtime_config import (
    AgentConfiguration,
    AppConfigAgentClient,
    RuntimeConfigManager,
    RuntimeConfigState,
    SnapshotRejectedError,
    _parse_credential_snapshot,
    _parse_model_snapshot,
    runtime_config_manager_from_env,
)
from model_library import model_library_settings


MODEL_DOCUMENT = Path("model_library/config/dummy_model.yaml").read_text()


def _models(revision: int) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "revision": revision,
            "originated_at": datetime.now(UTC).isoformat(),
            "documents": {"dummy_model.yaml": MODEL_DOCUMENT},
        }
    ).encode()


def _credentials(value: str | None) -> bytes:
    return json.dumps({} if value is None else {"OPENAI_API_KEY": value}).encode()


def _credential_snapshot(payload: bytes, version: str, current=None):
    return _parse_credential_snapshot(
        payload,
        version=version,
        current=current,
    )


async def test_agent_requires_verified_configuration_version_header():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b'{"OPENAI_API_KEY":"x"}')
        )
    )
    agent = AppConfigAgentClient(
        base_url="http://127.0.0.1:2772",
        application="gateway",
        environment="test",
        access_token=None,
        http_client=client,
    )
    try:
        with pytest.raises(ValueError, match="Configuration-Version"):
            await agent.fetch("credentials")
    finally:
        await agent.aclose()


def test_every_snapshot_field_is_managed_and_stays_managed_after_removal():
    first = _credential_snapshot(
        b'{"OPENAI_API_KEY":"first","SAKANA_API_KEY":"sakana"}', "version-a"
    )
    removed = _credential_snapshot(b'{"SAKANA_API_KEY":"sakana"}', "version-b", first)

    assert first.managed_fields == frozenset({"OPENAI_API_KEY", "SAKANA_API_KEY"})
    assert first.values["SAKANA_API_KEY"].get_secret_value() == "sakana"
    assert removed.managed_fields == first.managed_fields
    assert "OPENAI_API_KEY" not in removed.values


def test_blank_snapshot_field_is_managed_without_a_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    snapshot = _credential_snapshot(b'{"OPENAI_API_KEY":""}', "version-a")
    state = RuntimeConfigState.bootstrap()
    state.install(credentials=snapshot)

    with state.capture():
        assert model_library_settings.get("OPENAI_API_KEY") is None


def test_gateway_owned_snapshot_fields_are_ignored_and_logged_by_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("MODEL_GATEWAY_URL", "http://env-gateway")
    payload = json.dumps(
        {
            "MODEL_GATEWAY_URL": "http://snapshot-gateway",
            "OPENAI_API_KEY": "openai",
            "BASETEN_API_BASE_URL": "https://baseten",
        }
    ).encode()

    with caplog.at_level("WARNING", logger="model_proxy_server.runtime_config"):
        snapshot = _credential_snapshot(payload, "version-a")

    assert snapshot.managed_fields == frozenset(
        {"OPENAI_API_KEY", "BASETEN_API_BASE_URL"}
    )
    assert "MODEL_GATEWAY_URL" not in snapshot.values
    assert caplog.text.count("MODEL_GATEWAY_URL") == 1
    assert "snapshot-gateway" not in caplog.text
    state = RuntimeConfigState.bootstrap()
    state.install(credentials=snapshot)
    with state.capture():
        assert model_library_settings.get("MODEL_GATEWAY_URL") == "http://env-gateway"
        assert model_library_settings.OPENAI_API_KEY == "openai"


def test_non_string_snapshot_field_is_skipped_and_logged_by_name(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level("WARNING", logger="model_proxy_server.runtime_config"):
        snapshot = _credential_snapshot(
            b'{"OPENAI_API_KEY":12345,"SAKANA_API_KEY":"sakana"}', "version-a"
        )

    assert snapshot.managed_fields == frozenset({"SAKANA_API_KEY"})
    assert "OPENAI_API_KEY" not in snapshot.values
    assert caplog.text.count("OPENAI_API_KEY") == 1
    assert "12345" not in caplog.text


def test_non_object_credential_snapshot_is_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        _credential_snapshot(b'["OPENAI_API_KEY"]', "version-a")


def test_same_version_changed_bytes_are_rejected_without_secret_in_error():
    first = _credential_snapshot(_credentials("first"), "version-a")

    with pytest.raises(SnapshotRejectedError, match="changed contents") as error:
        _credential_snapshot(_credentials("second"), "version-a", first)

    assert "first" not in str(error.value)
    assert "second" not in str(error.value)


def test_unchanged_credential_payload_reuses_current_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _credentials("first")
    first = _credential_snapshot(payload, "version-a")

    def fail_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unchanged payload was decoded")

    monkeypatch.setattr(runtime_config_module.json, "loads", fail_decode)
    monkeypatch.setattr(runtime_config_module, "SecretStr", fail_decode)

    assert _credential_snapshot(payload, "version-a", first) is first


def test_explicit_a_b_a_rollback_gets_new_generation_and_cache_namespace():
    state = RuntimeConfigState.bootstrap()
    version_a = _credential_snapshot(_credentials("a"), "opaque-a")
    state.install(credentials=version_a)
    namespace_a1 = state.current.cache_namespace
    version_b = _credential_snapshot(_credentials("b"), "opaque-b", version_a)
    state.install(credentials=version_b)
    rollback_a = _credential_snapshot(_credentials("a"), "opaque-a", version_b)
    state.install(credentials=rollback_a)

    assert (version_a.generation, version_b.generation, rollback_a.generation) == (
        1,
        2,
        3,
    )
    assert namespace_a1 == "models=0:credentials=opaque-a:1"
    assert state.current.cache_namespace == "models=0:credentials=opaque-a:3"


class SelectableAgent:
    def __init__(self, slow_profile: str) -> None:
        self.slow_profile = slow_profile

    async def fetch(self, profile: str) -> AgentConfiguration:
        if profile == self.slow_profile:
            await asyncio.sleep(1)
        content = _models(2) if profile == "models" else _credentials("after")
        return AgentConfiguration(content, f"{profile}-2")

    async def aclose(self) -> None:
        pass


async def test_applied_logs_name_worker_and_versions_without_secret(
    caplog: pytest.LogCaptureFixture,
):
    manager = RuntimeConfigManager(
        state=RuntimeConfigState.bootstrap(),
        agent=SelectableAgent("none"),  # type: ignore[arg-type]
        model_profile="models",
        credential_profile="credentials",
    )

    with caplog.at_level("INFO", logger="model_proxy_server.runtime_config"):
        await manager.refresh_once()

    assert manager.state.current.models.revision == 2
    assert caplog.text.count(f"worker_id={worker_id()}") == 2
    assert "source=models revision=2" in caplog.text
    assert "source=credentials version=credentials-2 generation=1" in caplog.text
    assert "after" not in caplog.text


@pytest.mark.parametrize("timed_out", ["models", "credentials"])
async def test_timeout_installs_healthy_source_and_preserves_captured_view(
    timed_out: str,
):
    old_models = _parse_model_snapshot(_models(1))
    old_credentials = _credential_snapshot(_credentials("before"), "credentials-1")
    state = RuntimeConfigState.bootstrap()
    state.install(models=old_models, credentials=old_credentials)
    manager = RuntimeConfigManager(
        state=state,
        agent=SelectableAgent(timed_out),  # type: ignore[arg-type]
        model_profile="models",
        credential_profile="credentials",
        source_timeout_seconds=0.01,
    )

    with state.capture() as captured:
        result = await manager.refresh_once()
        assert captured.models is old_models
        assert captured.credentials is old_credentials

    healthy = "credentials" if timed_out == "models" else "models"
    assert result.failed == frozenset({timed_out})
    if healthy == "models":
        assert state.current.models is not old_models
        assert state.current.credentials is old_credentials
    else:
        assert state.current.models is old_models
        assert state.current.credentials is not old_credentials


def test_capture_reveals_only_the_credential_a_request_reads():
    revealed: list[str] = []

    class CountingSecret(SecretStr):
        def get_secret_value(self) -> str:
            value = super().get_secret_value()
            revealed.append(value)
            return value

    snapshot = replace(
        _credential_snapshot(_credentials("openai"), "version-a"),
        values={
            "OPENAI_API_KEY": CountingSecret("openai"),
            "ANTHROPIC_API_KEY": CountingSecret("anthropic"),
        },
    )
    state = RuntimeConfigState.bootstrap()
    state.install(credentials=snapshot)

    with state.capture():
        assert revealed == []
        assert model_library_settings.OPENAI_API_KEY == "openai"

    assert revealed == ["openai"]


def _runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "GATEWAY_APPCONFIG_APPLICATION": "gateway",
        "GATEWAY_APPCONFIG_ENVIRONMENT": "test",
        "GATEWAY_APPCONFIG_MODEL_PROFILE": "models",
        "GATEWAY_APPCONFIG_CREDENTIAL_PROFILE": "credentials",
        "GATEWAY_APPCONFIG_ACCESS_TOKEN": "token",
    }.items():
        monkeypatch.setenv(name, value)


async def test_factory_snapshot_fields_decide_which_credentials_requests_use(
    monkeypatch: pytest.MonkeyPatch,
):
    _runtime_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic")
    state = RuntimeConfigState.bootstrap()
    manager = runtime_config_manager_from_env(state)
    assert manager is not None
    await manager.aclose()
    manager._agent = SelectableAgent("none")  # type: ignore[assignment]

    await manager.refresh_once()

    with state.capture():
        assert model_library_settings.OPENAI_API_KEY == "after"
        assert model_library_settings.ANTHROPIC_API_KEY == "env-anthropic"


def test_factory_ignores_legacy_managed_fields_setting(
    monkeypatch: pytest.MonkeyPatch,
):
    _runtime_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_PROVIDER_MANAGED_FIELDS", "not-json")

    assert runtime_config_manager_from_env(RuntimeConfigState.bootstrap()) is not None
