import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.runtime_config as runtime_config_module
import model_library.base.base as model_base_module
import model_library.providers.openai.openai as openai_provider_module
from model_gateway.cache import ModelCache
from model_gateway.runtime_config import (
    AppConfigAgentClient,
    RuntimeConfigManager,
    RuntimeConfigState,
    _parse_credential_snapshot,
    _parse_model_snapshot,
)
from model_gateway.types import QueryRequest
from model_library import model_library_settings
from model_library.registry_utils import get_registry_config
from model_library.providers.openai import OpenAIModel


MODEL_NAME = "vals/dumbmar-5o-ultra-thinking"
MODEL_DOCUMENT = Path("tests/unit/model_gateway/dummy_model.yaml").read_text()
OPENAI_MODEL_NAME = "openai/runtime-config-credential-cache-test"
OPENAI_MODEL_DOCUMENT = """\
base-config:
  company: OpenAI
  country: United States
  open_source: false
  supports:
    images: false
    files: false
    tools: true
    temperature: true
    output_schema: true
  properties:
    context_window: 128000
    max_tokens: 1024
    reasoning_model: false
  costs_per_million_token:
    input: 0.0
    output: 0.0
runtime-config-models:
  openai/runtime-config-credential-cache-test:
    label: Runtime Config Credential Cache Test
    release_date: null
"""
_original_openai_get_client = OpenAIModel.get_client
_original_openai_settings = openai_provider_module.model_library_settings


def _model_payload(*, revision: int, label: str, originated_at: datetime) -> bytes:
    document = MODEL_DOCUMENT.replace("label: Dummy Model", f"label: {label}", 1)
    return json.dumps(
        {
            "schema_version": 1,
            "revision": revision,
            "originated_at": originated_at.isoformat(),
            "documents": {"dummy_model.yaml": document},
        }
    ).encode()


def _openai_model_payload(
    *, revision: int, label: str, originated_at: datetime
) -> bytes:
    payload = json.loads(
        _model_payload(revision=revision, label="Unused", originated_at=originated_at)
    )
    document = OPENAI_MODEL_DOCUMENT.replace(
        "label: Runtime Config Credential Cache Test", f"label: {label}", 1
    )
    payload["documents"] = {"runtime_config_cache_test.yaml": document}
    return json.dumps(payload).encode()


# The pools document every deployed stage publishes before its tasks start.
POOL_PAYLOAD = json.dumps(
    {"pools": {}, "model_pools": [], "weights": {}, "revision": 1}
).encode()


def _credential_payload(*, api_key: str) -> bytes:
    return json.dumps({"OPENAI_API_KEY": api_key}).encode()


class MutableAgent:
    def __init__(self, model_payload: bytes, credential_payload: bytes) -> None:
        self.model_payload = model_payload
        self.credential_payload = credential_payload
        self.model_version = "models-1"
        self.credential_version = "credentials-1"
        self.fail_profiles: set[str] = set()

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            profile = request.url.path.rsplit("/", 1)[-1]
            assert request.headers["Authorization"] == "Bearer local-token"
            if profile in self.fail_profiles:
                return httpx.Response(503, text="agent unavailable")
            if profile == "models":
                return httpx.Response(
                    200,
                    content=self.model_payload,
                    headers={"Configuration-Version": self.model_version},
                )
            if profile == "credentials":
                return httpx.Response(
                    200,
                    content=self.credential_payload,
                    headers={"Configuration-Version": self.credential_version},
                )
            if profile == "pools":
                return httpx.Response(
                    200, content=POOL_PAYLOAD, headers={"Configuration-Version": "1"}
                )
            return httpx.Response(404)

        return httpx.MockTransport(handle)


def _manager(agent: MutableAgent) -> RuntimeConfigManager:
    http_client = httpx.AsyncClient(transport=agent.transport())
    source = AppConfigAgentClient(
        base_url="http://127.0.0.1:2772",
        application="model-gateway",
        environment="test",
        access_token="local-token",
        http_client=http_client,
    )
    return RuntimeConfigManager(
        state=RuntimeConfigState.bootstrap(),
        agent=source,
        model_profile="models",
        credential_profile="credentials",
        pool_profile="pools",
    )


def _cached_observation(state: RuntimeConfigState, cache: ModelCache) -> object:
    with state.capture() as view:
        config = get_registry_config(MODEL_NAME)
        assert config is not None

        def create(_model: str, _config: dict[str, object]) -> object:
            return {
                "label": config.label,
                "api_key": model_library_settings.OPENAI_API_KEY,
            }

        return cache.get_or_create(
            MODEL_NAME,
            {},
            create,
            namespace=view.cache_namespace,
        )


async def test_two_workers_replace_continuously_used_cached_objects():
    started = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Before", originated_at=started),
        _credential_payload(api_key="cred-before"),
    )
    workers = [_manager(agent), _manager(agent)]
    caches = [ModelCache(), ModelCache()]
    try:
        for worker in workers:
            await worker.refresh_once()

        before = [
            _cached_observation(worker.state, cache)
            for worker, cache in zip(workers, caches, strict=True)
        ]
        for _ in range(20):
            assert [
                _cached_observation(worker.state, cache)
                for worker, cache in zip(workers, caches, strict=True)
            ] == before

        changed_at = datetime.now(UTC)
        agent.model_payload = _model_payload(
            revision=2,
            label="After",
            originated_at=changed_at,
        )
        agent.credential_payload = _credential_payload(api_key="cred-after")
        agent.credential_version = "credentials-2"

        for worker in workers:
            await worker.refresh_once()
        after = [
            _cached_observation(worker.state, cache)
            for worker, cache in zip(workers, caches, strict=True)
        ]

        assert after == [
            {"label": "After", "api_key": "cred-after"},
            {"label": "After", "api_key": "cred-after"},
        ]
        assert all(new is not old for new, old in zip(after, before, strict=True))
        assert before == [
            {"label": "Before", "api_key": "cred-before"},
            {"label": "Before", "api_key": "cred-before"},
        ]
    finally:
        for worker in workers:
            await worker.aclose()


async def test_run_loop_applies_published_snapshots_within_controlled_poll_window(
    monkeypatch: pytest.MonkeyPatch,
):
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Before", originated_at=now),
        _credential_payload(api_key="cred-before"),
    )
    worker = _manager(agent)
    clock = [100.0]
    publication_time = clock[0]
    wait_calls = 0
    refreshes = 0
    refreshes_before_first_wait: int | None = None
    first_wait_timeout: float | None = None
    stop = asyncio.Event()
    original_wait_for = asyncio.wait_for
    original_refresh_once = worker.refresh_once

    async def counted_refresh_once():
        nonlocal refreshes
        refreshes += 1
        return await original_refresh_once()

    monkeypatch.setattr(worker, "refresh_once", counted_refresh_once)

    async def controlled_wait_for(awaitable, timeout=None):
        nonlocal wait_calls
        frame = getattr(awaitable, "cr_frame", None)
        if frame is None or frame.f_locals.get("self") is not stop:
            return await original_wait_for(awaitable, timeout)

        nonlocal refreshes_before_first_wait, first_wait_timeout
        wait_calls += 1
        if wait_calls == 1:
            refreshes_before_first_wait = refreshes
            first_wait_timeout = timeout
            awaitable.close()
            agent.model_payload = _model_payload(
                revision=2,
                label="After",
                originated_at=datetime.now(UTC),
            )
            agent.credential_payload = _credential_payload(api_key="cred-after")
            agent.credential_version = "credentials-2"
            clock[0] += timeout or 0.0
            raise TimeoutError

        stop.set()
        return await awaitable

    monkeypatch.setattr(
        runtime_config_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(asyncio, "wait_for", controlled_wait_for)
    run = asyncio.create_task(worker.run(stop))
    try:
        await run
        assert wait_calls == 2
        assert refreshes_before_first_wait == 0
        assert first_wait_timeout == worker._poll_interval_seconds
        assert refreshes == 1
        assert clock[0] - publication_time < 60.0
        assert worker.state.current.models.revision == 2
        assert worker.state.current.credentials is not None
        assert worker.state.current.credentials.version == "credentials-2"
    finally:
        stop.set()
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        await worker.aclose()


@pytest.mark.parametrize("changed_source", ["models", "credentials", "both"])
async def test_production_cache_helper_rebuilds_model_instance_after_runtime_update(
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
):
    monkeypatch.setenv("MODEL_GATEWAY_URL", "")
    now = datetime.now(UTC)
    agent = MutableAgent(
        _openai_model_payload(revision=1, label="Before", originated_at=now),
        _credential_payload(api_key="test-runtime-credential-a"),
    )
    worker = _manager(agent)
    cache = ModelCache()
    request = QueryRequest(model=OPENAI_MODEL_NAME, inputs=[])
    sdk_constructions: list[dict[str, object]] = []

    def mock_async_openai(**kwargs: object) -> object:
        sdk_constructions.append(kwargs)
        return object()

    monkeypatch.setattr(openai, "AsyncOpenAI", mock_async_openai)
    # The unit autouse fixtures replace every provider client builder/settings
    # module. Restore only OpenAI's production path for this constructor test.
    monkeypatch.setattr(OpenAIModel, "get_client", _original_openai_get_client)
    monkeypatch.setattr(
        openai_provider_module, "model_library_settings", _original_openai_settings
    )
    monkeypatch.setattr(model_base_module, "client_registry", {})
    try:
        await worker.refresh_once()
        with worker.state.capture():
            before = model_helpers.get_cached_llm(cache, request, config={})
            assert model_helpers.get_cached_llm(cache, request, config={}) is before
        assert before.provider == "openai"
        assert before.native
        assert before.get_client() is not None
        assert before.metadata is not None
        assert before.metadata.label == "Before"
        assert len(sdk_constructions) == 1
        assert all(
            call.get("api_key") == "test-runtime-credential-a"
            for call in sdk_constructions
        )

        if changed_source in {"models", "both"}:
            agent.model_payload = _openai_model_payload(
                revision=2, label="After", originated_at=datetime.now(UTC)
            )
        if changed_source in {"credentials", "both"}:
            agent.credential_payload = _credential_payload(
                api_key="test-runtime-credential-b"
            )
            agent.credential_version = "credentials-2"
        await worker.refresh_once()

        with worker.state.capture():
            after = model_helpers.get_cached_llm(cache, request, config={})
            assert model_helpers.get_cached_llm(cache, request, config={}) is after
        assert after is not before
        assert after.provider == "openai"
        assert after.metadata is not None
        assert after.metadata.label == (
            "After" if changed_source in {"models", "both"} else "Before"
        )
        if changed_source in {"credentials", "both"}:
            assert after.get_client() is not before.get_client()
        else:
            assert after.get_client() is before.get_client()
        expected_keys = (
            ("test-runtime-credential-a", "test-runtime-credential-b")
            if changed_source in {"credentials", "both"}
            else ("test-runtime-credential-a",)
        )
        assert len(sdk_constructions) == len(expected_keys)
        assert all(
            call.get("api_key") == expected
            for call, expected in zip(sdk_constructions, expected_keys)
        )
    finally:
        for call in sdk_constructions:
            http_client = call.get("http_client")
            if isinstance(http_client, httpx.AsyncClient):
                await http_client.aclose()
        await worker.aclose()


async def test_invalid_duplicate_and_out_of_order_updates_keep_last_valid_snapshot(
    caplog: pytest.LogCaptureFixture,
):
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=3, label="Current", originated_at=now),
        _credential_payload(api_key="hidden-key"),
    )
    worker = _manager(agent)
    cache = ModelCache()
    try:
        await worker.refresh_once()
        current = _cached_observation(worker.state, cache)

        installed = worker.state.current
        await worker.refresh_once()
        assert worker.state.current is installed
        assert _cached_observation(worker.state, cache) is current

        agent.model_payload = _model_payload(
            revision=2,
            label="Older",
            originated_at=now - timedelta(seconds=10),
        )
        agent.credential_payload = _credential_payload(api_key="older-secret")
        agent.credential_version = "credentials-2"
        rejected = await worker.refresh_once()

        assert worker.state.current.models.revision == 3
        assert worker.state.current.credentials is not None
        assert worker.state.current.credentials.version == "credentials-2"
        assert rejected.rejected == frozenset({"models"})
        assert _cached_observation(worker.state, cache) is not current

        agent.model_payload = _model_payload(
            revision=4,
            label="Must Not Apply",
            originated_at=now,
        )
        agent.credential_payload = b'["OPENAI_API_KEY"]'
        agent.credential_version = "credentials-3"
        invalid = await worker.refresh_once()

        assert worker.state.current.models.revision == 4
        assert invalid.failed == frozenset({"credentials"})
        assert _cached_observation(worker.state, cache) == {
            "label": "Must Not Apply",
            "api_key": "older-secret",
        }

        agent.model_payload = json.dumps(
            {
                "schema_version": 1,
                "revision": 4,
                "originated_at": datetime.now(UTC).isoformat(),
                "documents": {"broken.yaml": "models: ["},
            }
        ).encode()
        agent.credential_payload = _credential_payload(api_key="must-not-apply")
        agent.credential_version = "credentials-4"
        invalid_model = await worker.refresh_once()

        assert invalid_model.failed == frozenset({"models"})
        assert worker.state.current.credentials is not None
        assert worker.state.current.credentials.version == "credentials-4"
        assert "hidden-key" not in caplog.text
        assert "must-not-apply" not in caplog.text
    finally:
        await worker.aclose()


async def test_removed_managed_field_does_not_fall_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Current", originated_at=now),
        _credential_payload(api_key="current"),
    )
    worker = _manager(agent)
    try:
        monkeypatch.setenv("OPENAI_API_KEY", "stale-startup-value")
        await worker.refresh_once()
        with worker.state.capture():
            assert model_library_settings.OPENAI_API_KEY == "current"

        agent.credential_payload = b"{}"
        agent.credential_version = "credentials-2"
        await worker.refresh_once()

        assert worker.state.current.credentials is not None
        assert worker.state.current.credentials.version == "credentials-2"
        with worker.state.capture():
            assert model_library_settings.get("OPENAI_API_KEY") is None
    finally:
        await worker.aclose()


async def test_flat_aggregate_secret_preserves_existing_top_level_fields():
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Current", originated_at=now),
        _credential_payload(api_key="current-key"),
    )
    worker = _manager(agent)
    try:
        await worker.refresh_once()

        with worker.state.capture():
            assert model_library_settings.OPENAI_API_KEY == "current-key"
            assert model_library_settings.get("ANTHROPIC_API_KEY") is None
    finally:
        await worker.aclose()


async def test_in_flight_capture_finishes_with_old_pair_after_refresh():
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Before", originated_at=now),
        _credential_payload(api_key="before"),
    )
    worker = _manager(agent)
    cache = ModelCache()
    captured = asyncio.Event()
    release = asyncio.Event()

    async def in_flight() -> tuple[str, str, object]:
        with worker.state.capture():
            before = _cached_observation(worker.state, cache)
            captured.set()
            await release.wait()
            config = get_registry_config(MODEL_NAME)
            assert config is not None
            return config.label, model_library_settings.OPENAI_API_KEY, before

    try:
        await worker.refresh_once()
        request = asyncio.create_task(in_flight())
        await captured.wait()

        changed_at = datetime.now(UTC)
        agent.model_payload = _model_payload(
            revision=2,
            label="After",
            originated_at=changed_at,
        )
        agent.credential_payload = _credential_payload(api_key="after")
        agent.credential_version = "credentials-2"
        await worker.refresh_once()
        after = _cached_observation(worker.state, cache)
        release.set()

        old_label, old_key, old_object = await request
        assert (old_label, old_key) == ("Before", "before")
        assert old_object == {"label": "Before", "api_key": "before"}
        assert after == {"label": "After", "api_key": "after"}
        assert after is not old_object
    finally:
        await worker.aclose()


async def test_asgi_middleware_keeps_in_flight_request_on_old_runtime_pair(
    monkeypatch: pytest.MonkeyPatch,
):
    class GatewaySettings:
        MODEL_GATEWAY_API_KEYS = '{"test":"test-api-key"}'
        MODEL_GATEWAY_HMAC_SECRET = "test-hmac-secret"

        def get(self, name: str, default=None):
            return getattr(self, name, default)

        def unset(self, _name: str) -> None:
            pass

    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "control")
    monkeypatch.delenv("GATEWAY_APPCONFIG_APPLICATION", raising=False)
    monkeypatch.setattr(gateway_app, "model_library_settings", GatewaySettings())
    monkeypatch.setattr(
        gateway_app, "runtime_config_manager_from_env", lambda _state: None
    )
    app = gateway_app.create_app()
    state = app.state.runtime_config_state
    state.install(
        models=_parse_model_snapshot(
            _model_payload(
                revision=1,
                label="Before",
                originated_at=datetime.now(UTC),
            )
        ),
        credentials=_parse_credential_snapshot(
            _credential_payload(api_key="cred-before"),
            version="credentials-1",
            current=None,
        ),
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    first_request: asyncio.Task[httpx.Response] | None = None

    async def runtime_view(hold_first: bool = False) -> dict[str, str]:
        config = get_registry_config(MODEL_NAME)
        assert config is not None
        captured = {
            "label": config.label,
            "api_key": model_library_settings.OPENAI_API_KEY,
        }
        if hold_first:
            first_entered.set()
            await release_first.wait()
            config_after_wait = get_registry_config(MODEL_NAME)
            assert config_after_wait is not None
            captured["label_after_wait"] = config_after_wait.label
            captured["api_key_after_wait"] = model_library_settings.OPENAI_API_KEY
        return captured

    async def held_runtime_view() -> dict[str, str]:
        return await runtime_view(hold_first=not first_entered.is_set())

    app.add_api_route(
        "/_test/runtime-view",
        held_runtime_view,
        methods=["GET"],
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer test-api-key"},
        ) as client:
            first_request = asyncio.create_task(client.get("/_test/runtime-view"))
            await asyncio.wait_for(first_entered.wait(), timeout=5.0)

            current_credentials = state.current.credentials
            assert current_credentials is not None
            state.install(
                models=_parse_model_snapshot(
                    _model_payload(
                        revision=2,
                        label="After",
                        originated_at=datetime.now(UTC),
                    )
                ),
                credentials=_parse_credential_snapshot(
                    _credential_payload(api_key="cred-after"),
                    version="credentials-2",
                    current=current_credentials,
                ),
            )
            second_response = await asyncio.wait_for(
                client.get("/_test/runtime-view"), timeout=5.0
            )
            release_first.set()
            first_response = await asyncio.wait_for(first_request, timeout=5.0)

        assert first_response.status_code == 200
        assert first_response.json() == {
            "label": "Before",
            "api_key": "cred-before",
            "label_after_wait": "Before",
            "api_key_after_wait": "cred-before",
        }
        assert second_response.status_code == 200
        assert second_response.json() == {
            "label": "After",
            "api_key": "cred-after",
        }
    finally:
        release_first.set()
        if first_request is not None:
            if not first_request.done():
                first_request.cancel()
            await asyncio.gather(first_request, return_exceptions=True)


async def test_source_failure_keeps_last_valid_snapshot_and_recovers():
    now = datetime.now(UTC)
    agent = MutableAgent(
        _model_payload(revision=1, label="Before", originated_at=now),
        _credential_payload(api_key="before"),
    )
    worker = _manager(agent)
    cache = ModelCache()
    try:
        await worker.refresh_once()
        current = _cached_observation(worker.state, cache)
        agent.fail_profiles.update({"models", "credentials"})

        failed = await worker.refresh_once()

        assert failed.failed == frozenset({"models", "credentials"})
        assert _cached_observation(worker.state, cache) is current

        agent.fail_profiles.clear()
        recovered_at = datetime.now(UTC)
        agent.model_payload = _model_payload(
            revision=2,
            label="Recovered",
            originated_at=recovered_at,
        )
        agent.credential_payload = _credential_payload(api_key="recovered")
        agent.credential_version = "credentials-2"

        await worker.refresh_once()

        assert _cached_observation(worker.state, cache) == {
            "label": "Recovered",
            "api_key": "recovered",
        }
    finally:
        await worker.aclose()


async def test_unchanged_model_payload_is_not_reparsed(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = MutableAgent(
        _model_payload(revision=3, label="Current", originated_at=datetime.now(UTC)),
        _credential_payload(api_key="hidden-key"),
    )
    parses: list[int] = []
    parse = runtime_config_module.parse_registry_documents

    def counting_parse(documents):
        parses.append(1)
        return parse(documents)

    monkeypatch.setattr(
        runtime_config_module, "parse_registry_documents", counting_parse
    )
    worker = _manager(agent)
    try:
        await worker.refresh_once()
        installed = worker.state.current
        await worker.refresh_once()

        assert worker.state.current is installed
        assert len(parses) == 1
    finally:
        await worker.aclose()
