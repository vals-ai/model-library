import json
from types import SimpleNamespace

import pytest

import model_library.telemetry as telemetry


class RecordingSpan:
    def __init__(self):
        self.events: list[tuple[str, dict[str, object]]] = []

    def is_recording(self) -> bool:
        return True

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, attributes))


@pytest.fixture(autouse=True)
def reset_telemetry_state(monkeypatch):
    monkeypatch.delenv("GATEWAY_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    telemetry.shutdown_telemetry()
    telemetry._seen_config_hashes.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(telemetry, "_httpx_instrumented", False)
    yield
    telemetry.shutdown_telemetry()
    telemetry._seen_config_hashes.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(telemetry, "_httpx_instrumented", False)


def test_telemetry_is_disabled_by_default():
    assert telemetry.configure_telemetry() is False
    assert telemetry.is_enabled() is False

    with telemetry.start_span("test.disabled") as span:
        telemetry.set_attributes({"run_id": "run-a"})
        telemetry.add_event("test.event", {"question_id": "q1"})
        assert span is None


def test_configure_telemetry_missing_sentry_dsn_does_not_raise(monkeypatch):
    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "true")

    assert telemetry.configure_telemetry() is False
    assert telemetry.is_enabled() is False


def test_start_span_sets_sentry_operation_attributes(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSpanContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            pass

    class FakeTracer:
        def start_as_current_span(
            self, *args: object, **kwargs: object
        ) -> FakeSpanContext:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeSpanContext()

    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        SimpleNamespace(get_tracer=lambda _name: FakeTracer()),
    )
    monkeypatch.setattr(telemetry, "_span_kind", None)
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar("test_sentry_op_context", default={}),
    )

    with telemetry.start_span("POST /query", {"run_id": "run-a"}, kind="server"):
        pass

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    attributes = kwargs["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["sentry.op"] == "http.server"
    assert attributes["sentry.origin"] == "manual.gateway"
    assert attributes["run_id"] == "run-a"

    with telemetry.start_span("model_library.provider_query", kind="client"):
        pass

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    attributes = kwargs["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["sentry.op"] == "gen_ai.request"
    assert attributes["sentry.origin"] == "manual.model_library"


def test_configure_telemetry_initializes_sentry_otlp_exporter(monkeypatch):
    sentry_init_kwargs: dict[str, object] = {}
    tracer_provider: object | None = None
    instrumented_kwargs: dict[str, object] = {}

    class FakeTracerProvider:
        def __init__(self, *, resource: object, sampler: object) -> None:
            self.resource = resource
            self.sampler = sampler
            self.processors: list[object] = []
            self.shutdown_called = False

        def add_span_processor(self, processor: object) -> None:
            self.processors.append(processor)

        def shutdown(self) -> None:
            self.shutdown_called = True

    class FakeHTTPXClientInstrumentor:
        def instrument(self, **kwargs: object) -> None:
            instrumented_kwargs.update(kwargs)

    def fake_sentry_init(**kwargs: object) -> None:
        sentry_init_kwargs.update(kwargs)

    def fake_import_module(name: str) -> object:
        modules = {
            "opentelemetry.instrumentation.httpx": SimpleNamespace(
                HTTPXClientInstrumentor=FakeHTTPXClientInstrumentor,
            ),
            "opentelemetry.sdk.resources": SimpleNamespace(
                Resource=SimpleNamespace(create=lambda attrs: attrs),
            ),
            "opentelemetry.sdk.trace": SimpleNamespace(
                TracerProvider=FakeTracerProvider,
                sampling=SimpleNamespace(
                    _KNOWN_SAMPLERS={"parentbased_always_on": "always-on-sampler"}
                ),
            ),
            "sentry_sdk": SimpleNamespace(init=fake_sentry_init),
            "sentry_sdk.consts": SimpleNamespace(
                INSTRUMENTER=SimpleNamespace(OTEL="otel"),
            ),
            "sentry_sdk.integrations.logging": SimpleNamespace(
                LoggingIntegration=lambda **kwargs: ("logging", kwargs),
            ),
            "sentry_sdk.integrations.otlp": SimpleNamespace(
                OTLPIntegration=lambda **kwargs: ("otlp", kwargs),
            ),
        }
        return modules[name]

    def fake_set_tracer_provider(provider: object) -> None:
        nonlocal tracer_provider
        tracer_provider = provider

    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "true")
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "gateway-test")
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.namespace=vals,deployment.environment=dev,service.name=ignored",
    )
    monkeypatch.setattr(telemetry, "import_module", fake_import_module)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        SimpleNamespace(set_tracer_provider=fake_set_tracer_provider),
    )

    assert telemetry.configure_telemetry() is True

    assert sentry_init_kwargs["dsn"] == "https://public@example.com/1"
    assert sentry_init_kwargs["server_name"] == "gateway-test"
    assert sentry_init_kwargs["environment"] == "dev"
    assert sentry_init_kwargs["instrumenter"] == "otel"
    assert sentry_init_kwargs["enable_logs"] is True
    assert sentry_init_kwargs["send_default_pii"] is True
    assert sentry_init_kwargs["before_send"] is telemetry._before_send  # pyright: ignore[reportPrivateUsage]
    assert (
        sentry_init_kwargs["before_send_transaction"]
        is telemetry._before_send_transaction
    )  # pyright: ignore[reportPrivateUsage]
    assert sentry_init_kwargs["before_send_log"] is telemetry._before_send_log  # pyright: ignore[reportPrivateUsage]
    assert isinstance(sentry_init_kwargs["integrations"], list)
    assert (
        "logging",
        {"level": None, "event_level": None, "sentry_logs_level": 30},
    ) in sentry_init_kwargs["integrations"]
    assert (
        "otlp",
        {
            "setup_otlp_traces_exporter": True,
            "collector_url": None,
            "setup_propagator": True,
            "capture_exceptions": False,
        },
    ) in sentry_init_kwargs["integrations"]
    assert isinstance(tracer_provider, FakeTracerProvider)
    assert tracer_provider.resource == {
        "service.name": "gateway-test",
        "service.namespace": "vals",
        "deployment.environment": "dev",
    }
    assert tracer_provider.sampler == "always-on-sampler"
    assert tracer_provider.processors == []
    assert instrumented_kwargs == {
        "request_hook": telemetry._httpx_request_hook,  # pyright: ignore[reportPrivateUsage]
        "async_request_hook": telemetry._httpx_async_request_hook,  # pyright: ignore[reportPrivateUsage]
    }
    assert telemetry.is_enabled() is True


def test_should_trace_http_route_only_allows_gateway_routes():
    assert telemetry.should_trace_http_route("/query") is True
    assert telemetry.should_trace_http_route("/models/resolve") is True
    assert telemetry.should_trace_http_route("/health/live") is False
    assert telemetry.should_trace_http_route("/.env") is False


def test_httpx_request_hook_renames_generic_method_span():
    class FakeSpan:
        def __init__(self) -> None:
            self.name: str | None = None
            self.attrs: dict[str, object] = {}

        def update_name(self, name: str) -> None:
            self.name = name

        def set_attributes(self, attrs: dict[str, object]) -> None:
            self.attrs.update(attrs)

    request_info = SimpleNamespace(method=b"POST", url=SimpleNamespace(path="/query"))
    span = FakeSpan()

    telemetry._httpx_request_hook(span, request_info)  # pyright: ignore[reportPrivateUsage]

    assert span.name == "POST /query"
    assert span.attrs == {
        "sentry.op": "http.client",
        "sentry.origin": "auto.httpx.gateway",
        "http.method": "POST",
        "http.request.method": "POST",
        "http.route": "/query",
        "url.path": "/query",
    }


def test_sanitize_attributes_keeps_ids_and_content_but_drops_credentials():
    attrs = telemetry.sanitize_attributes(
        {
            "run_id": "run-a",
            "question_id": "q1",
            "api_key": "secret",
            "authorization": "Bearer secret",
            "prompt": "do not record",
            "raw_input": "do not record",
            "output_text": "do not record",
            "gen_ai.usage.input_tokens": 12,
            "gateway.enabled": True,
            "gateway.error.code": "access_denied",
            "gateway.error.phase": "access_control",
            "gateway.error.provider": "openai",
            "long": "x" * (telemetry.MAX_ATTRIBUTE_LENGTH + 5),
            "none": None,
        }
    )

    assert attrs["run_id"] == "run-a"
    assert attrs["question_id"] == "q1"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gateway.enabled"] is True
    assert attrs["gateway.error_code"] == "access_denied"
    assert attrs["gateway.error_phase"] == "access_control"
    assert attrs["gateway.error_provider"] == "openai"
    assert attrs["long"] == "x" * telemetry.MAX_ATTRIBUTE_LENGTH
    assert attrs["prompt"] == "do not record"
    assert attrs["raw_input"] == "do not record"
    assert attrs["output_text"] == "do not record"
    assert "api_key" not in attrs
    assert "authorization" not in attrs
    assert "none" not in attrs


def test_model_attributes_use_genai_conventions():
    attrs = telemetry.model_attributes(operation="query", model="anthropic/claude")

    assert attrs == {
        "gateway.operation": "query",
        "gen_ai.operation.name": "query",
        "gen_ai.request.model": "anthropic/claude",
        "gen_ai.system": "anthropic",
        "model.provider": "anthropic",
        "model.name": "claude",
        "model.registry_key": "anthropic/claude",
    }


def test_run_attributes_reads_request_context_only():
    attrs = telemetry.run_attributes(
        {"run_id": "run-a", "question_id": "q1", "temperature": 0.2}
    )

    assert attrs == {
        "run_id": "run-a",
        "question_id": "q1",
        "query_id": None,
        "identity": None,
        "in_agent": None,
        "llm.in_agent.mode": "disabled",
    }


def test_run_attributes_serializes_identity_object_as_json_attribute():
    attrs = telemetry.run_attributes(
        {
            "identity": {
                "email": "user@example.com",
                "benchmark_name": "swebench",
                "agent_name": "swe-agent",
            }
        }
    )

    assert (
        attrs["identity"]
        == '{"agent_name":"swe-agent","benchmark_name":"swebench","email":"user@example.com"}'
    )


def test_sentry_search_attributes_keeps_any_provider_config_scalar_label():
    sanitized = telemetry.sanitize_attributes(
        {
            "llm.config.provider_config.new_flag.mode": "enabled",
            "llm.config.provider_config.custom_limit": 3,
            "llm.config.provider_config.prompt": "raw prompt",
            "llm.config.provider_config.secret_key": "secret",
            "llm.config.unrelated": "dropped",
        }
    )
    attrs = telemetry._sentry_search_attributes(  # pyright: ignore[reportPrivateUsage]
        sanitized
    )

    assert attrs == {
        "llm.config.provider_config.new_flag.mode": "enabled",
        "llm.config.provider_config.custom_limit": 3,
    }


def test_config_fingerprint_redacts_secret_fields_keeps_context_and_is_stable():
    hash_a, redacted_a = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {
            "temperature": 0,
            "context_window": 1000,
            "custom_api_key": "sk-secret",
            "provider_config": {"safe": "yes", "api_key": "nested-secret"},
        },
        {"max_tokens": 7, "run_id": "run-a"},
    )
    hash_b, redacted_b = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {
            "provider_config": {"api_key": "nested-secret", "safe": "yes"},
            "custom_api_key": "sk-secret",
            "temperature": 0,
            "context_window": 1000,
        },
        {"run_id": "run-a", "max_tokens": 7},
    )

    assert hash_a == hash_b
    assert redacted_a == redacted_b
    config = redacted_a["config"]
    params = redacted_a["params"]
    assert isinstance(config, dict)
    assert isinstance(params, dict)
    provider_config = config["provider_config"]
    assert isinstance(provider_config, dict)
    assert config["custom_api_key"] == "<redacted>"
    assert config["context_window"] == 1000
    assert provider_config["api_key"] == "<redacted>"
    assert params["max_tokens"] == 7


def test_config_fingerprint_buckets_custom_endpoint_without_raw_url():
    raw_a = "https://private-a.example.internal/v1"
    raw_b = "https://private-b.example.internal/v1"
    hash_a, redacted_a = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {"custom_endpoint": raw_a, "custom_api_key": "sk-secret"},
    )
    hash_b, redacted_b = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {"custom_endpoint": raw_b, "custom_api_key": "sk-secret"},
    )

    assert hash_a == hash_b
    assert redacted_a == redacted_b
    config = redacted_a["config"]
    assert isinstance(config, dict)
    assert config["custom_endpoint"] == "custom"
    assert raw_a not in json.dumps(redacted_a)
    assert raw_b not in json.dumps(redacted_b)


def test_config_fingerprint_redacts_content_variants_but_keeps_safe_config_shape():
    _config_hash, redacted = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {
            "system_prompt": "SECRET PROMPT",
            "request_json": {"messages": ["top secret"]},
            "response_text": "provider output",
            "max_context_window": 1000,
            "prompt_cache_retention": "24h",
            "request_timeout_seconds": 30,
            "response_json_schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            },
            "response_mime_type": "application/json",
        },
    )

    config = redacted["config"]
    assert isinstance(config, dict)
    assert config["system_prompt"] == "<redacted>"
    assert config["request_json"] == "<redacted>"
    assert config["response_text"] == "<redacted>"
    assert config["max_context_window"] == 1000
    assert config["prompt_cache_retention"] == "24h"
    assert config["request_timeout_seconds"] == 30
    assert config["response_json_schema"] == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    assert config["response_mime_type"] == "application/json"


def test_config_fingerprint_excludes_request_ids_from_params():
    hash_a, redacted_a = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {"max_tokens": 10},
        {"run_id": "run-a", "question_id": "q-a", "temperature": 0},
    )
    hash_b, redacted_b = telemetry.config_fingerprint(
        "openai/gpt-4o",
        {"max_tokens": 10},
        {"run_id": "run-b", "question_id": "q-b", "temperature": 0},
    )

    assert hash_a == hash_b
    assert redacted_a == redacted_b
    assert redacted_a["params"] == {"temperature": 0}


def test_config_fingerprint_distinguishes_context_window():
    hash_a, _ = telemetry.config_fingerprint("openai/gpt-4o", {"context_window": 1000})
    hash_b, _ = telemetry.config_fingerprint("openai/gpt-4o", {"context_window": 2000})

    assert hash_a != hash_b


def test_config_fingerprint_distinguishes_response_json_schema():
    hash_a, redacted_a = telemetry.config_fingerprint(
        "google/gemini", {"response_json_schema": {"type": "object"}}
    )
    hash_b, redacted_b = telemetry.config_fingerprint(
        "google/gemini", {"response_json_schema": {"type": "array"}}
    )

    assert hash_a != hash_b
    assert redacted_a["config"] != redacted_b["config"]


def test_record_config_seen_emits_deduped_lookup_event(monkeypatch):
    span = RecordingSpan()
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        type("TraceApi", (), {"get_current_span": staticmethod(lambda: span)}),
    )

    config_hash, redacted = telemetry.config_fingerprint(
        "anthropic/claude", {"max_tokens": 10}, {"temperature": 0}
    )
    telemetry.record_config_seen(config_hash, redacted)
    telemetry.record_config_seen(config_hash, redacted)

    assert len(span.events) == 1
    name, attrs = span.events[0]
    assert name == "model.config_seen"
    assert attrs["model.config_hash"] == config_hash
    assert attrs["model.config_redacted_json_truncated"] is False
    assert json.loads(str(attrs["model.config_redacted_json"])) == redacted


def test_record_config_seen_does_not_dedupe_when_no_span_records(monkeypatch):
    class NonRecordingSpan:
        def is_recording(self) -> bool:
            return False

    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        type(
            "TraceApi",
            (),
            {"get_current_span": staticmethod(lambda: NonRecordingSpan())},
        ),
    )

    config_hash, redacted = telemetry.config_fingerprint(
        "anthropic/claude", {"max_tokens": 10}, {"temperature": 0}
    )
    telemetry.record_config_seen(config_hash, redacted)

    assert config_hash not in telemetry._seen_config_hashes  # pyright: ignore[reportPrivateUsage]


def test_record_config_seen_reemits_after_ttl(monkeypatch):
    span = RecordingSpan()
    now = 1000.0
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        type("TraceApi", (), {"get_current_span": staticmethod(lambda: span)}),
    )
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: now)

    config_hash, redacted = telemetry.config_fingerprint(
        "anthropic/claude", {"max_tokens": 10}, {"temperature": 0}
    )
    telemetry.record_config_seen(config_hash, redacted)
    telemetry.record_config_seen(config_hash, redacted)
    now += telemetry.CONFIG_SEEN_CACHE_TTL_SECONDS + 1
    telemetry.record_config_seen(config_hash, redacted)

    assert [name for name, _attrs in span.events] == [
        "model.config_seen",
        "model.config_seen",
    ]


def test_shutdown_telemetry_clears_config_seen_cache():
    telemetry._seen_config_hashes["abc"] = 1.0  # pyright: ignore[reportPrivateUsage]

    telemetry.shutdown_telemetry()

    assert telemetry._seen_config_hashes == {}  # pyright: ignore[reportPrivateUsage]


def test_record_config_seen_large_payload_remains_valid_json(monkeypatch):
    span = RecordingSpan()
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        type("TraceApi", (), {"get_current_span": staticmethod(lambda: span)}),
    )

    config_hash, redacted = telemetry.config_fingerprint(
        "anthropic/claude", {f"param_{i}": "x" * 100 for i in range(200)}
    )
    telemetry.record_config_seen(config_hash, redacted)

    _name, attrs = span.events[0]
    payload = json.loads(str(attrs["model.config_redacted_json"]))
    assert attrs["model.config_redacted_json_truncated"] is True
    assert payload["truncated"] is True
    assert "param_0" in payload["config_keys"]


def test_child_span_inherits_search_context_attributes(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeSpan:
        def is_recording(self) -> bool:
            return True

        def set_attributes(self, attrs: dict[str, object]) -> None:
            pass

        def add_event(self, name: str, attrs: dict[str, object]) -> None:
            pass

    class FakeSpanContext:
        def __init__(self, attrs: dict[str, object]) -> None:
            self.attrs = attrs

        def __enter__(self) -> FakeSpan:
            captured.append(self.attrs)
            return FakeSpan()

        def __exit__(self, *args: object) -> None:
            pass

    class FakeTracer:
        def start_as_current_span(
            self, *args: object, **kwargs: object
        ) -> FakeSpanContext:
            attrs = kwargs["attributes"]
            assert isinstance(attrs, dict)
            return FakeSpanContext(attrs)

    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        SimpleNamespace(
            get_tracer=lambda _name: FakeTracer(),
            get_current_span=lambda: FakeSpan(),
        ),
    )
    monkeypatch.setattr(telemetry, "_span_kind", None)
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar("test_child_context_attrs", default={}),
    )

    with telemetry.start_span(
        "POST /query",
        {"model.provider": "openai", "model.name": "gpt-4o"},
        kind="server",
    ):
        with telemetry.start_span(
            "model_library.retry_attempt", {"retry.strategy": "token"}
        ):
            pass

    retry_attrs = captured[1]
    assert retry_attrs["model.provider"] == "openai"
    assert retry_attrs["model.name"] == "gpt-4o"
    assert retry_attrs["retry.strategy"] == "token"


def test_nested_span_search_context_survives_for_parent_exception(monkeypatch):
    class FakeSpan:
        def is_recording(self) -> bool:
            return True

        def set_attributes(self, attrs: dict[str, object]) -> None:
            pass

        def add_event(self, name: str, attrs: dict[str, object]) -> None:
            pass

    class FakeSpanContext:
        def __enter__(self) -> FakeSpan:
            return FakeSpan()

        def __exit__(self, *args: object) -> None:
            pass

    class FakeTracer:
        def start_as_current_span(
            self, *args: object, **kwargs: object
        ) -> FakeSpanContext:
            return FakeSpanContext()

    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        SimpleNamespace(
            get_tracer=lambda _name: FakeTracer(),
            get_current_span=lambda: FakeSpan(),
        ),
    )
    monkeypatch.setattr(telemetry, "_span_kind", None)
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar("test_nested_context", default={}),
    )

    with telemetry.start_span("POST /query", {"run_id": "run-a"}, kind="server"):
        with telemetry.start_span(
            "model_library.retry_attempt",
            {"retry.strategy": "token", "retry.attempt": 2},
        ):
            telemetry.set_attributes({"retry.attempts": 3})
        assert telemetry._sentry_search_context() == {  # pyright: ignore[reportPrivateUsage]
            "run_id": "run-a",
            "retry.strategy": "token",
            "retry.attempt": 2,
            "retry.attempts": 3,
        }

    assert telemetry._sentry_search_context() == {}  # pyright: ignore[reportPrivateUsage]


def test_before_send_adds_search_context_as_event_tags(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar(
            "test_sentry_context",
            default={
                "run_id": "run-a",
                "question_id": "q1",
                "api_key": "not included",
            },
        ),
    )

    event = telemetry._before_send({"tags": {"existing": "tag"}}, {})  # pyright: ignore[reportPrivateUsage]

    assert event == {
        "tags": {"existing": "tag", "run_id": "run-a", "question_id": "q1"}
    }


def test_before_send_fingerprints_gateway_errors_by_stable_fields(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar(
            "test_sentry_fingerprint_context",
            default={
                "gateway.error.code": "provider_error",
                "gateway.error.provider": "openai",
                "gateway.error.phase": "provider_call",
            },
        ),
    )

    event = telemetry._before_send(  # pyright: ignore[reportPrivateUsage]
        {"exception": {"values": [{"type": "RateLimitError"}]}},
        {},
    )

    assert event == {
        "exception": {"values": [{"type": "RateLimitError"}]},
        "tags": {
            "gateway.error.code": "provider_error",
            "gateway.error.provider": "openai",
            "gateway.error.phase": "provider_call",
        },
        "fingerprint": [
            "gateway",
            "provider_error",
            "openai",
            "provider_call",
            "RateLimitError",
        ],
    }


def test_before_send_transaction_only_attaches_search_tags(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar(
            "test_transaction_context",
            default={"run_id": "run-a", "model.provider": "openai"},
        ),
    )
    event: dict[str, object] = {
        "transaction": "POST /query",
        "tags": {
            "gateway.error.code": "invalid_request",
            "gateway.error.provider": "unknown_provider",
            "gateway.error.phase": "request_validation",
        },
        "exception": {"values": [{"type": "RequestValidationError"}]},
    }

    assert telemetry._before_send_transaction(event, {}) == {  # pyright: ignore[reportPrivateUsage]
        "transaction": "POST /query",
        "tags": {
            "gateway.error.code": "invalid_request",
            "gateway.error.provider": "unknown_provider",
            "gateway.error.phase": "request_validation",
            "run_id": "run-a",
            "model.provider": "openai",
        },
        "exception": {"values": [{"type": "RequestValidationError"}]},
    }
    assert "fingerprint" not in event


def test_before_send_log_adds_trace_and_search_context(monkeypatch):
    trace_id = "019e04c6fbf0397e32a8d9601f98e45c"
    span_id = "a1f0f4fc15b83e82"
    span_context = SimpleNamespace(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        is_valid=True,
    )
    span = SimpleNamespace(get_span_context=lambda: span_context)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        SimpleNamespace(get_current_span=lambda: span),
    )
    monkeypatch.setattr(
        telemetry,
        "_sentry_context",
        telemetry.ContextVar(
            "test_sentry_log_context",
            default={"run_id": "run-a", "model.provider": "openai"},
        ),
    )

    log = telemetry._before_send_log({"body": "failed"}, {})  # pyright: ignore[reportPrivateUsage]

    assert log == {
        "body": "failed",
        "run_id": "run-a",
        "model.provider": "openai",
        "trace_id": trace_id,
        "span_id": span_id,
    }


def test_record_exception_records_sentry_issue_without_exception_message_on_span(
    monkeypatch,
):
    span = RecordingSpan()
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_load_trace_api", lambda: True)
    monkeypatch.setattr(
        telemetry,
        "_trace_api",
        type("TraceApi", (), {"get_current_span": staticmethod(lambda: span)}),
    )
    captured: list[BaseException] = []

    def fake_import_module(name: str) -> object:
        if name == "sentry_sdk":
            return SimpleNamespace(capture_exception=captured.append)
        return telemetry.import_module(name)

    monkeypatch.setattr(telemetry, "import_module", fake_import_module)

    telemetry.record_exception(
        RuntimeError("provider response body should not be recorded"),
        {"gateway.error.code": "provider_error"},
    )

    assert [str(exc) for exc in captured] == [
        "provider response body should not be recorded"
    ]
    assert span.events == [
        (
            "exception",
            {
                "exception.type": "RuntimeError",
                "gateway.error.code": "provider_error",
                "gateway.error_code": "provider_error",
            },
        )
    ]
