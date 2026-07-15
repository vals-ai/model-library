"""Tests for proxy mode — GatewayLLM and serialization."""

import io
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from model_library.base.base import (
    LLM,
    LLMConfig,
    ProviderConfig,
    TokenRetryParams,
    dump_gateway_config,
)
from model_library.base.gateway import GatewayLLM
from model_library.registry_utils import (
    get_model_cost,
    get_model_input_context_window,
    get_model_names,
    get_registry_config,
    get_registry_model,
)
from model_library.base.input import (
    FileWithId,
    InputItem,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
)
from model_library.base.output import (
    FinishReason,
    FinishReasonInfo,
    ProviderToolEvent,
    QueryResult,
)
from model_library.exceptions import (
    GatewayMethodNotSupported,
    GatewayProviderError,
    is_retriable_error,
)
from model_library.providers.openai import OpenAIConfig
from model_gateway.history import sign_history
import model_gateway.app as gateway_app
from model_gateway import model_helpers

PROXY_ENV = {
    "MODEL_GATEWAY_URL": "http://localhost:8000",
    "MODEL_GATEWAY_API_KEY": "test-key",
}


class _GatewaySettings:
    def __init__(self, **values: str):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._values.get(name, default)


def _identity_with_compact_json_size(size: int) -> dict[str, str]:
    value_length = size - len('{"large":""}')
    assert value_length >= 0
    identity = {"large": "x" * value_length}
    assert len(json.dumps(identity, sort_keys=True, separators=(",", ":"))) == size
    return identity


def _make_signed_blob(items, secret=b"test-secret"):
    """Build a signed history blob using the new JSON format."""
    return sign_history(items, secret=secret)


def _make_gateway():
    with patch.dict("os.environ", PROXY_ENV):
        return GatewayLLM("gpt-4o", "openai")


def _registry_config_dict(model: str = "openai/gpt-4o") -> dict[str, Any]:
    from model_library.register_models import get_model_registry

    return get_model_registry()[model].model_dump(mode="json")


def test_dump_gateway_config_keeps_explicit_overrides_and_masks_secret():
    data = dump_gateway_config(
        LLMConfig(
            max_tokens=16,
            custom_api_key=SecretStr("provider-key"),
            custom_endpoint="https://provider.example/v1",
            provider_config={"example": "provider-specific"},
            registry_key="openai/gpt-4o",
            native=False,
        )
    )

    assert data == {
        "max_tokens": 16,
        "native": False,
        "provider_config": {"example": "provider-specific"},
        "registry_key": "openai/gpt-4o",
        "custom_api_key": "**********",
        "custom_endpoint": "https://provider.example/v1",
    }


# --- GatewayLLM ---


def test_gateway_is_llm_instance():
    llm = _make_gateway()
    assert isinstance(llm, LLM)


async def test_gateway_ensure_metadata_loaded_syncs_once():
    llm = _make_gateway()
    resolve = AsyncMock(
        return_value={
            "exists": True,
            "effective_config": {"supports_tools": True},
            "registry_config": _registry_config_dict(),
            "input_context_window": 1234,
        }
    )

    with patch.object(llm, "aresolve_model", resolve):
        await llm.ensure_metadata_loaded()
        await llm.ensure_metadata_loaded()

    resolve.assert_awaited_once()
    assert llm.supports_tools is True
    assert llm.metadata is not None
    assert llm.metadata.full_key == "openai/gpt-4o"
    assert llm.input_context_window is not None


def test_gateway_registry_model_does_not_require_client_registry_config():
    class GatewaySettings:
        MODEL_GATEWAY_URL = PROXY_ENV["MODEL_GATEWAY_URL"]
        MODEL_GATEWAY_API_KEY = PROXY_ENV["MODEL_GATEWAY_API_KEY"]

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    with (
        patch("model_library.model_library_settings", GatewaySettings()),
        patch("model_library.registry_utils.get_registry_config", return_value=None),
    ):
        llm = get_registry_model("newprovider/new-model")

    assert isinstance(llm, GatewayLLM)
    assert llm.provider == "newprovider"
    assert llm.model_name == "new-model"
    assert llm.supports_images is False
    assert llm.metadata is None


def test_old_gateway_env_names_do_not_activate_proxy_mode():
    class OldGatewaySettings:
        GATEWAY_URL = "http://gateway.test"
        GATEWAY_API_KEY = "old-key"

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    with patch("model_library.model_library_settings", OldGatewaySettings()):
        llm = get_registry_model("openai/gpt-4o")

    assert not isinstance(llm, GatewayLLM)


def test_gateway_repr_redacts_unsynced_metadata():
    llm = _make_gateway()
    rendered = repr(llm)

    assert "supports_temperature=True" not in rendered
    assert "supports_tools=False" not in rendered
    assert "<unloaded: call ensure_metadata_loaded()>" in rendered


async def test_gateway_query_rejects_custom_retrier_before_network():
    llm = _make_gateway()
    llm.custom_retrier = lambda query_func: query_func

    with pytest.raises(GatewayMethodNotSupported, match="custom_retrier"):
        await llm.query("hi")


async def test_gateway_query_rejects_unsupported_extra_parameters_before_network():
    llm = _make_gateway()

    with pytest.raises(GatewayMethodNotSupported, match="extra parameter"):
        await llm.query("hi", stop=["done"])


async def test_gateway_query_explicit_identity_overrides_settings_identity():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )
    settings_identity = {"user": "settings"}
    explicit_identity = {"user": "explicit"}

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV, IDENTITY=json.dumps(settings_identity)),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query("hi", identity=explicit_identity)

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == explicit_identity


async def test_gateway_query_empty_explicit_identity_does_not_fall_back_to_settings_identity():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV, IDENTITY=json.dumps({"user": "settings"})),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query("hi", identity={})

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == {}


async def test_gateway_query_explicit_none_identity_uses_settings_identity():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )
    settings_identity = {"user": "settings"}

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV, IDENTITY=json.dumps(settings_identity)),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query("hi", identity=None)

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == settings_identity


async def test_gateway_query_invalid_explicit_identity_raises_validation_error():
    llm = _make_gateway()

    with pytest.raises(ValidationError, match="identity"):
        await llm.build_body([TextInput(text="hi")], identity=[])


async def test_gateway_query_accepts_client_side_logger_kwarg():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query("hi", logger=logging.getLogger("test.gateway.local"))

    body = json.loads(mock_post.call_args[1]["content"])
    assert "logger" not in body


async def test_gateway_query_logs_started_and_completed_locally(caplog):
    llm = _make_gateway()
    logger = logging.getLogger("test.gateway.local")
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query(
            "hi", logger=logger, run_id="run", question_id="q", query_id="qry"
        )

    body = json.loads(mock_post.call_args[1]["content"])
    messages = [record.getMessage() for record in caplog.records]
    assert "logger" not in body
    assert any(
        record.name == "test.gateway.local.<question=q><query=qry>"
        for record in caplog.records
    )
    assert any(message.startswith("Query started:") for message in messages)
    assert any(message.startswith("Query completed:") for message in messages)


async def test_gateway_query_http_retry_log_includes_run_question_query_and_identity(
    caplog, monkeypatch: pytest.MonkeyPatch
):
    llm = _make_gateway()
    identity = {"team": "evals", "user": "person@example.com"}
    retry_response = httpx.Response(429, json={"detail": "busy"})
    success_response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    async def no_sleep(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr("model_library.base.gateway.random.uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        "model_library.base.gateway._sleep_before_gateway_retry", no_sleep
    )
    with (
        caplog.at_level(logging.WARNING, logger="llm.gateway"),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=[retry_response, success_response],
        ),
    ):
        await llm.query(
            "hi",
            run_id="run",
            question_id="q",
            query_id="qry",
            identity=identity,
        )

    retry_messages = [
        record.getMessage() for record in caplog.records if record.name == "llm.gateway"
    ]
    assert retry_messages == [
        "gateway_http_retry path=/query attempt=1 max_attempts=8 "
        'run_id=run question_id=q query_id=qry identity={"team":"evals","user":"person@example.com"} '
        "status_code=429 retry_after_s=0.000"
    ]


async def test_gateway_query_transport_retry_log_includes_run_question_query_and_identity(
    caplog, monkeypatch: pytest.MonkeyPatch
):
    llm = _make_gateway()
    identity = {"team": "evals", "user": "person@example.com"}
    success_response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    async def no_sleep(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr("model_library.base.gateway.random.uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        "model_library.base.gateway._sleep_before_gateway_retry", no_sleep
    )
    with (
        caplog.at_level(logging.WARNING, logger="llm.gateway"),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=[httpx.ReadError("read failed"), success_response],
        ),
    ):
        await llm.query(
            "hi",
            run_id="run",
            question_id="q",
            query_id="qry",
            identity=identity,
        )

    retry_messages = [
        record.getMessage() for record in caplog.records if record.name == "llm.gateway"
    ]
    assert retry_messages == [
        "gateway_http_retry path=/query attempt=1 max_attempts=8 "
        'run_id=run question_id=q query_id=qry identity={"team":"evals","user":"person@example.com"} '
        "error_type=ReadError retry_after_s=0.000"
    ]


async def test_gateway_query_in_agent_suppresses_local_info_logs(caplog):
    llm = _make_gateway()
    logger = logging.getLogger("test.gateway.agent")
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        await llm.query("hi", logger=logger, in_agent=True, query_id="qry")

    messages = [record.getMessage() for record in caplog.records]
    assert not any(message.startswith("Query started:") for message in messages)
    assert not any(message.startswith("Query completed:") for message in messages)


async def test_gateway_query_logs_history_separately_and_omits_identity(caplog):
    llm = _make_gateway()
    logger = logging.getLogger("test.gateway.history")
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        await llm.query(
            "current",
            history=[TextInput(text="previous")],
            logger=logger,
            identity={"email": "user@example.com"},
            run_id="run",
            question_id="q",
            query_id="qry",
        )

    start_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Query started:")
    ]
    assert len(start_messages) == 1
    start_message = start_messages[0]
    assert "--- input (1):" in start_message
    assert "--- history(1):" in start_message
    assert "identity" not in start_message
    assert "user@example.com" not in start_message


async def test_gateway_query_logs_start_but_not_completion_on_parse_failure(caplog):
    llm = _make_gateway()
    logger = logging.getLogger("test.gateway.failure")
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
            "finish_reason": "invalid",
        },
    )

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(TypeError, match="Gateway finish_reason must be an object"),
    ):
        await llm.query("hi", logger=logger)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("Query started:") for message in messages)
    assert not any(message.startswith("Query completed:") for message in messages)


async def test_gateway_query_prepends_system_prompt_without_leaking_kwarg():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "signed_history": None,
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query("hi", system_prompt="follow policy")

    body = json.loads(mock_post.call_args[1]["content"])
    assert "system_prompt" not in body
    assert body["inputs"] == [
        {"kind": "system", "text": "follow policy"},
        {"kind": "text", "text": "hi"},
    ]


async def test_gateway_query_rejects_duplicate_system_input_before_network():
    llm = _make_gateway()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        with pytest.raises(ValueError, match="At most one SystemInput"):
            await llm.query(
                [SystemInput(text="existing"), TextInput(text="hi")],
                system_prompt="new",
            )

    mock_post.assert_not_called()


async def test_gateway_query_rejects_non_leading_system_input_before_network():
    llm = _make_gateway()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        with pytest.raises(ValueError, match="SystemInput must be the first"):
            await llm.query(
                [SystemInput(text="policy"), TextInput(text="hi")],
                history=[TextInput(text="previous")],
            )

    mock_post.assert_not_called()


def test_gateway_registry_model_does_not_fetch_registry_for_known_model():
    class GatewaySettings:
        MODEL_GATEWAY_URL = PROXY_ENV["MODEL_GATEWAY_URL"]
        MODEL_GATEWAY_API_KEY = PROXY_ENV["MODEL_GATEWAY_API_KEY"]

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    with (
        patch("model_library.model_library_settings", GatewaySettings()),
        patch("model_library.registry_utils.get_registry_config") as mock_get_config,
    ):
        llm = get_registry_model("openai/gpt-4o")

    assert isinstance(llm, GatewayLLM)
    assert llm.provider == "openai"
    assert llm.model_name == "gpt-4o"
    assert llm.supports_temperature is True
    assert llm.metadata is None
    mock_get_config.assert_not_called()


async def test_gateway_registry_model_syncs_metadata_when_called_explicitly():
    class GatewaySettings:
        MODEL_GATEWAY_URL = PROXY_ENV["MODEL_GATEWAY_URL"]
        MODEL_GATEWAY_API_KEY = PROXY_ENV["MODEL_GATEWAY_API_KEY"]

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    response = httpx.Response(
        200,
        json={
            "exists": True,
            "model": "openai/gpt-4o",
            "effective_config": {
                "supports_tools": True,
                "supports_images": False,
                "supports_files": False,
                "supports_videos": False,
                "supports_batch": True,
                "supports_temperature": True,
                "supports_output_schema": False,
                "max_tokens": 456,
            },
            "registry_config": _registry_config_dict(),
            "input_context_window": 123_456,
        },
    )

    with patch("model_library.model_library_settings", GatewaySettings()):
        llm = get_registry_model("openai/gpt-4o")

    assert isinstance(llm, GatewayLLM)

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.sync_model_metadata()

    assert mock_post.call_args[0][0].endswith("/models/resolve")
    assert llm.supports_tools is True
    assert llm.supports_batch is True
    assert llm.batch is not None
    with pytest.raises(GatewayMethodNotSupported, match="batch is not supported"):
        await llm.batch.batch_query("batch", [])
    assert llm.supports_temperature is True
    assert llm.max_tokens == 456
    assert llm.metadata is not None
    assert llm.metadata.full_key == "openai/gpt-4o"
    assert llm.gateway_input_context_window is not None


def _model_keys_with_provider_config() -> list[str]:
    from model_library.register_models import get_model_registry, get_provider_registry

    result: list[str] = []
    for model_key, config in get_model_registry().items():
        provider_class = get_provider_registry().get(config.provider_name)
        if provider_class is None:
            continue
        if getattr(provider_class, "provider_config", None) is not None:
            result.append(model_key)
    return result


@pytest.mark.parametrize("model_key", _model_keys_with_provider_config())
def test_gateway_config_roundtrips_all_provider_configs(model_key: str):
    from model_library.base.base import dump_llm_config, normalize_llm_config_for_model
    from model_library.registry_utils import create_config
    from model_library.register_models import get_provider_registry
    from model_gateway.types import ModelResolveRequest

    registry_config = get_registry_config(model_key)
    assert registry_config is not None
    provider_config_template = getattr(
        get_provider_registry()[registry_config.provider_name], "provider_config"
    )

    config = create_config(registry_config, None)
    config.custom_api_key = SecretStr("provider-key")
    config.custom_endpoint = "https://provider.test/v1"

    dumped = dump_llm_config(config)
    request = ModelResolveRequest.model_validate({"model": model_key, "config": dumped})
    rebuilt = normalize_llm_config_for_model(model_key, request.config)

    assert rebuilt is not None
    assert rebuilt.custom_api_key is not None
    assert rebuilt.custom_api_key.get_secret_value() == "provider-key"
    assert rebuilt.custom_endpoint == "https://provider.test/v1"
    assert isinstance(rebuilt.provider_config, ProviderConfig)
    assert isinstance(rebuilt.provider_config, type(provider_config_template))
    assert config.provider_config is not None
    assert (
        cast(Any, rebuilt.provider_config).model_dump()
        == cast(Any, config.provider_config).model_dump()
    )


# --- proxy query flow ---


async def test_proxy_query_sends_full_input():
    """Client joins history+input and sends as single inputs field."""
    llm = _make_gateway()

    turn1_items = [
        SystemInput(text="sys"),
        TextInput(text="hi"),
        TextInput(text="hello"),
    ]
    turn1_blob = _make_signed_blob(turn1_items)

    first_response = httpx.Response(
        200,
        json={
            "output_text": "hello",
            "tool_calls": [],
            "signed_history": turn1_blob,
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=first_response,
    ) as mock_post:
        result = await llm.query([TextInput(text="hi")])
        assert len(result.history) == 3
        assert isinstance(result.history[0], SystemInput)

        # Verify inputs were sent as flat list
        body = json.loads(mock_post.call_args[1]["content"])
        assert "signed_history" not in body
        assert len(body["inputs"]) == 1
        assert body["inputs"][0]["kind"] == "text"


def _moderation_response_payload(flagged: bool) -> dict[str, object]:
    category_keys = [
        "harassment",
        "harassment/threatening",
        "hate",
        "hate/threatening",
        "illicit",
        "illicit/violent",
        "self-harm",
        "self-harm/instructions",
        "self-harm/intent",
        "sexual",
        "sexual/minors",
        "violence",
        "violence/graphic",
    ]
    return {
        "id": "modr-test",
        "model": "omni-moderation-latest",
        "results": [
            {
                "flagged": flagged,
                "categories": {key: False for key in category_keys},
                "category_scores": {key: 0.0 for key in category_keys},
                "category_applied_input_types": {
                    key: ["text"] for key in category_keys
                },
            }
        ],
    }


async def test_gateway_init_token_retry_only_stores_params_and_query_sends_them():
    llm = _make_gateway()
    token_retry_params = TokenRetryParams(
        input_modifier=1,
        output_modifier=2,
        use_dynamic_estimate=False,
        limit=1000,
        limit_refresh_seconds=60,
    )
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "metadata": {},
        },
    )

    with (
        patch(
            "model_library.base.base.TokenRetrier.init_remaining_tokens",
            new_callable=AsyncMock,
        ) as init_remaining_tokens,
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.init_token_retry(token_retry_params)
        await llm.query([TextInput(text="hi")])

    init_remaining_tokens.assert_not_awaited()
    body = json.loads(mock_post.call_args[1]["content"])
    assert body["token_retry_params"] == {
        "input_modifier": 1.0,
        "output_modifier": 2.0,
        "use_dynamic_estimate": False,
        "limit": 1000,
        "limit_refresh_seconds": 60,
    }


async def test_gateway_token_count_forwards_request_and_returns_count():
    llm = _make_gateway()
    response = httpx.Response(200, json={"tokens": 42})

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        tokens = await llm.count_tokens([TextInput(text="hi")], system_prompt="policy")

    assert tokens == 42
    assert mock_post.call_args[0][0].endswith("/tokens/count")
    body = json.loads(mock_post.call_args[1]["content"])
    assert body["model"] == "openai/gpt-4o"
    assert body["inputs"] == [
        {"kind": "system", "text": "policy"},
        {"kind": "text", "text": "hi"},
    ]


async def test_gateway_rate_limit_calls_reserved_endpoint():
    llm = _make_gateway()
    response = httpx.Response(
        501,
        json={"detail": "Gateway token retry use only"},
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        with pytest.raises(Exception, match="Gateway token retry use only"):
            await llm.get_rate_limit()

    assert mock_post.call_args[0][0].endswith("/rate-limit")
    body = json.loads(mock_post.call_args[1]["content"])
    assert body == {"model": "openai/gpt-4o", "config": {}}


async def test_gateway_registry_model_forwards_only_explicit_override_config():
    class GatewaySettings:
        MODEL_GATEWAY_URL = PROXY_ENV["MODEL_GATEWAY_URL"]
        MODEL_GATEWAY_API_KEY = PROXY_ENV["MODEL_GATEWAY_API_KEY"]

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    with patch("model_library.model_library_settings", GatewaySettings()):
        llm = get_registry_model(
            "openai/gpt-4o", override_config=LLMConfig(temperature=0.25)
        )

    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["config"] == {"temperature": 0.25}


async def test_gateway_syncs_model_metadata_from_gateway():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "exists": True,
            "model": "openai/gpt-4o",
            "effective_config": {
                "supports_tools": True,
                "supports_images": True,
                "supports_files": False,
                "supports_videos": False,
                "supports_batch": True,
                "supports_temperature": True,
                "supports_output_schema": False,
                "max_tokens": 123,
            },
            "registry_config": _registry_config_dict(),
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.sync_model_metadata()

    body = json.loads(mock_post.call_args[1]["content"])
    assert mock_post.call_args[0][0].endswith("/models/resolve")
    assert body == {
        "model": "openai/gpt-4o",
        "config": {},
    }
    assert llm.supports_tools is True
    assert llm.supports_images is True
    assert llm.supports_batch is True
    assert llm.batch is not None
    with pytest.raises(GatewayMethodNotSupported, match="batch is not supported"):
        await llm.batch.batch_query("batch", [])
    assert llm.supports_temperature is True
    assert llm.max_tokens == 123
    assert llm.metadata is not None
    assert llm.metadata.full_key == "openai/gpt-4o"


def test_gateway_mode_legacy_metadata_helpers_raise_without_registry_fetch():
    class GatewaySettings:
        MODEL_GATEWAY_URL = PROXY_ENV["MODEL_GATEWAY_URL"]
        MODEL_GATEWAY_API_KEY = PROXY_ENV["MODEL_GATEWAY_API_KEY"]

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    helpers = [
        lambda: get_registry_config("openai/gpt-4o"),
        lambda: get_model_cost("openai/gpt-4o"),
        lambda: get_model_input_context_window("openai/gpt-4o"),
        get_model_names,
    ]
    with (
        patch("model_library.model_library_settings", GatewaySettings()),
        patch("model_library.registry_utils.get_model_registry") as mock_registry,
    ):
        for helper in helpers:
            with pytest.raises(RuntimeError, match="ensure_metadata_loaded"):
                helper()

    mock_registry.assert_not_called()


async def test_proxy_query_uses_settings_identity_and_ids_when_ids_are_not_explicit():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    identity = {
        "email": "user@example.com",
        "benchmark_name": "swebench",
        "custom": {"team": "evals", "priority": 3, "tags": ["nightly", "dev"]},
    }
    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                IDENTITY=json.dumps(identity),
                RUN_ID="run-from-settings",
                QUESTION_ID="question-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == identity
    assert body["run_id"] == "run-from-settings"
    assert body["question_id"] == "question-from-settings"
    assert isinstance(body["query_id"], str)
    assert body["query_id"]


async def test_proxy_query_generates_run_question_and_query_ids_without_env_ids():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch("model_library.model_library_settings", _GatewaySettings(**PROXY_ENV)),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert isinstance(body["run_id"], str)
    assert len(body["run_id"]) == 8
    assert isinstance(body["question_id"], str)
    assert len(body["question_id"]) == 14
    assert isinstance(body["query_id"], str)
    assert len(body["query_id"]) == 14


async def test_proxy_query_treats_blank_settings_ids_as_absent():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV, RUN_ID="", QUESTION_ID=" "),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert isinstance(body["run_id"], str)
    assert len(body["run_id"]) == 8
    assert isinstance(body["question_id"], str)
    assert len(body["question_id"]) == 14


@pytest.mark.parametrize(
    "identity_env",
    [
        "not-json",
        "[]",
        json.dumps({"value": float("nan")}),
        json.dumps({"large": "x" * 4097}),
        json.dumps(
            {
                "d1": {
                    "d2": {
                        "d3": {"d4": {"d5": {"d6": {"d7": {"d8": {"d9": "too-deep"}}}}}}
                    }
                }
            }
        ),
    ],
)
async def test_proxy_query_omits_invalid_identity_setting(identity_env: str):
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV, IDENTITY=identity_env),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert "identity" not in body


async def test_proxy_query_keeps_exact_identity_boundaries():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )
    exact_size_identity = _identity_with_compact_json_size(4096)
    exact_depth_identity = {
        "d1": {"d2": {"d3": {"d4": {"d5": {"d6": {"d7": {"d8": "ok"}}}}}}}
    }

    for identity in (exact_size_identity, exact_depth_identity):
        with (
            patch(
                "model_library.model_library_settings",
                _GatewaySettings(**PROXY_ENV, IDENTITY=json.dumps(identity)),
            ),
            patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                return_value=response,
            ) as mock_post,
        ):
            await llm.query([TextInput(text="hi")])

        body = json.loads(mock_post.call_args[1]["content"])
        assert body["identity"] == identity


async def test_proxy_query_preserves_identity_keys_and_ignores_query_id_env():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )
    identity = {
        "email": "user@example.com",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
    }

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                IDENTITY=json.dumps(identity),
                QUERY_ID="query-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == identity
    assert body["query_id"] != "query-from-settings"


async def test_proxy_query_uses_task_id_setting_when_question_id_absent():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                RUN_ID="run-from-settings",
                TASK_ID="task-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["run_id"] == "run-from-settings"
    assert body["question_id"] == "task-from-settings"


async def test_proxy_query_uses_task_id_setting_when_question_id_blank():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                QUESTION_ID=" ",
                TASK_ID="task-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["question_id"] == "task-from-settings"


async def test_proxy_query_question_id_setting_precedes_task_id_setting():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                RUN_ID="run-from-settings",
                QUESTION_ID="question-from-settings",
                TASK_ID="task-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["question_id"] == "question-from-settings"


async def test_proxy_query_explicit_none_ids_use_settings_ids():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                RUN_ID="run-from-settings",
                QUESTION_ID="question-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query(
            [TextInput(text="hi")],
            run_id=None,
            question_id=None,
        )

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["run_id"] == "run-from-settings"
    assert body["question_id"] == "question-from-settings"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run_id": ""}, "run_id must not be blank"),
        ({"run_id": " "}, "run_id must not be blank"),
        ({"question_id": ""}, "question_id must not be blank"),
        ({"question_id": "\t"}, "question_id must not be blank"),
    ],
)
async def test_proxy_query_rejects_blank_explicit_ids(
    kwargs: dict[str, Any], match: str
):
    llm = GatewayLLM("gpt-4o", "openai")

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                RUN_ID="run-from-settings",
                QUESTION_ID="question-from-settings",
            ),
        ),
        pytest.raises(ValueError, match=match),
    ):
        await llm.query([TextInput(text="hi")], **cast(dict[str, Any], kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run_id": ""}, "run_id must not be blank"),
        ({"run_id": " "}, "run_id must not be blank"),
        ({"question_id": ""}, "question_id must not be blank"),
        ({"question_id": "\t"}, "question_id must not be blank"),
    ],
)
async def test_gateway_build_body_rejects_blank_explicit_run_and_question_ids(
    kwargs: dict[str, Any], match: str
):
    llm = GatewayLLM("gpt-4o", "openai")

    with pytest.raises(ValueError, match=match):
        await llm.build_body(
            [TextInput(text="hi")], tools=[], **cast(dict[str, Any], kwargs)
        )


async def test_proxy_query_explicit_ids_override_settings_ids():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    identity = {"custom_context": "kept"}
    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(
                **PROXY_ENV,
                IDENTITY=json.dumps(identity),
                RUN_ID="run-from-settings",
                QUESTION_ID="question-from-settings",
            ),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_post,
    ):
        await llm.query(
            [TextInput(text="hi")],
            run_id="run-explicit",
            question_id="question-explicit",
            query_id="query-explicit",
        )

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["identity"] == identity
    assert body["run_id"] == "run-explicit"
    assert body["question_id"] == "question-explicit"
    assert body["query_id"] == "query-explicit"
    assert mock_post.call_args[1]["headers"] == {
        "X-Run-Id": "run-explicit",
        "X-Question-Id": "question-explicit",
        "X-Query-Id": "query-explicit",
    }


async def test_proxy_query_bounds_correlation_headers():
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query(
            [TextInput(text="hi")],
            run_id="r" * 200,
            question_id="q" * 200,
            query_id="z" * 200,
        )

    assert mock_post.call_args[1]["headers"] == {
        "X-Run-Id": "r" * 128,
        "X-Question-Id": "q" * 128,
        "X-Query-Id": "z" * 128,
    }


@pytest.mark.parametrize("query_id", ["", " ", "\t"])
async def test_proxy_query_treats_blank_explicit_query_id_as_absent(query_id: str):
    llm = GatewayLLM("gpt-4o", "openai")
    response = httpx.Response(
        200,
        json={"output_text": "ok", "tool_calls": [], "metadata": {}},
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query([TextInput(text="hi")], query_id=query_id)

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["query_id"] != query_id
    assert len(body["query_id"]) == 14


async def test_proxy_query_forwards_override_config():
    llm = GatewayLLM(
        "gpt-4o",
        "openai",
        config=LLMConfig(
            max_tokens=123,
            custom_api_key=SecretStr("provider-key"),
            custom_endpoint="https://provider.test/v1",
            provider_config=OpenAIConfig(verbosity="low"),
        ),
    )

    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        await llm.query([TextInput(text="hi")])

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["config"]["max_tokens"] == 123
    assert body["config"]["custom_api_key"] == "provider-key"
    assert body["config"]["custom_endpoint"] == "https://provider.test/v1"
    assert body["config"]["provider_config"]["verbosity"] == "low"


async def test_gateway_upload_file_forwards_request_and_returns_file_id():
    llm = GatewayLLM(
        "gpt-4o",
        "openai",
        config=LLMConfig(max_tokens=123, custom_api_key=SecretStr("provider-key")),
    )
    response = httpx.Response(
        200,
        json={
            "file": {
                "kind": "file_base",
                "type": "file",
                "name": "doc.pdf",
                "mime": "application/pdf",
                "append_type": "file_id",
                "file_id": "file-test",
            }
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        file = await llm.upload_file(
            "doc.pdf", "application/pdf", io.BytesIO(b"pdf bytes")
        )

    body = json.loads(mock_post.call_args[1]["content"])
    assert mock_post.call_args[0][0].endswith("/files/upload")
    assert body["model"] == "openai/gpt-4o"
    assert body["name"] == "doc.pdf"
    assert body["mime"] == "application/pdf"
    assert body["content_base64"] == "cGRmIGJ5dGVz"
    assert body["config"]["max_tokens"] == 123
    assert body["config"]["custom_api_key"] == "provider-key"
    assert file == FileWithId(
        type="file",
        name="doc.pdf",
        mime="application/pdf",
        file_id="file-test",
    )


async def test_gateway_get_embedding_forwards_request_and_returns_embedding():
    llm = GatewayLLM(
        "gpt-4o",
        "openai",
        config=LLMConfig(custom_api_key=SecretStr("provider-key")),
    )
    response = httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        embedding = await llm.get_embedding(
            "embed this", model="text-embedding-3-large"
        )

    body = json.loads(mock_post.call_args[1]["content"])
    assert mock_post.call_args[0][0].endswith("/embeddings")
    assert body["model"] == "openai/gpt-4o"
    assert body["text"] == "embed this"
    assert body["embedding_model"] == "text-embedding-3-large"
    assert body["config"]["custom_api_key"] == "provider-key"
    assert embedding == [0.1, 0.2, 0.3]


async def test_gateway_moderate_content_forwards_request_and_parses_response():
    llm = GatewayLLM(
        "moderation",
        "openai",
        config=LLMConfig(custom_api_key=SecretStr("provider-key")),
    )
    response = httpx.Response(
        200, json={"response": _moderation_response_payload(flagged=True)}
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        moderation = await llm.moderate_content("check this")

    body = json.loads(mock_post.call_args[1]["content"])
    assert mock_post.call_args[0][0].endswith("/moderation")
    assert body["model"] == "openai/moderation"
    assert body["text"] == "check this"
    assert body["config"]["custom_api_key"] == "provider-key"
    assert moderation.results[0].flagged is True


@pytest.mark.parametrize(
    ("call", "expected_path"),
    [
        (
            lambda llm: llm.upload_file(
                "doc.pdf",
                "application/pdf",
                io.BytesIO(b"pdf bytes"),
                type="file",
            ),
            "/files/upload",
        ),
        (lambda llm: llm.count_tokens([TextInput(text="hi")]), "/tokens/count"),
        (
            lambda llm: llm.get_embedding("embed this", model="text-embedding-3-large"),
            "/embeddings",
        ),
        (lambda llm: llm.moderate_content("check this"), "/moderation"),
    ],
)
async def test_gateway_non_query_methods_raise_provider_error_envelope(
    call: Callable[[GatewayLLM], Awaitable[object]], expected_path: str
):
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "error": {
                "type": "ProviderError",
                "code": "provider_auth_error",
                "message": "invalid provider key",
                "provider": "openai",
            }
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        with pytest.raises(GatewayProviderError) as exc_info:
            await call(llm)

    assert mock_post.call_args[0][0].endswith(expected_path)
    assert mock_post.await_count == 1
    err = exc_info.value
    assert err.code == "provider_auth_error"
    assert err.provider == "openai"
    assert err.exception_type is None
    assert err.status_code is None
    assert not is_retriable_error(err)


async def test_gateway_provider_error_envelope_code_is_optional():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "error": {
                "type": "ProviderError",
                "message": "provider failed",
                "provider": "openai",
                "exception_type": "RuntimeError",
            }
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ):
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.error_type == "ProviderError"
    assert err.code is None
    assert err.message == "provider failed"
    assert err.provider == "openai"
    assert err.exception_type == "RuntimeError"
    assert err.status_code is None
    assert err.raw_error == response.json()["error"]
    assert str(err) == "ProviderError: provider failed"
    assert not is_retriable_error(err)


async def test_gateway_provider_error_envelope_exposes_raw_code_and_status():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "error": {
                "type": "ProviderError",
                "code": "rate_limit_exceeded",
                "message": "provider says slow down",
                "provider": "openai",
                "exception_type": "ProviderRateLimitError",
                "status_code": 429,
            }
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ):
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.code == "rate_limit_exceeded"
    assert err.exception_type == "ProviderRateLimitError"
    assert err.status_code == 429
    assert str(err) == "ProviderError (rate_limit_exceeded): provider says slow down"
    assert not is_retriable_error(err)


async def test_proxy_query_forwards_pydantic_schema_and_validates_response():
    class Answer(BaseModel):
        value: int

    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "output_text": '{"value": 7}',
            "output_parsed": {"value": 7},
            "tool_calls": [],
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        result = await llm.query([TextInput(text="hi")], output_schema=Answer)

    body = json.loads(mock_post.call_args[1]["content"])
    assert body["output_schema"]["title"] == "Answer"
    assert body["output_schema"]["additionalProperties"] is False
    assert isinstance(result.output_parsed, Answer)
    assert result.output_parsed.value == 7


async def test_proxy_query_preserves_response_metadata_fields():
    llm = _make_gateway()
    response = httpx.Response(
        200,
        json={
            "output_text": "ok",
            "tool_calls": [],
            "metadata": {},
            "finish_reason": {"reason": "stop", "raw": "stop"},
            "extras": {"citations": [{"title": "doc"}]},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ):
        result = await llm.query([TextInput(text="hi")])

    assert result.finish_reason == FinishReasonInfo(
        reason=FinishReason.STOP, raw="stop"
    )
    assert result.extras.citations[0].title == "doc"


async def test_gateway_client_to_server_contract_with_mock_model():
    class Answer(BaseModel):
        value: int

    from model_gateway import main

    class GatewaySettings:
        MODEL_GATEWAY_URL = "http://gateway.test"
        MODEL_GATEWAY_API_KEY = "integration-key"

        def get(self, name: str, default: str | None = None) -> str | None:
            return getattr(self, name, default)

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "integration-key"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    provider_obj = {"role": "assistant", "content": "provider object"}
    calls: list[tuple[list[InputItem], dict[str, Any]]] = []
    seen: dict[str, object] = {}

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            if len(calls) == 1:
                return QueryResult(
                    output_text='{"value": 7}',
                    output_parsed={"value": 7},
                    history=[*inputs, RawResponse(response=provider_obj)],
                )

            return QueryResult(
                output_text="done",
                history=[*inputs, TextInput(text="done")],
            )

    def fake_get_registry_model(model, config):
        seen["model"] = model
        seen["config"] = config
        return FakeLLM()

    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(
            model_helpers, "get_registry_model", side_effect=fake_get_registry_model
        ),
    ):
        app = main.create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers={
                "Authorization": "Bearer integration-key",
                "Content-Type": "application/json",
            },
        ) as client:
            with (
                patch("model_library.model_library_settings", GatewaySettings()),
                patch(
                    "model_library.base.gateway.gateway_httpx_client",
                    return_value=client,
                ),
            ):
                llm = GatewayLLM(
                    "gpt-4o",
                    "openai",
                    config=LLMConfig(
                        max_tokens=17,
                        custom_api_key=SecretStr("provider-key"),
                        custom_endpoint="https://provider.test/v1",
                        provider_config=OpenAIConfig(verbosity="low"),
                    ),
                )
                first = await llm.query([TextInput(text="hi")], output_schema=Answer)
                second = await llm.query(
                    [TextInput(text="again")], history=first.history
                )

    assert seen["model"] == "openai/gpt-4o"
    assert isinstance(seen["config"], LLMConfig)
    assert seen["config"].max_tokens == 17
    assert seen["config"].custom_api_key is not None
    assert seen["config"].custom_api_key.get_secret_value() == "provider-key"
    assert seen["config"].custom_endpoint == "https://provider.test/v1"
    assert isinstance(seen["config"].provider_config, OpenAIConfig)
    assert seen["config"].provider_config.verbosity == "low"
    assert first.output_parsed == Answer(value=7)
    assert second.output_text == "done"
    assert len(calls) == 2
    output_schema = calls[0][1]["output_schema"]
    assert isinstance(output_schema, dict)
    assert output_schema["title"] == "Answer"
    assert isinstance(first.history[1], RawResponse)
    assert isinstance(first.history[1].response, dict)
    assert "pickle" in first.history[1].response
    assert isinstance(calls[1][0][1], RawResponse)
    assert calls[1][0][1].response == provider_obj


async def test_proxy_query_history_joined_into_inputs():
    """When history is passed, it's joined with input into a single inputs field."""
    llm = _make_gateway()

    turn1_items = [SystemInput(text="sys"), TextInput(text="hi")]
    turn1_blob = _make_signed_blob(turn1_items)

    first_response = httpx.Response(
        200,
        json={
            "output_text": "hello",
            "tool_calls": [],
            "signed_history": turn1_blob,
            "metadata": {},
        },
    )
    second_response = httpx.Response(
        200,
        json={
            "output_text": "world",
            "tool_calls": [],
            "signed_history": _make_signed_blob(
                turn1_items + [TextInput(text="follow up")]
            ),
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=[first_response, second_response],
    ) as mock_post:
        result1 = await llm.query([TextInput(text="hi")])

        # Pass result.history as history
        result2 = await llm.query(
            [TextInput(text="follow up")], history=result1.history
        )

        # Second call should have history items + new input in inputs
        second_body = json.loads(mock_post.call_args_list[1][1]["content"])
        assert len(second_body["inputs"]) == 3  # sys + hi + follow up
        assert second_body["inputs"][0]["kind"] == "system"
        assert result2.output_text == "world"


async def test_raw_response_echoed_as_blob():
    """RawResponse fields from server stay as opaque blobs on the client."""
    llm = _make_gateway()
    items = [TextInput(text="hi"), RawResponse(response={"role": "assistant"})]
    blob = _make_signed_blob(items)

    response = httpx.Response(
        200,
        json={
            "output_text": "reply",
            "tool_calls": [],
            "signed_history": blob,
            "metadata": {},
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ):
        result = await llm.query([TextInput(text="hi")])

    # Client should have RawResponse with blob (not unpickled)
    raw = [i for i in result.history if isinstance(i, RawResponse)]
    assert len(raw) == 1
    # The response field should be a string or dict (blob), not the original object
    assert isinstance(raw[0].response, (str, dict))


@pytest.mark.parametrize(
    "side_effect",
    [
        [
            httpx.ReadError("server disconnected"),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
        [
            httpx.Response(502, text="<html>Bad Gateway</html>"),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
        [
            httpx.Response(
                429,
                json={
                    "code": "gateway_overloaded",
                    "message": "Gateway request queue is full",
                },
            ),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
        [
            httpx.Response(
                429,
                json={"code": "unexpected", "message": "actual gateway throttling"},
            ),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
        [
            httpx.Response(
                502,
                json={"code": "provider_auth_error", "message": "invalid provider key"},
            ),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
        [
            httpx.Response(
                502,
                json={
                    "code": "internal_error",
                    "message": "upstream mentioned HTTP 400",
                },
            ),
            httpx.Response(
                200,
                json={"output_text": "ok", "tool_calls": [], "metadata": {}},
            ),
        ],
    ],
    ids=[
        "transport-error",
        "non-json-5xx",
        "gateway-overloaded-429",
        "actual-429-regardless-body-code",
        "actual-5xx-with-provider-code",
        "actual-5xx-message-mentions-400",
    ],
)
async def test_proxy_mode_retries_retryable_gateway_failures(
    side_effect: list[httpx.Response | Exception],
    caplog: pytest.LogCaptureFixture,
):
    llm = _make_gateway()
    caplog.set_level(logging.WARNING, logger="llm.gateway")

    with (
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ) as mock_post,
        patch("model_library.base.gateway.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await llm.query([TextInput(text="hi")])

    assert result.output_text == "ok"
    assert mock_post.await_count == 2
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "llm.gateway"
    assert "gateway_http_retry path=/query attempt=1 max_attempts=8" in caplog.text


async def test_proxy_mode_200_provider_error_envelope_does_not_retry():
    llm = _make_gateway()
    provider_error = httpx.Response(
        200,
        json={
            "error": {
                "type": "ProviderError",
                "code": "provider_auth_error",
                "message": "invalid provider key",
                "provider": "openai",
                "details": {"retry_after": 12},
            }
        },
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=provider_error,
    ) as mock_post:
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.error_type == "ProviderError"
    assert err.code == "provider_auth_error"
    assert err.provider == "openai"
    assert err.exception_type is None
    assert err.status_code is None
    assert err.raw_error == provider_error.json()["error"]
    assert str(err) == "ProviderError (provider_auth_error): invalid provider key"
    assert not is_retriable_error(err)
    assert mock_post.await_count == 1


@pytest.mark.parametrize("raw_error", [None, "not-an-object"])
async def test_proxy_mode_malformed_200_error_envelope_raises_typed_error(
    raw_error: object,
):
    llm = _make_gateway()
    provider_error = httpx.Response(200, json={"error": raw_error})

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=provider_error,
    ):
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.error_type == "GatewayError"
    assert err.code == "malformed_error_envelope"
    assert err.raw_error == raw_error
    assert not is_retriable_error(err)


@pytest.mark.parametrize("payload", ["not-an-object", ["not", "an", "object"]])
async def test_proxy_mode_malformed_200_success_body_raises_typed_error(
    payload: object,
):
    llm = _make_gateway()
    response = httpx.Response(200, json=payload)

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.error_type == "GatewayError"
    assert err.code == "malformed_gateway_response"
    assert err.raw_error == payload
    assert not is_retriable_error(err)
    assert mock_post.await_count == 1


async def test_proxy_mode_non_json_200_success_body_raises_typed_error():
    llm = _make_gateway()
    response = httpx.Response(200, text="<html>oops</html>")

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=response,
    ) as mock_post:
        with pytest.raises(GatewayProviderError) as exc_info:
            await llm.query([TextInput(text="hi")])

    err = exc_info.value
    assert err.error_type == "GatewayError"
    assert err.code == "malformed_gateway_response"
    assert err.raw_error == "<html>oops</html>"
    assert not is_retriable_error(err)
    assert mock_post.await_count == 1


async def test_proxy_mode_actual_http_400_does_not_retry():
    llm = _make_gateway()
    error_response = httpx.Response(
        400, json={"code": "invalid_model", "message": "Model x/y not found"}
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=error_response,
    ) as mock_post:
        with pytest.raises(Exception, match="Gateway error"):
            await llm.query([TextInput(text="hi")])

    assert mock_post.await_count == 1


# --- serialize_input / deserialize_input ---


def test_serialize_input_returns_json_string():
    items = [TextInput(text="hello"), SystemInput(text="sys")]
    result = LLM.serialize_input(items)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["kind"] == "text"
    assert parsed[1]["kind"] == "system"


def test_serialize_pickles_raw_response():
    """RawResponse.response is pickled+base64 inline."""
    provider_obj = {"role": "assistant", "content": "hi", "nested": [1, 2, 3]}
    item = RawResponse(response=provider_obj)
    result = json.loads(LLM.serialize_input([item]))

    assert result[0]["kind"] == "raw_response"
    # Without secret: plain base64 string
    assert isinstance(result[0]["response"], str)


def test_serialize_pickles_raw_input():
    """RawInput.input is pickled+base64 inline."""
    provider_obj = {"messages": [{"role": "user", "content": "hello"}]}
    item = RawInput(input=provider_obj)
    result = json.loads(LLM.serialize_input([item]))

    assert result[0]["kind"] == "raw_input"
    assert isinstance(result[0]["input"], str)


def test_serialize_leaves_text_input_unchanged():
    item = TextInput(text="hello")
    result = json.loads(LLM.serialize_input([item]))
    assert result[0] == {"kind": "text", "text": "hello"}


def test_serialize_with_secret_adds_hmac():
    """With a secret, pickled fields get {pickle, hmac} dicts."""
    item = RawResponse(response={"key": "value"})
    result = json.loads(LLM.serialize_input([item], secret=b"test-secret"))

    field = result[0]["response"]
    assert isinstance(field, dict)
    assert "pickle" in field
    assert "hmac" in field


def test_deserialize_roundtrip_no_secret():
    provider_obj = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    items = [RawResponse(response=provider_obj), TextInput(text="hello")]

    serialized = LLM.serialize_input(items)
    restored = LLM.deserialize_input(serialized)

    assert len(restored) == 2
    assert isinstance(restored[0], RawResponse)
    assert restored[0].response == provider_obj
    assert isinstance(restored[1], TextInput)
    assert restored[1].text == "hello"


def test_deserialize_roundtrip_with_secret():
    secret = b"my-secret"
    provider_obj = {"role": "assistant", "content": "hi"}
    items = [RawResponse(response=provider_obj), TextInput(text="hello")]

    serialized = LLM.serialize_input(items, secret=secret)
    restored = LLM.deserialize_input(serialized, secret=secret)

    assert len(restored) == 2
    assert isinstance(restored[0], RawResponse)
    assert restored[0].response == provider_obj


def test_deserialize_rejects_tampered_hmac():
    secret = b"my-secret"
    item = RawResponse(response={"key": "value"})
    serialized = LLM.serialize_input([item], secret=secret)

    # Tamper with the HMAC
    data = json.loads(serialized)
    data[0]["response"]["hmac"] = "deadbeef" * 8
    tampered = json.dumps(data)

    with pytest.raises(ValueError, match="HMAC verification failed"):
        LLM.deserialize_input(tampered, secret=secret)


def test_deserialize_rejects_unsigned_when_secret_expected():
    """If secret is given but blob is unsigned, reject it."""
    item = RawResponse(response={"key": "value"})
    # Serialize without secret
    serialized = LLM.serialize_input([item])

    with pytest.raises(ValueError, match="Expected HMAC-signed pickle blob"):
        LLM.deserialize_input(serialized, secret=b"my-secret")


def test_deserialize_from_bytes():
    items = [TextInput(text="hello")]
    serialized = LLM.serialize_input(items)
    restored = LLM.deserialize_input(serialized.encode())
    assert len(restored) == 1
    assert isinstance(restored[0], TextInput)
    assert restored[0].text == "hello"


def test_deserialize_from_file(tmp_path):
    items = [TextInput(text="hello")]
    serialized = LLM.serialize_input(items)
    path = tmp_path / "history.json"
    path.write_text(serialized)
    restored = LLM.deserialize_input(path)
    assert len(restored) == 1
    assert isinstance(restored[0], TextInput)
    assert restored[0].text == "hello"


def test_raw_input_roundtrip_with_secret():
    secret = b"my-secret"
    provider_obj = {"messages": [{"role": "user", "content": "hello"}]}
    items = [RawInput(input=provider_obj)]

    serialized = LLM.serialize_input(items, secret=secret)
    restored = LLM.deserialize_input(serialized, secret=secret)

    assert isinstance(restored[0], RawInput)
    assert restored[0].input == provider_obj


def test_restore_raw_fields_with_hmac():
    """restore_raw_fields unpickles signed blobs in-place."""
    secret = b"test-secret"
    provider_obj = {"role": "assistant", "content": "hi"}
    items = [RawResponse(response=provider_obj), TextInput(text="hello")]

    # Serialize with HMAC (as server would)
    serialized = LLM.serialize_input(items, secret=secret)

    # Parse JSON into InputItems (as Pydantic would on the server)
    from pydantic import TypeAdapter

    adapter = TypeAdapter(list[InputItem])
    parsed = list(adapter.validate_json(serialized))

    assert isinstance(parsed[0], RawResponse)
    assert isinstance(parsed[1], TextInput)
    # Raw field is still a blob
    assert isinstance(parsed[0].response, dict)

    # Restore in-place
    LLM.restore_raw_fields(parsed, secret=secret)

    assert parsed[0].response == provider_obj
    assert parsed[1].text == "hello"


def test_restore_raw_fields_rejects_tampered():
    """restore_raw_fields rejects tampered HMAC blobs."""
    secret = b"test-secret"
    items = [RawResponse(response={"key": "value"})]
    serialized = LLM.serialize_input(items, secret=secret)

    from pydantic import TypeAdapter

    # Tamper with HMAC
    data = json.loads(serialized)
    data[0]["response"]["hmac"] = "deadbeef" * 8

    adapter = TypeAdapter(list[InputItem])
    parsed = list(adapter.validate_json(json.dumps(data)))

    with pytest.raises(ValueError, match="HMAC verification failed"):
        LLM.restore_raw_fields(parsed, secret=secret)


@pytest.mark.anyio
async def test_gateway_query_preserves_provider_tool_events():
    llm = _make_gateway()
    event = {
        "provider": "anthropic",
        "type": "web_search_tool_result",
        "name": "web_search",
        "status": "error",
        "input": "current AAPL stock price",
        "output": "unavailable",
        "sequence": 0,
        "id": "tu_abc123",
    }
    response = httpx.Response(
        200,
        json={
            "output_text": None,
            "tool_calls": [],
            "provider_tool_events": [event],
            "metadata": {},
        },
    )

    with (
        patch(
            "model_library.model_library_settings",
            _GatewaySettings(**PROXY_ENV),
        ),
        patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        result = await llm.query("hi")

    assert len(result.provider_tool_events) == 1
    e = result.provider_tool_events[0]
    assert isinstance(e, ProviderToolEvent)
    assert e.provider == "anthropic"
    assert e.name == "web_search"
    assert e.status == "error"
    assert e.input == "current AAPL stock price"
    assert e.output == "unavailable"
