"""Consumer-facing sampling behavior at chat request assembly boundaries."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from model_library.base import LLM, LLMConfig
from model_library.base.input import TextInput
from model_library.providers.ai21labs import AI21LabsModel
from model_library.providers.amazon.chat import AmazonModel
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.providers.delegates.openrouter.openrouter import OpenRouterModel
from model_library.providers.google.google import GoogleConfig, GoogleModel
from model_library.providers.mistral.chat import MistralModel
from model_library.providers.openai.openai import OpenAIModel
from model_library.providers.vals import DummyAIModel
from model_library.providers.xai import XAIModel
from model_library.registry_utils import get_registry_model


_INPUT = [TextInput(text="hello")]
_LOGGER = logging.getLogger("test.chat.sampling")


def _warnings(caplog):
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def _model(kind, config):
    if kind == "responses":
        return OpenAIModel("gpt-test", config=config, use_completions=False)
    if kind == "completions":
        return OpenAIModel("gpt-test", config=config, use_completions=True)
    if kind == "bedrock":
        return AmazonModel("anthropic.claude-test", config=config)
    if kind == "mistral":
        return MistralModel("mistral-test", config=config)
    if kind == "ai21":
        return AI21LabsModel("jamba-test", config=config)
    if kind == "vals":
        return DummyAIModel("vals-test", config=config)
    if kind == "xai":
        return XAIModel("grok-test", config=config)
    if kind == "google":
        return GoogleModel("gemini-test", config=config)
    raise AssertionError(kind)


def _sampling(kind, body):
    if kind == "google":
        config = body["config"]
        return {
            key: getattr(config, key)
            for key in ("temperature", "top_p", "top_k")
            if getattr(config, key) is not None
        }
    if kind == "bedrock":
        inference = body["inferenceConfig"]
        assert "top_p" not in inference
        assert "top_p" not in body
        return {
            key: inference[key]
            for key in ("temperature", "topP")
            if key in inference
        }
    values = {key: body[key] for key in ("temperature", "top_p") if key in body}
    if kind == "completions":
        assert "top_k" not in body
        if "top_k" in body.get("extra_body", {}):
            values["top_k"] = body["extra_body"]["top_k"]
    else:
        assert "top_k" not in body
    return values


@pytest.mark.parametrize(
    ("kind", "supports_top_k"),
    [
        ("responses", False),
        ("completions", True),
        ("bedrock", False),
        ("mistral", False),
        ("ai21", False),
        ("vals", False),
        ("xai", False),
        ("google", True),
    ],
)
async def test_leaf_config_sampling_and_raw_kwargs_override(
    kind, supports_top_k, caplog
):
    configured = dict(temperature=0.4, top_p=0.8, top_k=11)
    model = _model(kind, LLMConfig(**configured))
    p_key = "topP" if kind == "bedrock" else "top_p"

    with caplog.at_level(logging.WARNING, logger="llm"):
        inherited = _sampling(kind, await model.build_body(_INPUT, tools=[]))
    assert inherited == {
        "temperature": 0.4,
        p_key: 0.8,
        **({"top_k": 11} if supports_top_k else {}),
    }
    assert len(_warnings(caplog)) == (0 if supports_top_k else 1)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        replaced = await model.build_body(
            _INPUT, tools=[], temperature=0, top_p=None, top_k=0
        )
    if kind == "google":
        assert replaced["config"].temperature == 0
        assert replaced["config"].top_p is None
        assert replaced["config"].top_k == 0
    elif kind == "bedrock":
        assert replaced["inferenceConfig"]["temperature"] == 0
        assert replaced["inferenceConfig"]["topP"] == 0.8
        assert replaced["inferenceConfig"]["top_p"] is None
        assert replaced["inferenceConfig"]["top_k"] == 0
    else:
        assert replaced["temperature"] == 0
        assert replaced["top_p"] is None
        assert replaced["top_k"] == 0
        if kind == "completions":
            assert replaced["extra_body"]["top_k"] == 11
    assert len(_warnings(caplog)) == (0 if supports_top_k else 1)

    caplog.clear()
    disabled = _model(
        kind,
        LLMConfig(
            **configured,
            supports_temperature=False,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        assert _sampling(kind, await disabled.build_body(_INPUT, tools=[])) == {}
    assert {record.getMessage() for record in _warnings(caplog)} == {
        f"Omitting unsupported configured sampling parameter {name}"
        for name in ("temperature", "top_p", "top_k")
    }

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        raw_none = await disabled.build_body(
            _INPUT, tools=[], temperature=None, top_p=None, top_k=None
        )
    if kind == "google":
        assert raw_none["config"].temperature is None
        assert raw_none["config"].top_p is None
        assert raw_none["config"].top_k is None
    elif kind == "bedrock":
        assert all(
            raw_none["inferenceConfig"][key] is None
            for key in ("temperature", "top_p", "top_k")
        )
    else:
        assert all(raw_none[key] is None for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        raw_zero = await disabled.build_body(
            _INPUT, tools=[], temperature=0, top_p=0, top_k=0
        )
    if kind == "google":
        assert (
            raw_zero["config"].temperature,
            raw_zero["config"].top_p,
            raw_zero["config"].top_k,
        ) == (0, 0, 0)
    elif kind == "bedrock":
        assert all(
            raw_zero["inferenceConfig"][key] == 0
            for key in ("temperature", "top_p", "top_k")
        )
    else:
        assert all(raw_zero[key] == 0 for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3

    caplog.clear()
    configured_none = _model(
        kind,
        LLMConfig(
            temperature=None,
            top_p=None,
            top_k=None,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        assert _sampling(kind, await configured_none.build_body(_INPUT, tools=[])) == {}
    assert _warnings(caplog) == []

    caplog.clear()
    configured_zero = _model(
        kind,
        LLMConfig(temperature=0, top_p=0, top_k=0),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        assert _sampling(kind, await configured_zero.build_body(_INPUT, tools=[])) == {
            "temperature": 0,
            p_key: 0,
            **({"top_k": 0} if supports_top_k else {}),
        }
    assert len(_warnings(caplog)) == (0 if supports_top_k else 1)


@pytest.mark.parametrize(
    "reasoning,provider,expected",
    [
        (False, "anthropic", {"temperature": 0.4}),
        (True, "anthropic", {}),
        (True, "minimax", {"top_p": 0.8}),
    ],
)
async def test_anthropic_sampling_eligibility(reasoning, provider, expected, caplog):
    model = AnthropicModel(
        "claude-test",
        provider=provider,
        config=LLMConfig(
            max_tokens=4096, reasoning=reasoning,
            temperature=0.4, top_p=0.8, top_k=11,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        body = await model.build_body(_INPUT, tools=[])
    assert {key: body[key] for key in ("temperature", "top_p") if key in body} == expected
    assert "top_k" not in body
    assert len(_warnings(caplog)) == 3 - len(expected)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        raw_none = await model.build_body(
            _INPUT, tools=[], temperature=None, top_p=None, top_k=None
        )
    assert all(raw_none[key] is None for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3 - len(expected)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        zero = await model.build_body(_INPUT, tools=[], temperature=0, top_p=0, top_k=0)
    assert all(zero[key] == 0 for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3 - len(expected)


async def test_anthropic_disabled_config_does_not_gate_raw_kwargs(caplog):
    model = AnthropicModel(
        "claude-test",
        config=LLMConfig(
            max_tokens=4096,
            temperature=0.4,
            top_p=0.8,
            top_k=11,
            supports_temperature=False,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="llm"):
        body = await model.build_body(_INPUT, tools=[])
    assert not any(key in body for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm"):
        body = await model.build_body(
            _INPUT, tools=[], temperature=None, top_p=None, top_k=None
        )
    assert all(body[key] is None for key in ("temperature", "top_p", "top_k"))
    assert len(_warnings(caplog)) == 3


async def test_registry_inherited_temperature_can_be_overridden_with_raw_none(caplog):
    model = get_registry_model("anthropic/claude-sonnet-4-5-20250929")
    assert isinstance(model, AnthropicModel)
    assert model.temperature == 1
    with caplog.at_level(logging.WARNING, logger="llm"):
        inherited = await model.build_body(_INPUT, tools=[])
        raw_none = await model.build_body(_INPUT, tools=[], temperature=None)
    assert inherited["temperature"] == 1
    assert raw_none["temperature"] is None
    assert _warnings(caplog) == []


async def test_completions_raw_extra_body_keeps_explicit_sampling():
    model = _model(
        "completions", LLMConfig(temperature=0.4, top_p=0.8, top_k=11)
    )
    raw = {"top_k": 42, "custom": "preserved"}
    body = await model.build_body(
        _INPUT, tools=[], temperature=None, top_k=0, extra_body=raw
    )
    assert body["temperature"] is None
    assert body["top_k"] == 0
    assert body["extra_body"] == raw
    assert body["extra_body"]["top_k"] == 42
    assert raw == {"top_k": 42, "custom": "preserved"}


async def test_bedrock_explicit_wire_top_p_keeps_existing_precedence():
    model = _model("bedrock", LLMConfig(temperature=0.4, top_p=0.8))
    body = await model.build_body(_INPUT, tools=[], top_p=0, topP=0.6)
    assert body["inferenceConfig"]["topP"] == 0.6
    assert body["inferenceConfig"]["top_p"] == 0


class _SDKIntercept(Exception):
    pass


@pytest.mark.parametrize(
    "provider,budget_mode",
    [
        ("google", False),
        ("google", True),
        ("anthropic", False),
        ("openrouter", False),
    ],
)
async def test_outer_delegation_reaches_real_leaf_body(provider, budget_mode, caplog):
    config = LLMConfig(
        native=False,
        temperature=0.4,
        top_p=0.8,
        top_k=11,
        custom_api_key=SecretStr("test-key"),
        provider_config=(
            GoogleConfig(uses_thinking_budget=budget_mode)
            if provider == "google" else None
        ),
    )
    if provider == "google":
        model = GoogleModel("gemini-test", config=config)
    elif provider == "anthropic":
        model = AnthropicModel("claude-test", config=config)
    else:
        model = OpenRouterModel("router-test", config=config)
    assert model.delegate is not None
    captured = {}

    async def intercept(**body):
        captured.update(body)
        raise _SDKIntercept

    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = intercept
    with (
        patch.object(model.delegate, "get_client", return_value=sdk),
        caplog.at_level(logging.WARNING, logger="llm"),
        pytest.raises(_SDKIntercept),
    ):
        await model._query_impl(
            _INPUT,
            tools=[],
            query_logger=_LOGGER,
            temperature=None,
            top_p=0,
            top_k=0,
            **({"thinking_budget": 0} if provider == "google" else {}),
        )
    assert captured["temperature"] is None
    assert captured["top_p"] == 0
    assert captured["top_k"] == 0
    assert captured["extra_body"]["top_k"] == 11
    if provider == "google":
        assert captured["thinking_budget"] == 0
    assert len(_warnings(caplog)) == 0


@pytest.mark.parametrize(
    "provider,budget_mode",
    [
        ("google", False),
        ("google", True),
        ("anthropic", False),
        ("openrouter", False),
    ],
)
@pytest.mark.parametrize("case", ["inherited", "raw_none", "raw_zero"])
async def test_outer_delegation_only_omits_disabled_config_sampling(
    provider, budget_mode, case, caplog
):
    config = LLMConfig(
        native=False,
        temperature=0.4,
        top_p=0.8,
        top_k=11,
        supports_temperature=False,
        custom_api_key=SecretStr("test-key"),
        provider_config=(
            GoogleConfig(uses_thinking_budget=budget_mode)
            if provider == "google" else None
        ),
    )
    if provider == "google":
        model = GoogleModel("gemini-test", config=config)
    elif provider == "anthropic":
        model = AnthropicModel("claude-test", config=config)
    else:
        model = OpenRouterModel("router-test", config=config)
    assert model.delegate is not None
    captured = {}

    async def intercept(**body):
        captured.update(body)
        raise _SDKIntercept

    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = intercept
    with (
        patch.object(model.delegate, "get_client", return_value=sdk),
        caplog.at_level(logging.WARNING, logger="llm"),
        pytest.raises(_SDKIntercept),
    ):
        kwargs = {
            "inherited": {},
            "raw_none": {"temperature": None, "top_p": None, "top_k": None},
            "raw_zero": {"temperature": 0, "top_p": 0, "top_k": 0},
        }[case]
        await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER, **kwargs)
    if case == "inherited":
        assert not any(key in captured for key in ("temperature", "top_p", "top_k"))
        assert "top_k" not in captured.get("extra_body", {})
    else:
        value = None if case == "raw_none" else 0
        assert all(captured[key] == value for key in ("temperature", "top_p", "top_k"))
        assert "top_k" not in captured.get("extra_body", {})
    assert len(_warnings(caplog)) == 3


@pytest.mark.parametrize(
    "extra,temperature,expected_temperature",
    [
        (None, None, None),
        ({"custom": "preserved"}, 0, 0),
        ({"custom": "preserved", "temperature": 0.7}, 0, 0.7),
    ],
)
async def test_anthropic_native_fallback_owns_extras_after_assembly(
    extra, temperature, expected_temperature, caplog
):
    model = AnthropicModel(
        "claude-test",
        config=LLMConfig(
            max_tokens=4096,
            temperature=0.4,
            provider_config=AnthropicConfig(fallback_models=["claude-fallback"]),
        ),
    )
    captured = {}

    def intercept(**body):
        captured.update(body)
        raise _SDKIntercept

    sdk = MagicMock()
    sdk.beta.messages.stream.side_effect = intercept
    raw = dict(extra) if extra is not None else None
    with (
        patch.object(model, "get_client", return_value=sdk),
        caplog.at_level(logging.WARNING, logger="llm"),
        pytest.raises(_SDKIntercept),
    ):
        await model._query_impl(
            _INPUT,
            tools=[],
            query_logger=_LOGGER,
            temperature=temperature,
            **({"extra_body": raw} if raw is not None else {}),
        )
    extras = captured["extra_body"]
    assert extras["fallbacks"] == [{"model": "claude-fallback"}]
    assert "temperature" not in captured
    if expected_temperature is None:
        assert extras["temperature"] is None
    else:
        assert extras["temperature"] == expected_temperature
    assert raw == extra
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "provider,configured_top_p,kwargs",
    [
        ("anthropic", None, {"top_p": 0.8, "top_k": 11}),
        ("minimax", 0.8, {"top_p": None}),
    ],
)
async def test_anthropic_count_raw_sampling_uses_existing_fallback(
    provider, configured_top_p, kwargs, caplog
):
    model = AnthropicModel(
        "claude-test",
        provider=provider,
        config=LLMConfig(max_tokens=4096, top_p=configured_top_p),
    )
    received = []

    async def native_count(**body):
        received.append(body)
        raise TypeError(
            "AsyncMessages.count_tokens() got an unexpected keyword argument 'top_p'"
        )

    client = MagicMock()
    client.messages.count_tokens = AsyncMock(side_effect=native_count)
    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(LLM, "count_tokens", new_callable=AsyncMock) as fallback,
        caplog.at_level(logging.WARNING, logger="llm"),
    ):
        fallback.return_value = 53
        count = await model.count_tokens(_INPUT, **kwargs)

    assert count == 53
    assert len(received) == 1
    assert received[0]["top_p"] == kwargs["top_p"]
    if provider == "anthropic":
        assert received[0]["top_k"] == 11
    else:
        assert "top_k" not in received[0]
    fallback.assert_awaited_once()
    assert _warnings(caplog) == []
