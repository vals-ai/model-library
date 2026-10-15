"""The typed-decision provider (Jev, Mercury Decide, Microsoft-Decision-1 through OpenRouter, GPT-6 Luna through OpenAI),
on a mocked transport."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import httpx
import pytest
import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from model_library.base import (
    DelegateOnly,
    FinishReason,
    LLMConfig,
    SystemInput,
    TextInput,
    client_registry,
)
from model_gateway.route_helpers import provider_error_from_exception
from model_library.exceptions import NoRetryException, is_retriable_error
from model_library.providers.delegates.openrouter.decision import (
    DecisionConfig,
    DecisionError,
    DecisionModel,
    decision_request,
    parse_decision_output,
)
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import (
    get_registry_config,
    get_registry_model,
    recompute_cost,
)

pytestmark = pytest.mark.unit

# registry key, pinned OpenRouter slug, input price per 1M tokens, the provider OpenRouter reports
MODELS = [
    ("decision/jev-1.13.0", "typesafe/jev-1.13-20260917", 0.042, "TypeSafe"),
    ("decision/mercury-decide", "inception/mercury-decide-20260930", 0.02, "Inception"),
    (
        "decision/microsoft-decision-1",
        "microsoft/microsoft-decision-1-20261009",
        0.042,
        "Microsoft",
    ),
]
JEV = MODELS[0][0]

TEAMS = {
    "billing": "Charges and refunds",
    "engineering": "Bugs and outages",
    "support": "How-to questions",
}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": TEAMS}
}
STATE = "The API returns 500 on every call."
ANSWERS = {
    "team": {
        "type": "choice",
        "choice": "engineering",
        "probabilities": {"billing": 0.01, "engineering": 0.97, "support": 0.02},
        "confidence": 0.9,
    }
}
OPENROUTER_ANSWER = {
    "id": "gen-dec-abc123",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": ANSWERS,
    "usage": {"input_tokens": 500, "output_tokens": 50, "cost": 0.000021},
}

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    client_registry.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    yield
    client_registry.clear()


# The unit conftest replaces every provider's get_client; keep the real ones so the key and client wiring is exercised.
_REAL_DELEGATE_GET_CLIENT = DelegateOnly.get_client
_REAL_OPENAI_GET_CLIENT = OpenAIModel.get_client


def serve(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    """Route the HTTP transport under the real client getter to `handler`, and record each request it receives."""
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def create_client(
        api_key: str, base_url: str | None = None, dns_resolve: object = None
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wrapped)),
            max_retries=0,
        )

    monkeypatch.setattr(DecisionModel, "get_client", _REAL_DELEGATE_GET_CLIENT)
    monkeypatch.setattr(OpenAIModel, "get_client", _REAL_OPENAI_GET_CLIENT)
    monkeypatch.setattr(
        "model_library.providers.openai.openai.create_openai_client_with_defaults",
        create_client,
    )
    return seen


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=OPENROUTER_ANSWER)


async def call(m: DecisionModel, text: str | None = None, **kwargs: object):
    text = text if text is not None else decision_request(STATE, QUESTIONS)
    return await m._query_impl(
        [TextInput(text=text)], tools=[], query_logger=logging.getLogger("t"), **kwargs
    )


# --- request shape, result and cost ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "slug", "price", "upstream"), MODELS)
async def test_every_model_goes_through_openrouter_with_a_pinned_slug_and_its_own_price(
    monkeypatch: pytest.MonkeyPatch, key: str, slug: str, price: float, upstream: str
) -> None:
    payload = {
        **OPENROUTER_ANSWER,
        "model": slug,
        "provider": upstream,
        "usage": {"input_tokens": 500, "output_tokens": 3, "cost": 0.5},
    }
    seen = serve(monkeypatch, lambda _: httpx.Response(200, json=payload))

    model = get_registry_model(
        key, override_config=LLMConfig(custom_api_key=SecretStr("or-config-key"))
    )
    result = await model.query(decision_request(STATE, QUESTIONS))

    assert str(seen[0].url) == "https://openrouter.ai/api/alpha/decisions"
    assert seen[0].headers["authorization"] == "Bearer or-config-key"
    assert json.loads(seen[0].content) == {
        "model": slug,
        "state": STATE,
        "questions": QUESTIONS,
    }  # a dated slug, never the alias
    answers = parse_decision_output(result.output_text)
    assert (
        answers["team"]["choice"] == "engineering"
        and answers["team"]["probabilities"]["engineering"] == 0.97
    )
    assert result.metadata.served_model == slug
    assert (result.metadata.in_tokens, result.metadata.out_tokens) == (500, 3)
    # billed at the registry price, while what OpenRouter says it billed is kept for reconciliation
    assert (
        result.metadata.cost is not None
        and result.metadata.cost.input == pytest.approx(500 * price / 1e6)
    )
    assert result.metadata.cost.output == 0.0
    assert result.metadata.extra["upstream_cost"] == 0.5
    assert result.metadata.extra["upstream_provider"] == upstream


async def test_noul_and_score_questions_and_a_json_state_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = {
        "refund": {"type": "noul", "instructions": "Promises a refund?"},
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["Minor", "Major"],
        },
    }
    answers = {
        "refund": {"type": "noul", "noul": 0.8},
        "severity": {"type": "score", "score": 0.7},
    }
    seen = serve(
        monkeypatch,
        lambda _: httpx.Response(200, json={**OPENROUTER_ANSWER, "answers": answers}),
    )

    result = await get_registry_model(JEV).query(
        decision_request({"message": "hi"}, questions)
    )

    sent = json.loads(seen[0].content)
    assert sent["questions"] == questions and sent["state"] == {"message": "hi"}
    assert parse_decision_output(result.output_text) == answers


async def test_openai_gets_its_own_request_shape_and_its_answers_come_back_in_the_shared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = {
        **QUESTIONS,
        "refund": {"type": "noul", "instructions": "Promises a refund?"},
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["Minor", "Major"],
        },
    }
    payload = {
        "model": "gpt-6-luna",
        "answers": [
            {
                "type": "choice",
                "name": "team",
                "choice": "engineering",
                "probabilities": [
                    {"value": "billing", "probability": 0.1},
                    {"value": "engineering", "probability": 0.9},
                ],
                "confidence": 0.8,
            },
            {"type": "predicate", "name": "refund", "probability": 0.3},
            {
                "type": "score",
                "name": "severity",
                "score": 0.75,
                "probabilities": [
                    {"value": 0, "label": "Minor", "probability": 0.25},
                    {"value": 1, "label": "Major", "probability": 0.75},
                ],
                "confidence": 0.5,
            },
        ],
        "usage": {"input_tokens": 400, "output_tokens": 0, "total_tokens": 400},
    }
    seen = serve(monkeypatch, lambda _: httpx.Response(200, json=payload))

    result = await get_registry_model("decision/gpt-6-luna").query(
        decision_request({"message": "hi"}, questions)
    )

    assert str(seen[0].url) == "https://api.openai.com/v1/decisions"
    assert seen[0].headers["authorization"] == "Bearer mock_ENV_OPENAI_API_KEY"
    assert json.loads(seen[0].content) == {
        "model": "gpt-6-luna",
        "input": '{"message": "hi"}',
        "questions": [
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team?",
                "choices": [
                    {"value": value, "description": text}
                    for value, text in TEAMS.items()
                ],
            },
            {
                "type": "predicate",
                "name": "refund",
                "instructions": "Promises a refund?",
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe?",
                "levels": [{"label": "Minor"}, {"label": "Major"}],
            },
        ],
    }
    assert parse_decision_output(result.output_text) == {
        "team": {
            "type": "choice",
            "choice": "engineering",
            "probabilities": {"billing": 0.1, "engineering": 0.9},
            "confidence": 0.8,
        },
        "refund": {"type": "noul", "noul": 0.3},
        "severity": {
            "type": "score",
            "score": 0.75,
            "legend": {"0": "Minor", "1": "Major"},
            "probabilities": {"0": 0.25, "1": 0.75},
            "confidence": 0.5,
        },
    }
    assert result.metadata.served_model == "gpt-6-luna"
    assert result.metadata.cost is not None
    assert result.metadata.cost.input == pytest.approx(400 * 0.1 / 1e6)
    assert result.metadata.extra == {}


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({"type": "choice", "instructions": "?", "criteria": ["a", "b"]}, "value: description"),
        ({"type": "score", "instructions": "?", "criteria": {"a": "b"}}, "list of levels"),
    ],
)
async def test_criteria_openai_cannot_take_are_rejected_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch, question: dict[str, object], message: str
) -> None:
    seen = serve(monkeypatch, ok)
    with pytest.raises(DecisionError, match=message):
        await call(
            get_registry_model("decision/gpt-6-luna"),
            decision_request(STATE, {"q": question}),
        )
    assert seen == []


# --- malformed responses --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}},  # no answers
        {
            "model": "m",
            "answers": {},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },  # question not answered
        {"model": "m", "answers": ANSWERS},  # no usage
        {
            "model": "m",
            "answers": ANSWERS,
            "usage": {"input_tokens": 5},
        },  # half the usage
        {
            "model": "m",
            "answers": ANSWERS,
            "usage": {"input_tokens": "5", "output_tokens": 1},
        },  # wrong type
        [],  # not an object
    ],
)
async def test_malformed_responses_raise_instead_of_reporting_zero_cost(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    serve(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(DecisionError) as raised:
        await call(get_registry_model(JEV))
    assert is_retriable_error(raised.value) is False


async def test_a_200_with_an_error_body_surfaces_the_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(
        monkeypatch,
        lambda _: httpx.Response(200, json={"error": {"message": "model overloaded"}}),
    )
    with pytest.raises(DecisionError, match="model overloaded"):
        await call(get_registry_model(JEV))


# --- upstream HTTP errors -------------------------------------------------------------------------------------------

UPSTREAM_BODY = {"error": {"message": "questions.team.criteria must have 2 to 20 options", "code": 422}}


def reply(status: int, body: object = None) -> Handler:
    return lambda _: httpx.Response(status, json=UPSTREAM_BODY if body is None else body)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the library retrier's waits instant, so that a retried call shows up as extra requests, not as a slow test."""
    monkeypatch.setattr("model_library.retriers.backoff.jitter", lambda _: 0.0)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
async def test_a_deterministic_upstream_4xx_is_not_retried_and_keeps_its_status(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None, status: int
) -> None:
    # A persistent 422 used to be retried for the library's whole backoff schedule (20 tries, about 51 minutes).
    seen = serve(monkeypatch, reply(status))

    with pytest.raises(NoRetryException) as raised:
        await get_registry_model(JEV).query(decision_request(STATE, QUESTIONS))

    assert len(seen) == 1
    assert is_retriable_error(raised.value) is False
    assert getattr(raised.value, "status_code") == status
    assert "questions.team.criteria" in str(raised.value)  # the upstream's reason is kept


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_the_gateway_error_envelope_is_the_one_the_sdk_error_gave(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # Clients classify by exception_type and status_code (claimbench's decision adapter and agent), so the type that
    # used to come through (UnprocessableEntityError, ...) must not change, only the retrying.
    handler = reply(status)
    seen = serve(monkeypatch, handler)
    sdk_error: Exception | None = None
    try:  # what the SDK raised before: the same request through the bare client
        await get_registry_model(JEV).get_client().post(
            "https://openrouter.ai/api/alpha/decisions", body={}, cast_to=object
        )
    except openai.APIStatusError as exc:
        sdk_error = exc
    assert sdk_error is not None and len(seen) == 1

    with pytest.raises(NoRetryException) as raised:
        await call(get_registry_model(JEV))

    envelope = provider_error_from_exception(raised.value, provider="decision")
    expected = provider_error_from_exception(sdk_error, provider="decision")
    assert envelope == expected
    assert envelope.status_code == status
    assert envelope.exception_type == type(sdk_error).__name__ != "DecisionError"


def test_an_override_of_another_setting_keeps_the_registry_decision_api() -> None:
    model = get_registry_model(
        "decision/gpt-6-luna",
        override_config=LLMConfig(provider_config=DecisionConfig(openrouter_cost_tier="low")),
    )
    assert isinstance(model, DecisionModel)
    assert model.provider_config.decision_api == "openai"
    assert model.provider_config.openrouter_cost_tier == "low"


async def test_an_openai_refusal_is_flagged_as_a_content_filter_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "model": "gpt-6-luna",
        "answers": [{"type": "refusal", "name": "team"}],
        "usage": {"input_tokens": 400, "output_tokens": 0, "total_tokens": 400},
    }
    serve(monkeypatch, lambda _: httpx.Response(200, json=payload))

    result = await get_registry_model("decision/gpt-6-luna").query(
        decision_request(STATE, QUESTIONS)
    )

    assert result.finish_reason.reason == FinishReason.CONTENT_FILTER
    assert parse_decision_output(result.output_text) == {"team": {"type": "refusal"}}


async def test_an_upstream_error_code_that_marks_a_refusal_is_still_recognised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model_library.exceptions import is_content_filter_error

    body = {"error": {"message": "blocked", "code": "content_filter"}}
    serve(monkeypatch, reply(400, body))
    with pytest.raises(NoRetryException) as raised:
        await call(get_registry_model(JEV))
    assert is_content_filter_error(raised.value) is True


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_a_4xx_whose_body_mentions_a_retriable_word_is_still_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None, status: int
) -> None:
    # The library's fallback matches substrings such as "500", "timeout" or "rate limit" in the message.
    body = {"error": {"message": "state is 500 characters over the limit; do not retry (timeout, rate limit)"}}
    seen = serve(monkeypatch, reply(status, body))

    with pytest.raises(NoRetryException):
        await get_registry_model(JEV).query(decision_request(STATE, QUESTIONS))

    assert len(seen) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_rate_limits_and_server_errors_are_still_retried(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None, status: int
) -> None:
    replies = [httpx.Response(status, json=UPSTREAM_BODY)] * 2 + [
        httpx.Response(200, json=OPENROUTER_ANSWER)
    ]
    seen = serve(monkeypatch, lambda _: replies[len(seen) - 1])

    result = await get_registry_model(JEV).query(decision_request(STATE, QUESTIONS))

    assert len(seen) == 3
    assert parse_decision_output(result.output_text)["team"]["choice"] == "engineering"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_rate_limits_and_server_errors_reach_the_retrier_as_the_sdk_error(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    serve(monkeypatch, reply(status))
    with pytest.raises(openai.APIStatusError) as raised:
        await call(get_registry_model(JEV))
    assert raised.value.status_code == status
    assert is_retriable_error(raised.value) is True


# --- bad requests are rejected before any HTTP call -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not valid JSON"),
        ('["a"]', "must be a JSON object"),
        (json.dumps({"questions": QUESTIONS}), "no 'state'"),
        (json.dumps({"state": "s"}), "non-empty 'questions'"),
        (json.dumps({"state": "s", "questions": {}}), "non-empty 'questions'"),
        (
            json.dumps({"state": "s", "questions": {"q": {"type": "essay"}}}),
            "needs a 'type'",
        ),
        (json.dumps({"state": "s", "questions": {"q": "choice"}}), "needs a 'type'"),
        (
            json.dumps({"state": "s", "questions": QUESTIONS, "model": "gpt-5"}),
            "only 'state' and 'questions'",
        ),
    ],
)
async def test_bad_decision_requests_are_rejected_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch, text: str, message: str
) -> None:
    seen = serve(monkeypatch, ok)
    with pytest.raises(DecisionError, match=message):
        await call(get_registry_model(JEV), text)
    assert seen == []


@pytest.mark.parametrize("kind", [["choice"], {"a": 1}, 5, None, True, ""])
async def test_a_non_string_question_type_is_a_request_error_not_a_type_error(
    monkeypatch: pytest.MonkeyPatch, kind: object
) -> None:
    serve(monkeypatch, ok)
    with pytest.raises(DecisionError, match="needs a 'type'"):
        await call(
            get_registry_model(JEV),
            json.dumps({"state": "s", "questions": {"q": {"type": kind}}}),
        )


@pytest.mark.parametrize(
    "state", ["x" * 469 + "\ud800", "\ud800", float("nan"), float("inf")]
)
async def test_unsendable_text_is_a_non_retriable_request_error_not_a_raw_exception(
    monkeypatch: pytest.MonkeyPatch, state: object
) -> None:
    # A lone surrogate at a position containing "500" used to escape from the HTTP client as a UnicodeEncodeError
    # whose message matched the retrier's "500" substring, so it was retried for about an hour.
    seen = serve(monkeypatch, ok)
    with pytest.raises(DecisionError, match="cannot be sent as JSON") as raised:
        await call(
            get_registry_model(JEV),
            json.dumps({"state": state, "questions": QUESTIONS}),
        )
    assert is_retriable_error(raised.value) is False
    assert seen == []


async def test_only_a_single_text_input_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(monkeypatch, ok)
    m = get_registry_model(JEV)
    text = decision_request(STATE, QUESTIONS)
    log = logging.getLogger("t")
    with pytest.raises(DecisionError, match="exactly one text input"):
        await m._query_impl(
            [SystemInput(text="be nice"), TextInput(text=text)],
            tools=[],
            query_logger=log,
        )
    with pytest.raises(DecisionError, match="exactly one text input"):
        await m._query_impl(
            [TextInput(text=text), TextInput(text=text)], tools=[], query_logger=log
        )
    assert seen == []


def test_a_custom_endpoint_is_rejected_instead_of_sending_its_key_to_openrouter() -> (
    None
):
    config = LLMConfig(
        custom_endpoint="https://custom.invalid/api/v1",
        custom_api_key=SecretStr("custom-key"),
    )
    with pytest.raises(DecisionError, match="custom_endpoint"):
        get_registry_model(JEV, override_config=config)


def test_direct_construction_has_the_decision_identity_and_leaves_the_config_alone() -> (
    None
):
    config = LLMConfig()

    first = DecisionModel("typesafe/jev-1.13-20260917", config=config)
    DecisionModel("typesafe/jev-1.13-20260917", config=config)

    assert first.provider == "decision"
    assert config.custom_endpoint is None


# --- registry -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "slug", "price", "upstream"), MODELS)
def test_every_model_has_the_release_date_its_pinned_slug_carries(
    key: str, slug: str, price: float, upstream: str
) -> None:
    # The benchmark exporter needs a release date for each model, and the pinned slug ends in it (YYYYMMDD).
    stamp = slug.rsplit("-", 1)[1]
    config = get_registry_config(key)
    assert config is not None and config.release_date is not None
    assert config.release_date.strftime("%Y%m%d") == stamp


async def test_cost_can_be_recomputed_from_the_registry() -> None:
    cost = await recompute_cost(JEV, {"in_tokens": 520, "out_tokens": 54})
    assert cost.input == pytest.approx(520 * 0.042 / 1_000_000)
    assert cost.output == 0.0 and cost.total == pytest.approx(520 * 0.042 / 1_000_000)
