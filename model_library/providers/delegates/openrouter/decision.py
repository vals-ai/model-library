"""Typed-decision models, served through OpenRouter's Decisions API (Jev, Mercury Decide, Microsoft-Decision-1) or
OpenAI's (``decision_api: openai``).

A typed-decision model takes a ``state`` and typed ``questions`` and returns a typed answer for each: ``choice`` (one
option plus a probability for every option), ``noul`` (a probability) or ``score`` (a value on an ordered scale). It
does not generate text.

This is an OpenRouter model whose input is a decision request and whose output is the answers. It reuses
``OpenRouterModel``'s client, key and retries and posts to the Decisions endpoint itself; the model sent is the
registry's ``provider_endpoint``. For OpenAI it points that client at OpenAI with ``OPENAI_API_KEY`` and translates the
request and answers, so every decision model takes and returns the same shapes.

The query's single text input *is* the decision request, as JSON::

    {"state": "...", "questions": {"verdict": {"type": "choice", "instructions": "...", "criteria": {"a": "...", ...}}}}

The result's ``output_text`` is the answers object as JSON (``{"answers": {...}}``). Use ``decision_request`` and
``parse_decision_output`` to build and read them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Sequence, cast

from openai import APIStatusError
from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
    TextInput,
    ToolDefinition,
)
from model_library.exceptions import NoRetryException, ProviderErrorPayload
from model_library.register_models import register_provider

from .openrouter import OpenRouterConfig, OpenRouterModel

# OpenRouter marks its Decisions API "alpha", so its path may change. Absolute, because the client's base URL is /v1.
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

OPENAI_URL = "https://api.openai.com/v1"

QUESTION_TYPES = ("choice", "noul", "score")


class DecisionConfig(OpenRouterConfig):
    decision_api: Literal["openrouter", "openai"] = "openrouter"


class DecisionError(NoRetryException):
    """A request or response that cannot succeed on retry. A bare ``ValueError`` would fall through to the library's
    message-substring matcher, which can mistake a digit in the text for an HTTP status."""


# Upstream 4xx statuses that can succeed on a retry, so they stay with the library's retrier: request timeout, conflict,
# too early and rate limit. Any other 4xx is a verdict on the request or the key (400 and 422 a request the API
# rejects, 401 and 403 a key it rejects, 402 no credits, 404 an unknown model) and repeating it cannot change it.
RETRIABLE_4XX = frozenset({408, 409, 425, 429})


class DecisionRequestRejected(NoRetryException):
    """The Decisions API rejected the request with a 4xx that retrying cannot change.

    The library classifies by exception type, and ``openai.UnprocessableEntityError`` (422) is one of the types it
    retries, for about 51 minutes. This carries the same message, status and error code, and reports the SDK error's class
    name as ``exception_type`` (also in the gateway's error envelope), so a client that classifies by status or by that
    name sees what it saw before."""

    def __init__(self, upstream: APIStatusError):
        super().__init__(str(upstream))
        self.status_code = upstream.status_code
        self.exception_type = type(upstream).__name__
        code, kind = getattr(upstream, "code", None), getattr(upstream, "type", None)
        self.code = code if isinstance(code, str) else None
        self.type = kind if isinstance(kind, str) else None

    @override
    def to_provider_error(self) -> ProviderErrorPayload:
        return {**super().to_provider_error(), "exception_type": self.exception_type}


def decision_request(state: object, questions: dict[str, dict[str, Any]]) -> str:
    """The text a caller sends as the query input."""
    return json.dumps({"state": state, "questions": questions})


def _obj(value: object) -> dict[str, Any] | None:
    """`value` as a JSON object, or None."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def parse_decision_output(output_text: str) -> dict[str, dict[str, Any]]:
    """The answers object from a result's ``output_text``, keyed by question name."""
    output = _obj(json.loads(output_text))
    answers = _obj(output.get("answers")) if output is not None else None
    if answers is None:
        raise DecisionError("decision output has no answers object")
    return cast(dict[str, dict[str, Any]], answers)


def _openai_question(name: str, question: dict[str, Any]) -> dict[str, Any]:
    """An OpenRouter question as an OpenAI one: ``noul`` is ``predicate``, choice ``criteria`` ({value: description})
    are ``choices`` and score ``criteria`` ([label, ...], lowest first) are ``levels``."""
    kind, criteria = question["type"], question.get("criteria")
    out = {
        "type": "predicate" if kind == "noul" else kind,
        "name": name,
        "instructions": question.get("instructions"),
    }
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise DecisionError(
                f"choice question {name!r} needs 'criteria' as {{value: description}}"
            )
        out["choices"] = [
            {"value": value, "description": text}
            for value, text in cast(dict[str, Any], criteria).items()
        ]
    elif kind == "score":
        if not isinstance(criteria, list):
            raise DecisionError(
                f"score question {name!r} needs 'criteria' as a list of levels, lowest first"
            )
        out["levels"] = [{"label": label} for label in cast(list[Any], criteria)]
    return out


def _openrouter_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI answer in OpenRouter's shape: probabilities keyed by value, a score's labels as its ``legend``."""
    answer = {key: value for key, value in answer.items() if key != "name"}
    probabilities = cast(list[dict[str, Any]], answer.get("probabilities") or [])
    match answer.get("type"):
        case "predicate":
            return {"type": "noul", "noul": answer.get("probability")}
        case "choice":
            return {
                **answer,
                "probabilities": {p["value"]: p["probability"] for p in probabilities},
            }
        case "score":
            return {
                **answer,
                "legend": {str(p["value"]): p["label"] for p in probabilities},
                "probabilities": {
                    str(p["value"]): p["probability"] for p in probabilities
                },
            }
        case _:  # a refusal
            return answer


@register_provider("decision")
class DecisionModel(OpenRouterModel):
    provider_config = DecisionConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["decision"] = "decision",
        *,
        config: LLMConfig | None = None,
    ):
        # The Decisions URL is absolute, so an endpoint override would be ignored while its key was still sent.
        if config is not None and config.custom_endpoint is not None:
            raise DecisionError(
                "typed-decision models always use their Decisions API; custom_endpoint is not supported"
            )
        # The parent writes its endpoint into the config it is given.
        config = config.model_copy() if config else LLMConfig()
        if (
            isinstance(config.provider_config, DecisionConfig)
            and config.provider_config.decision_api == "openai"
        ):
            config.custom_endpoint = OPENAI_URL
            config.custom_api_key = config.custom_api_key or SecretStr(
                model_library_settings.OPENAI_API_KEY
            )
        super().__init__(model_name, provider, config=config)

    @override
    async def parse_input(
        self, input: Sequence[InputItem], **kwargs: Any
    ) -> dict[str, Any]:
        """Turn the query input into ``{"state", "questions"}``, validating the shape."""
        if len(input) != 1 or not isinstance(input[0], TextInput):
            raise DecisionError(
                "a typed-decision query takes exactly one text input holding the decision request as JSON"
            )
        try:
            request = _obj(json.loads(input[0].text))
        except json.JSONDecodeError as exc:
            raise DecisionError(
                f"the decision request is not valid JSON: {exc}"
            ) from exc
        if request is None:
            raise DecisionError("the decision request must be a JSON object")
        extra = request.keys() - {"state", "questions"}
        if extra:
            raise DecisionError(
                f"unexpected decision request fields {sorted(extra)}; "
                "only 'state' and 'questions' are allowed (the model comes from the registry)"
            )
        if "state" not in request:
            raise DecisionError("the decision request has no 'state'")
        questions = _obj(request.get("questions"))
        if not questions:
            raise DecisionError(
                "the decision request needs a non-empty 'questions' object"
            )
        for name, question in questions.items():
            question = _obj(question)
            if question is None or question.get("type") not in QUESTION_TYPES:
                raise DecisionError(
                    f"question {name!r} needs a 'type' of {list(QUESTION_TYPES)}"
                )
        try:
            # A lone surrogate or NaN would otherwise escape from the HTTP client as a raw exception.
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except ValueError as exc:
            raise DecisionError(
                f"the decision request cannot be sent as JSON: {exc}"
            ) from exc
        return request

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        if tools or output_schema is not None or kwargs:
            raise DecisionError(
                "typed-decision models take only a decision request: no tools, output schema or query options"
            )
        return {"model": self.model_name, **await self.parse_input(input)}

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )
        via_openai = self.provider_config.decision_api == "openai"
        url, sent = DECISIONS_URL, body
        if via_openai:
            state = body["state"]
            url, sent = (
                "/decisions",
                {
                    "model": body["model"],
                    "input": state if isinstance(state, str) else json.dumps(state),
                    "questions": [
                        _openai_question(name, question)
                        for name, question in body["questions"].items()
                    ],
                },
            )
        try:
            payload = _obj(await self.get_client().post(url, body=sent, cast_to=object))
        except APIStatusError as exc:
            if 400 <= exc.status_code < 500 and exc.status_code not in RETRIABLE_4XX:
                raise DecisionRequestRejected(exc) from exc
            raise  # 429, 5xx and the other statuses keep the library's retry rules
        if payload is None:
            raise DecisionError("the Decisions API returned a non-object response")

        answers = payload.get("answers")
        if via_openai and isinstance(answers, list):
            answers = {
                answer.get("name"): _openrouter_answer(answer)
                for answer in cast(list[dict[str, Any]], answers)
            }
        answers = _obj(answers)
        if answers is None:
            detail = payload.get("error")
            raise DecisionError(
                f"the Decisions API returned no answers object{f' (error: {str(detail)[:300]})' if detail else ''}"
            )
        missing = [
            name for name in body["questions"] if _obj(answers.get(name)) is None
        ]
        if missing:
            raise DecisionError(
                f"the Decisions API returned no answer for questions {missing}"
            )

        # Answers still come back for the other questions, so the refusal is flagged rather than raised.
        refused = any(answer.get("type") == "refusal" for answer in answers.values())

        usage = _obj(payload.get("usage")) or {}
        in_tokens, out_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if not isinstance(in_tokens, int) or not isinstance(out_tokens, int):
            # A missing count would silently price the call at $0.
            raise DecisionError(
                "the Decisions API returned no token usage; refusing to report a $0 cost"
            )

        # What OpenRouter says it billed and which provider served the call, kept for reconciliation (OpenAI reports
        # neither). The result is billed at the registry price.
        extra = {
            f"upstream_{key}": value
            for key, value in (
                ("cost", usage.get("cost")),
                ("provider", payload.get("provider")),
            )
            if value is not None
        }
        served = payload.get("model")
        return QueryResult(
            output_text=json.dumps({"answers": answers}, separators=(",", ":")),
            finish_reason=FinishReasonInfo(
                reason=FinishReason.CONTENT_FILTER, raw="refusal"
            )
            if refused
            else FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
            metadata=QueryResultMetadata(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                served_model=served if isinstance(served, str) else None,
                extra=extra,
            ),
        )
