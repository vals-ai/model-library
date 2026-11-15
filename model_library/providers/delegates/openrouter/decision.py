"""Typed-decision models (the ``decision`` provider), served by the Decisions API their ``decision_api`` names.

A typed-decision model takes a ``state`` and typed ``questions`` and returns a typed answer for each: ``choice`` (one
option plus a probability for every option), ``noul`` (a probability) or ``score`` (a value on an ordered scale). It
does not generate text.

The query's single text input *is* the decision request, as JSON::

    {"state": "...", "questions": {"verdict": {"type": "choice", "instructions": "...", "criteria": {"a": "...", ...}}}}

The result's ``output_text`` is the answers object as JSON (``{"answers": {...}}``). Use ``decision_request`` and
``parse_decision_output`` to build and read them. Every API takes and returns OpenRouter's shapes; ``DECISION_APIS``
holds how each one differs. ``DecisionModel`` reuses ``OpenRouterModel``'s client, key and retries, points the client
at its API and posts to ``/decisions`` itself; the model sent is the registry's ``provider_endpoint``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

from openai import APIStatusError, Omit
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

QUESTION_TYPES = ("choice", "noul", "score")


@dataclass(frozen=True)
class DecisionAPI:
    """Where one Decisions API serves ``/decisions``, the setting holding its key, and how it differs from OpenRouter's."""

    base_url: str
    key_setting: str
    # Sends ``state`` as ``input``, as JSON text when it is not a string.
    state_as_input: bool = False
    # Takes and returns OpenAI's question and answer shapes (``_openai_question``, ``_openrouter_answer``).
    openai_shapes: bool = False
    # Sends the key in this header instead of as a bearer token.
    key_header: str | None = None
    # Prefix of the usage token counts.
    usage_prefix: str = ""


DECISION_APIS = {
    # OpenRouter marks its Decisions API "alpha", so its path may change.
    "openrouter": DecisionAPI("https://openrouter.ai/api/alpha", "OPENROUTER_API_KEY"),
    "openai": DecisionAPI(
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        state_as_input=True,
        openai_shapes=True,
    ),
    "perplexity": DecisionAPI("https://api.perplexity.ai/v1", "PERPLEXITY_API_KEY"),
    # Google rejects an API key sent as a bearer token.
    "google": DecisionAPI(
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
        state_as_input=True,
        key_header="x-goog-api-key",
        usage_prefix="total_",
    ),
}


class DecisionConfig(OpenRouterConfig):
    decision_api: Literal["openrouter", "openai", "perplexity", "google"] = "openrouter"


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
        # The SDK copies these from the error body, where an API may put a number.
        self.code = upstream.code if isinstance(upstream.code, str) else None
        self.type = upstream.type if isinstance(upstream.type, str) else None

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


def reject_query_options(
    tools: list[ToolDefinition], output_schema: object, kwargs: dict[str, object]
) -> None:
    if tools or output_schema is not None or kwargs:
        raise DecisionError(
            "typed-decision models take only a decision request: no tools, output schema or query options"
        )


def parse_decision_request(input: Sequence[InputItem]) -> dict[str, Any]:
    """``{"state", "questions"}`` from the query's single text input, validated. Shared by every typed-decision
    provider."""
    if len(input) != 1 or not isinstance(input[0], TextInput):
        raise DecisionError(
            "a typed-decision query takes exactly one text input holding the decision request as JSON"
        )
    try:
        request = _obj(json.loads(input[0].text))
    except json.JSONDecodeError as exc:
        raise DecisionError(f"the decision request is not valid JSON: {exc}") from exc
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
        raise DecisionError("the decision request needs a non-empty 'questions' object")
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


def state_text(state: object) -> str:
    """``state`` as text: itself when it is a string, else its JSON."""
    return state if isinstance(state, str) else json.dumps(state)


def decision_result(
    answers: dict[str, dict[str, Any]], metadata: QueryResultMetadata
) -> QueryResult:
    """The result every typed-decision provider returns. A refusal is flagged, not raised, since the other questions
    are still answered."""
    refused = any(answer.get("type") == "refusal" for answer in answers.values())
    return QueryResult(
        output_text=json.dumps({"answers": answers}, separators=(",", ":")),
        finish_reason=FinishReasonInfo(
            reason=FinishReason.CONTENT_FILTER, raw="refusal"
        )
        if refused
        else FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
        metadata=metadata,
    )


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
    # The key follows `decision_api`, not one setting.
    api_key_setting = None

    provider_config = DecisionConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["decision"] = "decision",
        *,
        config: LLMConfig | None = None,
    ):
        # The client always points at the Decisions API, so an endpoint override would be ignored while its key was sent.
        if config is not None and config.custom_endpoint is not None:
            raise DecisionError(
                "typed-decision models always use their Decisions API; custom_endpoint is not supported"
            )
        # The parent writes its endpoint into the config it is given.
        config = config.model_copy() if config else LLMConfig()
        provider_config = (
            config.provider_config
            if isinstance(config.provider_config, DecisionConfig)
            else self.provider_config
        )
        self.api = DECISION_APIS[provider_config.decision_api]
        config.custom_endpoint = self.api.base_url
        config.custom_api_key = config.custom_api_key or SecretStr(
            getattr(model_library_settings, self.api.key_setting)
        )
        super().__init__(model_name, provider, config=config)

    @override
    async def parse_input(
        self, input: Sequence[InputItem], **kwargs: object
    ) -> dict[str, Any]:
        return parse_decision_request(input)

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """The request in OpenRouter's shape: the model, the decision request and any pinned hosts."""
        reject_query_options(tools, output_schema, kwargs)
        body = {"model": self.model_name, **await self.parse_input(input)}
        if self.provider_config.openrouter_provider_only:
            body["provider"] = {"only": self.provider_config.openrouter_provider_only}
        return body

    def _wire_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """``body`` in this API's request shape."""
        if not self.api.state_as_input:
            return body
        questions = body["questions"]
        return {
            "model": body["model"],
            "input": state_text(body["state"]),
            "questions": [_openai_question(name, q) for name, q in questions.items()]
            if self.api.openai_shapes
            else questions,
        }

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
        client = self.get_client()
        headers: dict[str, str | Omit] = (
            {"Authorization": Omit(), self.api.key_header: client.api_key}
            if self.api.key_header
            else {}
        )
        try:
            payload = _obj(
                await client.post(
                    "/decisions",
                    body=self._wire_body(body),
                    cast_to=object,
                    options={"headers": headers},
                )
            )
        except APIStatusError as exc:
            if 400 <= exc.status_code < 500 and exc.status_code not in RETRIABLE_4XX:
                raise DecisionRequestRejected(exc) from exc
            raise  # 429, 5xx and the other statuses keep the library's retry rules
        if payload is None:
            raise DecisionError("the Decisions API returned a non-object response")

        answers = payload.get("answers")
        if self.api.openai_shapes and isinstance(answers, list):
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

        usage = _obj(payload.get("usage")) or {}
        in_tokens = usage.get(f"{self.api.usage_prefix}input_tokens")
        out_tokens = usage.get(f"{self.api.usage_prefix}output_tokens")
        if not isinstance(in_tokens, int) or not isinstance(out_tokens, int):
            # A missing count would silently price the call at $0.
            raise DecisionError(
                "the Decisions API returned no token usage; refusing to report a $0 cost"
            )

        # OpenRouter's bill supplies effective_cost; cost remains the registry estimate.
        # Keep its serving provider for reconciliation; the other APIs report neither field.
        extra = {
            f"upstream_{key}": value
            for key, value in (
                ("cost", usage.get("cost")),
                ("provider", payload.get("provider")),
            )
            if value is not None
        }
        served = payload.get("model")
        return decision_result(
            cast(dict[str, dict[str, Any]], answers),
            QueryResultMetadata(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                served_model=served if isinstance(served, str) else None,
                extra=extra,
            ),
        )
