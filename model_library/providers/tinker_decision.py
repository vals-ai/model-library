"""Typed-decision models served by Tinker with Tinker Cookbook's decisions recipe (the ``tinker_decision`` provider).

A query takes the same decision request as the ``decision`` provider (``decision_request``) and returns the same answers
(``parse_decision_output``), so a benchmark's decision adapter works with either.

The recipe asks the model to pick one of the question's choices, renders that with ``DecisionRenderer`` and the base
model's recommended chat renderer, and reads one ``max_tokens=1`` sample's log-probabilities of the choice tokens at the
answer position (``sample_decision``). The probabilities are the softmax over those tokens; there is no classifier head.
A checkpoint trained with the recipe is scored exactly as it was trained.

The registry entry's ``provider_endpoint`` is a ``tinker://.../sampler_weights/...`` path or a Tinker base model ID;
``base_model`` names the checkpoint's base model (its tokenizer and renderer) and ``renderer`` overrides the recommended
renderer. The ``tinker`` and ``tinker-cookbook`` packages are the ``tinker`` extra.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

from pydantic import BaseModel
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
    InputItem,
    LLMConfig,
    ProviderConfig,
    QueryResult,
    QueryResultMetadata,
    ToolDefinition,
)
from model_library.providers.delegates.openrouter.decision import (
    DecisionError,
    decision_result,
    parse_decision_request,
    reject_query_options,
    state_text,
)
from model_library.register_models import register_provider

if TYPE_CHECKING:
    from tinker import SamplingClient
    from tinker_cookbook.recipes.decisions import DecisionRenderer

# A noul question is a two-way decision; its answer is the first choice's probability.
NOUL_CHOICES = (("yes", "True"), ("no", "False"))


def _decision_text(state: object, question: dict[str, Any]) -> str:
    text = state_text(state)
    instructions = question.get("instructions")
    return f"{text}\n\n{instructions}" if instructions else text


def _choices(name: str, question: dict[str, Any]) -> list[tuple[str, str]]:
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        return list(NOUL_CHOICES)
    if kind == "choice" and isinstance(criteria, dict):
        options = cast(dict[str, Any], criteria)
        if len(options) >= 2:
            return [(str(value), str(text or "")) for value, text in options.items()]
    raise DecisionError(
        f"question {name!r}: Tinker decisions take choice questions with at least two "
        "'criteria' ({value: description}) or noul questions; score questions are not supported"
    )


class TinkerDecider:
    """One sampling client and renderer per registry model, created on first use."""

    def __init__(
        self,
        endpoint: str,
        *,
        base_model: str | None,
        renderer_name: str | None,
        api_key: str | None,
    ):
        self.endpoint = endpoint
        self.base_model = base_model or (
            None if endpoint.startswith("tinker://") else endpoint
        )
        if self.base_model is None:
            raise DecisionError(
                "a tinker:// checkpoint needs base_model in its provider_properties"
            )
        self.renderer_name = renderer_name
        self.api_key = api_key
        self._ready: tuple[SamplingClient, DecisionRenderer] | None = None
        self._lock = asyncio.Lock()

    async def _clients(self) -> tuple[SamplingClient, DecisionRenderer]:
        async with self._lock:
            if self._ready is None:
                try:
                    import tinker
                    from tinker_cookbook import model_info, renderers
                    from tinker_cookbook.recipes.decisions import DecisionRenderer
                    from tinker_cookbook.tokenizer_utils import get_tokenizer
                except ImportError as exc:
                    raise DecisionError(
                        "Tinker decisions need the 'tinker' extra (tinker, tinker-cookbook)"
                    ) from exc
                base = cast(str, self.base_model)
                service = tinker.ServiceClient(api_key=self.api_key)
                if self.endpoint.startswith("tinker://"):
                    sampling = await service.create_sampling_client_async(
                        model_path=self.endpoint
                    )
                else:
                    sampling = await service.create_sampling_client_async(
                        base_model=self.endpoint
                    )
                renderer = renderers.get_renderer(
                    self.renderer_name
                    or model_info.get_recommended_renderer_name(base),
                    get_tokenizer(base),
                )
                self._ready = (sampling, DecisionRenderer(renderer))
            return self._ready

    async def answer(
        self, state: object, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        """The answers in the shared shape, the prompt tokens and the sampled tokens."""
        choices = {name: _choices(name, q) for name, q in questions.items()}
        sampling, decision_renderer = await self._clients()
        from tinker_cookbook.exceptions import DataValidationError, RendererError
        from tinker_cookbook.recipes.decisions import Decision, sample_decision

        try:
            renderings = {
                name: decision_renderer.render(
                    Decision(_decision_text(state, q), choices[name])
                )
                for name, q in questions.items()
            }
        except (DataValidationError, RendererError) as exc:
            raise DecisionError(str(exc)) from exc
        results = await asyncio.gather(
            *(sample_decision(sampling, r) for r in renderings.values())
        )
        answers: dict[str, dict[str, Any]] = {}
        for name, probabilities in zip(renderings, results, strict=True):
            if questions[name]["type"] == "noul":
                answers[name] = {"type": "noul", "noul": probabilities["yes"]}
            else:
                answers[name] = {
                    "type": "choice",
                    "choice": max(probabilities, key=probabilities.__getitem__),
                    "probabilities": probabilities,
                }
        in_tokens = sum(len(r.tokens) for r in renderings.values())
        return answers, in_tokens, len(renderings)


class TinkerDecisionConfig(ProviderConfig):
    base_model: str | None = None
    renderer: str | None = None
    # A tinker://.../sampler_weights/... path on this entry's base model, sampled instead of provider_endpoint; set per
    # request, so any checkpoint runs under its base model's entry without one of its own.
    checkpoint: str | None = None


@register_provider("tinker_decision")
class TinkerDecisionModel(LLM):
    api_key_setting = "TINKER_API_KEY"

    provider_config = TinkerDecisionConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["tinker_decision"] = "tinker_decision",
        *,
        config: LLMConfig | None = None,
    ):
        if config is not None and config.custom_endpoint is not None:
            raise DecisionError("Tinker decision models do not take a custom_endpoint")
        super().__init__(model_name, provider, config=config)
        checkpoint = self.provider_config.checkpoint
        if checkpoint is not None and not checkpoint.startswith("tinker://"):
            raise DecisionError(
                f"checkpoint must be a tinker://.../sampler_weights/... path, not {checkpoint!r}"
            )
        self.decider = TinkerDecider(
            checkpoint or model_name,
            base_model=self.provider_config.base_model,
            renderer_name=self.provider_config.renderer,
            api_key=config.custom_api_key.get_secret_value()
            if config and config.custom_api_key
            else model_library_settings.TINKER_API_KEY,
        )

    def _get_default_api_key(self) -> str:
        return self._api_key_from(model_library_settings)

    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        """The Tinker clients belong to ``decider``, created on its first query."""

    @override
    async def parse_image(self, image: FileInput) -> Any:
        raise DecisionError("typed-decision models take no images")

    @override
    async def parse_file(self, file: FileInput) -> Any:
        raise DecisionError("typed-decision models take no files")

    @override
    async def parse_tools(self, tools: list[ToolDefinition]) -> Any:
        raise DecisionError("typed-decision models take no tools")

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise DecisionError("typed-decision models take no files")

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
        reject_query_options(tools, output_schema, kwargs)
        return await self.parse_input(input)

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
        answers, in_tokens, out_tokens = await self.decider.answer(
            body["state"], body["questions"]
        )
        return decision_result(
            answers, QueryResultMetadata(in_tokens=in_tokens, out_tokens=out_tokens)
        )
