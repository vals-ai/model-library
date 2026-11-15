"""The `tinker_decision` provider: typed decisions answered with Tinker Cookbook's decisions recipe."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from model_library.base import LLMConfig
from model_library.providers.delegates.openrouter.decision import decision_request, parse_decision_output
from model_library.providers.tinker_decision import (
    TinkerDecider,
    TinkerDecisionConfig,
    DecisionError,
    TinkerDecisionModel,
)
from model_library.registry_utils import get_registry_model

decisions = pytest.importorskip("tinker_cookbook.recipes.decisions")

CHECKPOINT = "tinker://run-1:train:0/sampler_weights/step-10"
STATE = "USER: Can SNAP buy a hot rotisserie chicken?\nASSISTANT: No, not hot food."
QUESTIONS: dict[str, dict[str, Any]] = {
    "verdict": {
        "type": "choice",
        "instructions": "Criterion: says SNAP cannot buy hot food.",
        "criteria": {"pass": "Satisfied", "fail": "Not satisfied"},
    },
    "mentions_rmp": {"type": "noul", "instructions": "Mentions the Restaurant Meals Program?"},
}


def tinker_model(**config: Any) -> TinkerDecisionModel:
    return TinkerDecisionModel(
        CHECKPOINT,
        config=LLMConfig(provider_config=TinkerDecisionConfig(base_model="Qwen/Qwen3-8B", **config)),
    )


def fake_recipe(
    monkeypatch: pytest.MonkeyPatch, model: TinkerDecisionModel, probabilities: dict[str, dict[str, float]]
) -> list[Any]:
    """Stands in for the Tinker sampler and tokenizer; records each rendered Decision."""
    rendered: list[Any] = []

    def render(decision: Any) -> Any:
        rendered.append(decision)
        return SimpleNamespace(tokens=[0] * 100, decision=decision)

    async def sample_decision(_sampling: Any, rendering: Any) -> dict[str, float]:
        labels = [label for label, _ in rendering.decision.choices]
        return probabilities["noul" if labels == ["yes", "no"] else "choice"]

    monkeypatch.setattr(decisions, "sample_decision", sample_decision)
    model.decider._ready = (object(), SimpleNamespace(render=render))  # pyright: ignore[reportAttributeAccessIssue]
    return rendered


async def test_answers_carry_the_recipes_choice_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    model = tinker_model()
    rendered = fake_recipe(
        monkeypatch, model, {"choice": {"pass": 0.3, "fail": 0.7}, "noul": {"yes": 0.9, "no": 0.1}}
    )

    result = await model.query(decision_request(STATE, QUESTIONS))

    assert parse_decision_output(result.output_text) == {
        "verdict": {"type": "choice", "choice": "fail", "probabilities": {"pass": 0.3, "fail": 0.7}},
        "mentions_rmp": {"type": "noul", "noul": 0.9},
    }
    assert result.metadata.in_tokens == 200 and result.metadata.out_tokens == 2
    verdict = rendered[0]
    assert verdict.text == f"{STATE}\n\nCriterion: says SNAP cannot buy hot food."
    assert verdict.choices == [("pass", "Satisfied"), ("fail", "Not satisfied")]


async def test_a_structured_state_is_sent_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    model = tinker_model()
    rendered = fake_recipe(monkeypatch, model, {"choice": {"pass": 1.0, "fail": 0.0}})

    await model.query(decision_request({"turns": 2}, {"verdict": QUESTIONS["verdict"]}))

    assert rendered[0].text.startswith('{"turns": 2}\n\n')


async def test_a_query_without_the_tinker_extra_raises_a_decision_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "tinker" or m.startswith(("tinker.", "tinker_cookbook"))]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(DecisionError, match="'tinker' extra"):
        await tinker_model().query(decision_request(STATE, QUESTIONS))


async def test_score_questions_are_rejected_before_any_tinker_call() -> None:
    model = tinker_model()
    score = {"rating": {"type": "score", "instructions": "1-5", "criteria": ["1", "2", "3"]}}
    with pytest.raises(DecisionError, match="score questions"):
        await model.query(decision_request(STATE, score))


@pytest.mark.parametrize(
    "text",
    ["not json", '{"state": "x"}', '{"state": "x", "questions": {}}', '{"state": "x", "questions": {"q": {}}, "model": "m"}'],
)
async def test_malformed_requests_are_rejected(text: str) -> None:
    with pytest.raises(DecisionError):
        await tinker_model().query(text)


def test_a_checkpoint_without_its_base_model_is_rejected() -> None:
    with pytest.raises(DecisionError, match="base_model"):
        TinkerDecider(CHECKPOINT, base_model=None, renderer_name=None, api_key=None)


def test_a_base_model_endpoint_is_its_own_base_model() -> None:
    decider = TinkerDecider("Qwen/Qwen3-8B", base_model=None, renderer_name=None, api_key=None)
    assert decider.base_model == "Qwen/Qwen3-8B"


def test_the_registry_entry_routes_to_tinker() -> None:
    model = get_registry_model("tinker_decision/qwen3-8b")
    assert isinstance(model, TinkerDecisionModel)
    assert model.decider.base_model == "Qwen/Qwen3-8B"


def test_a_request_checkpoint_runs_under_its_base_models_entry() -> None:
    model = get_registry_model(
        "tinker_decision/qwen3-8b",
        override_config=LLMConfig(provider_config=TinkerDecisionConfig(checkpoint=CHECKPOINT)),
    )
    assert isinstance(model, TinkerDecisionModel)
    assert (model.decider.endpoint, model.decider.base_model) == (CHECKPOINT, "Qwen/Qwen3-8B")


def test_a_checkpoint_that_is_not_a_tinker_path_is_rejected() -> None:
    with pytest.raises(DecisionError, match="tinker://"):
        tinker_model(checkpoint="Qwen/Qwen3-8B")
