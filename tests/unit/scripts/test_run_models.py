import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

import model_gateway.model_helpers as model_helpers
from model_library.base import QueryResult, QueryResultMetadata, TextInput
from model_library.settings import ModelLibrarySettings
from model_library.registry_utils import CLI_ONLY_PROVIDERS, get_registry_model
from scripts import run_models


def test_smoke_model_config_uses_default_key_when_secondary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ModelLibrarySettings()
    settings.set(META_API_KEY="primary")
    monkeypatch.setattr(run_models, "model_library_settings", settings)
    monkeypatch.setattr(model_helpers, "model_library_settings", settings)

    config = run_models.smoke_model_config("meta/muse_spark_1_3_max")

    assert config.custom_api_key is None


def test_select_models_skips_cli_only_providers() -> None:
    runnable, skipped = run_models.select_models(
        run_models.model_registry, research=False
    )

    assert not [
        key
        for key in runnable
        if run_models.model_registry[key].provider_name in CLI_ONLY_PROVIDERS
    ]
    assert set(skipped) <= CLI_ONLY_PROVIDERS
    assert "devin/adaptive" in skipped["devin"]
    assert "factory/router" in skipped["factory"]


def test_dashboard_reports_skips_without_counting_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(run_models.skipped_model_names, "devin", ["devin/adaptive"])
    monkeypatch.setattr(run_models, "providers", {"devin"})

    console = Console(width=120)
    with console.capture() as capture:
        console.print(run_models.create_dashboard(total=0, completed_count=0))
    output = capture.get()

    assert "devin: 1 models, 0 failed, 1 skipped (cli-only)" in output


async def test_decision_smoke_prompt_is_a_decision_request() -> None:
    prompt = run_models.smoke_cases["decision"].prompt

    parsed = await get_registry_model("decision/jev-1.13.0").parse_input(
        [TextInput(text=prompt)]
    )

    assert set(parsed["questions"]) == {"team"}


def _smoke_answer(choice: str, probabilities: dict[str, float]) -> QueryResult:
    answers = {"team": {"type": "choice", "choice": choice, "probabilities": probabilities}}
    return QueryResult(
        output_text=json.dumps({"answers": answers}),
        metadata=QueryResultMetadata(in_tokens=120, out_tokens=0),
    )


async def _run_decision_smoke(monkeypatch: pytest.MonkeyPatch, result: QueryResult) -> None:
    model = SimpleNamespace(
        instance_logger=logging.getLogger("t"), query=AsyncMock(return_value=result)
    )
    monkeypatch.setattr(run_models, "get_registry_model", lambda *_a, **_k: model)
    run_models.failed_model_names.clear()
    run_models.completed_models.clear()
    run_models.exceptions.clear()

    await run_models.process_model("decision/jev-1.13.0", "decision")


async def test_decision_smoke_passes_a_typed_answer_with_no_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = {"billing": 0.01, "engineering": 0.97, "support": 0.02}

    await _run_decision_smoke(monkeypatch, _smoke_answer("engineering", probabilities))

    assert run_models.failed_model_names["decision"] == []
    assert run_models.completed_models["decision"] == 1


async def test_decision_smoke_fails_an_answer_outside_the_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_decision_smoke(monkeypatch, _smoke_answer("marketing", {"billing": 1.0}))

    assert run_models.failed_model_names["decision"] == ["decision/jev-1.13.0"]
    assert "not one of the options" in str(run_models.exceptions[-1][1])
