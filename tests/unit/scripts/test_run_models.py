import pytest
from rich.console import Console

import model_gateway.model_helpers as model_helpers
from model_library.settings import ModelLibrarySettings
from model_library.registry_utils import CLI_ONLY_PROVIDERS
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
    assert config.supports_batch is False


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
