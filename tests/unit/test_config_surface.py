from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
import tomllib

import pytest

from model_library.base import LLMConfig, ProviderConfig
from model_library.register_models import (
    ModelConfig,
    get_model_registry,
    get_provider_registry,
)
from model_library.registry_utils import create_config

ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_UNREGISTERED_PROVIDERS = {"cursor", "devin"}


@pytest.mark.unit
def test_package_discovery_pattern_includes_runtime_subpackages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include_patterns = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    package_names = {
        path.parent.relative_to(ROOT).as_posix().replace("/", ".")
        for path in ROOT.rglob("__init__.py")
    }

    for package in [
        "model_library.base.output",
        "model_library.providers.google",
        "model_gateway.usage_ledger.lambdas",
    ]:
        assert package in package_names
        assert any(fnmatchcase(package, pattern) for pattern in include_patterns)


def _active_registry_configs() -> list[ModelConfig]:
    return list(get_model_registry().values())


def _runtime_registry_configs() -> list[ModelConfig]:
    providers = get_provider_registry()
    active_configs = _active_registry_configs()
    missing_provider_names = sorted(
        {config.provider_name for config in active_configs}
        - set(providers)
        - _ALLOWED_UNREGISTERED_PROVIDERS
    )
    assert not missing_provider_names, (
        "Active model configs reference unregistered providers: "
        + ", ".join(missing_provider_names)
    )
    return [config for config in active_configs if config.provider_name in providers]


@pytest.mark.unit
def test_runtime_registry_configs_fail_when_active_provider_is_unregistered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_to_remove = next(
        config.provider_name
        for config in _active_registry_configs()
        if config.provider_name not in _ALLOWED_UNREGISTERED_PROVIDERS
    )
    providers = dict(get_provider_registry())
    providers.pop(provider_to_remove, None)
    monkeypatch.setattr(
        "tests.unit.test_config_surface.get_provider_registry", lambda: providers
    )

    with pytest.raises(AssertionError, match=provider_to_remove):
        _runtime_registry_configs()


@pytest.mark.unit
def test_active_registry_defaults_round_trip_to_llm_config() -> None:
    for registry_config in _runtime_registry_configs():
        llm_config = create_config(registry_config, override_config=None)
        defaults = registry_config.default_parameters.model_dump(
            exclude_unset=True, mode="json"
        )

        expected_max_tokens = defaults.get(
            "max_tokens", registry_config.properties.max_tokens
        )
        assert llm_config.max_tokens == expected_max_tokens
        assert llm_config.reasoning is registry_config.properties.reasoning_model
        assert llm_config.supports_images is registry_config.supports.images
        assert llm_config.supports_files is registry_config.supports.files
        assert llm_config.supports_videos is registry_config.supports.videos
        assert llm_config.supports_batch is registry_config.supports.batch
        assert llm_config.supports_temperature is registry_config.supports.temperature
        assert llm_config.supports_tools is registry_config.supports.tools
        assert (
            llm_config.supports_output_schema is registry_config.supports.output_schema
        )
        for field_name, expected_value in defaults.items():
            assert getattr(llm_config, field_name) == expected_value, (
                registry_config.full_key,
                field_name,
            )


@pytest.mark.unit
def test_active_registry_provider_properties_validate_to_provider_config() -> None:
    for registry_config in _runtime_registry_configs():
        provider_properties = registry_config.provider_properties.model_dump(
            exclude_none=True, exclude_unset=True, mode="json"
        )
        llm_config = create_config(registry_config, override_config=None)
        if provider_properties:
            assert isinstance(llm_config.provider_config, ProviderConfig), (
                registry_config.full_key,
                provider_properties,
            )
            actual_provider_config = llm_config.provider_config.model_dump(mode="json")
            for field_name, expected_value in provider_properties.items():
                assert actual_provider_config[field_name] == expected_value


@pytest.mark.unit
def test_override_config_only_replaces_explicit_fields() -> None:
    registry_config = get_model_registry()["openai/gpt-4o"]
    override = LLMConfig(temperature=0.25)

    llm_config = create_config(registry_config, override)

    assert llm_config.temperature == 0.25
    assert llm_config.max_tokens == registry_config.properties.max_tokens
    assert llm_config.supports_tools is registry_config.supports.tools
