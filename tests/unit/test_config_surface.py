from __future__ import annotations

from fnmatch import fnmatchcase
import json
from pathlib import Path
import tomllib  # pyright: ignore[reportMissingImports]

import pytest
from pydantic import ValidationError

from model_library.base import LLMConfig, ProviderConfig
from model_library.register_models import (
    DefaultParameters,
    ModelConfig,
    RegistryEntry,
    TranscriptionModelConfig,
    model_config_from_json,
    registry_entry_adapter,
    get_model_registry,
    get_provider_registry,
    get_transcription_registry,
    parse_yaml_blocks,
)
from model_library.registry_utils import (
    create_config,
    get_model_input_context_window,
)

ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_UNREGISTERED_PROVIDERS = {"cursor", "devin", "factory"}


def _registry_entry(key: str) -> RegistryEntry:
    return get_model_registry().get(key) or get_transcription_registry()[key]


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


def test_package_data_includes_deprecated_registry_configs() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["model_library"]

    assert "config/deprecated/*.yaml" in package_data


def _active_registry_configs() -> list[RegistryEntry]:
    return list(get_model_registry().values())


def _runtime_registry_configs() -> list[RegistryEntry]:
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
            "max_tokens",
            registry_config.properties.max_tokens
            if registry_config.properties is not None
            else LLMConfig.model_fields["max_tokens"].default,
        )
        expected_reasoning = (
            registry_config.properties.reasoning_model
            if registry_config.properties is not None
            else LLMConfig.model_fields["reasoning"].default
        )
        assert llm_config.max_tokens == expected_max_tokens
        assert llm_config.reasoning is expected_reasoning
        assert llm_config.supports_images is registry_config.supports.images
        assert llm_config.supports_files is registry_config.supports.files
        assert llm_config.supports_audio is registry_config.supports.audio
        assert (
            llm_config.supports_transcription is registry_config.supports.transcription
        )
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
def test_transcription_registry_omits_llm_properties() -> None:
    transcription_models = list(get_transcription_registry().values())
    assert transcription_models

    for registry_config in transcription_models:
        assert registry_config.properties is None
        assert registry_config.transcription_language is not None
        assert registry_config.transcription_streaming is not None
        if (
            registry_config.supports.batch is False
            and registry_config.costs_per_million_token is not None
        ):
            assert registry_config.costs_per_million_token.batch is None

        llm_config = create_config(registry_config, override_config=None)
        assert llm_config.max_tokens == LLMConfig.model_fields["max_tokens"].default
        assert llm_config.reasoning is LLMConfig.model_fields["reasoning"].default
        assert llm_config.supports_transcription is True
        with pytest.raises(Exception, match="not found in registry"):
            get_model_input_context_window(registry_config.full_key)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name", ["transcription_language", "transcription_streaming"]
)
def test_transcription_registry_requires_request_metadata(field_name: str) -> None:
    config = get_transcription_registry()["openai/gpt-4o-transcribe"]
    payload = config.model_dump()
    payload[field_name] = None

    with pytest.raises(ValueError, match=field_name):
        registry_entry_adapter.validate_python(payload)


@pytest.mark.parametrize("omit_properties", [True, False])
@pytest.mark.parametrize("transcription", [True, False])
def test_registry_requires_properties_only_for_chat(
    omit_properties: bool, transcription: bool
) -> None:
    key = "openai/gpt-4o-transcribe" if transcription else "openai/gpt-4o"
    payload = _registry_entry(key).model_dump()
    if omit_properties:
        payload.pop("properties")
    else:
        payload["properties"] = None

    if transcription:
        config = registry_entry_adapter.validate_python(payload)
        assert config.properties is None
        assert create_config(config, None).max_tokens == LLMConfig().max_tokens
    else:
        with pytest.raises(ValidationError, match="properties"):
            registry_entry_adapter.validate_python(payload)


@pytest.mark.parametrize("field_name", ["context_window", "max_tokens", "reasoning_model"])
@pytest.mark.parametrize("transcription", [True, False])
def test_registry_rejects_incomplete_properties(
    field_name: str, transcription: bool
) -> None:
    key = "openai/gpt-4o-transcribe" if transcription else "openai/gpt-4o"
    payload = _registry_entry(key).model_dump()
    properties = {"context_window": 128_000, "max_tokens": 16_000, "reasoning_model": False}
    properties.pop(field_name)
    payload["properties"] = properties

    with pytest.raises(ValidationError, match=field_name):
        registry_entry_adapter.validate_python(payload)


def test_transcription_alternative_key_preserves_absent_properties() -> None:
    payload = get_transcription_registry()["openai/gpt-4o-transcribe"].model_dump(
        exclude={"provider_name", "full_key", "slug"}
    )
    payload["alternative_keys"] = [
        "openai/transcription-alias",
        {"openai/transcription-override": {"label": "Transcription alias"}},
    ]
    registry: dict[str, RegistryEntry] = {}

    parse_yaml_blocks({"models": {"openai/transcription-source": payload}}, registry)

    assert len(registry) == 3
    for config in registry.values():
        assert config.properties is None
        assert create_config(config, None).supports_transcription


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
    assert registry_config.properties is not None
    assert llm_config.max_tokens == registry_config.properties.max_tokens
    assert llm_config.supports_tools is registry_config.supports.tools


@pytest.mark.parametrize(
    "key", ["anthropic/claude-fable-5", "openai/gpt-4o-transcribe"]
)
def test_registry_round_trips_models_and_json(key: str) -> None:
    """Gateway round trips preserve provider settings and explicit defaults."""
    config = _registry_entry(key).model_copy(deep=True)
    config.default_parameters = DefaultParameters(temperature=0.25, max_tokens=None)
    expected_type = TranscriptionModelConfig if config.properties is None else ModelConfig
    assert type(registry_entry_adapter.validate_python(config)) is expected_type
    payload = config.model_dump(mode="json")
    expected_properties = config.provider_properties.model_dump(mode="json")
    if key.startswith("anthropic/"):
        assert expected_properties["supports_auto_thinking"] is True
        assert expected_properties["fallback_models"] == ["claude-opus-4-8"]
    assert payload["provider_properties"] == expected_properties
    assert payload["default_parameters"] == {"temperature": 0.25, "max_tokens": None}
    assert "reasoning" not in payload["default_parameters"]
    for gateway_payload in [payload, {**payload, "future_registry_field": True}]:
        reparsed = model_config_from_json(gateway_payload)
        assert type(reparsed) is expected_type
        assert reparsed.model_dump(mode="json") == payload
        assert create_config(
            reparsed, None, resolve_provider_config=False
        ) == create_config(config, None, resolve_provider_config=False)
    python_payload = config.model_dump()
    python_payload["supports"] = config.supports
    assert type(registry_entry_adapter.validate_python(python_payload)) is expected_type


def test_hybrid_registry_payload_selects_chat_schema() -> None:
    """Hybrid payloads retain chat properties through both registry parsers."""
    payload = get_transcription_registry()["openai/gpt-4o-transcribe"].model_dump(mode="json")
    payload["properties"] = {
        "context_window": 128_000,
        "max_tokens": 16_000,
        "reasoning_model": False,
    }
    raw_payload = get_transcription_registry()["openai/gpt-4o-transcribe"].model_dump(
        exclude={"provider_name", "full_key", "slug"}
    )
    raw_payload["properties"] = payload["properties"]
    registry: dict[str, RegistryEntry] = {}
    parse_yaml_blocks({"models": {"openai/hybrid": raw_payload}}, registry)
    for config in [
        registry["openai/hybrid"],
        registry_entry_adapter.validate_json(json.dumps(payload)),
        model_config_from_json(payload),
    ]:
        assert isinstance(config, ModelConfig)
        assert config.supports.transcription is True
        assert config.properties.max_tokens == 16_000
        assert create_config(
            config, None, resolve_provider_config=False
        ).supports_transcription is True


def test_chat_schema_requires_properties_even_with_transcription_support() -> None:
    payload = get_transcription_registry()["openai/gpt-4o-transcribe"].model_dump()
    with pytest.raises(ValidationError, match="properties"):
        ModelConfig.model_validate(payload)


def test_transcription_schema_rejects_chat_properties() -> None:
    payload = get_model_registry()["openai/gpt-4o"].model_dump()
    with pytest.raises(ValidationError, match="properties"):
        TranscriptionModelConfig.model_validate(payload)


def test_transcription_schema_requires_transcription_capability() -> None:
    payload = get_transcription_registry()["openai/gpt-4o-transcribe"].model_dump()
    payload["supports"] = {}
    with pytest.raises(ValidationError, match="supports.transcription"):
        TranscriptionModelConfig.model_validate(payload)
    with pytest.raises(ValidationError, match="properties"):
        registry_entry_adapter.validate_python(payload)
    payload.pop("supports")
    with pytest.raises(ValidationError, match="supports"):
        registry_entry_adapter.validate_python(payload)


@pytest.mark.parametrize("key", ["deepgram/flux-general-en", "openai/gpt-4o"])
def test_only_chat_requires_explicit_token_pricing(key: str) -> None:
    payload = _registry_entry(key).model_dump()
    payload.pop("costs_per_million_token")
    if key == "openai/gpt-4o":
        with pytest.raises(ValidationError, match="costs_per_million_token"):
            registry_entry_adapter.validate_python(payload)
    else:
        config = registry_entry_adapter.validate_python(payload)
        assert isinstance(config, TranscriptionModelConfig)
        assert config.costs_per_million_token is None
        assert config.transcription_cost is not None
        assert create_config(config, None).supports_transcription


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_name", "override"),
        ("full_key", "override"),
        ("slug", "override"),
        ("provider_endpoint", False),
        ("provider_endpoint", 0),
    ],
)
def test_yaml_rejects_generated_metadata_and_invalid_endpoint(
    field: str, value: object
) -> None:
    """YAML inputs cannot override generated metadata or bypass endpoint typing."""
    payload = get_model_registry()["openai/gpt-4o"].model_dump(
        exclude={"provider_name", "full_key", "slug"}
    )
    payload[field] = value
    with pytest.raises(ValueError):
        parse_yaml_blocks({"models": {"openai/test": payload}}, {})
