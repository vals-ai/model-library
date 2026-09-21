"""Unit tests for custom_register_models"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from model_library.custom_register_models import (
    DEFAULT_URL_TIMEOUT_SECONDS,
    load_custom_model_configs,
    load_latest_vals_model_configs,
)
from model_library.register_models import (
    get_model_registry,
    get_transcription_registry,
)
import model_library.register_models as register_models_module
from model_library.registry_utils import (
    get_registry_config,
    get_transcription_registry_config,
)

CUSTOM_YAML = textwrap.dedent("""\
    base-config:
      company: TestCo
      open_source: false
      supports:
        tools: true
        temperature: true
      metadata:
        available_for_everyone: true
      costs_per_million_token:
        input: 1.0
        output: 2.0
      default_parameters: {}

    test-models:
      testprovider/test-model-1:
        label: Test Model 1
        release_date: 2025-01-01
        properties:
          context_window: 8000
          max_tokens: 1000
          reasoning_model: false

      testprovider/test-model-2:
        label: Test Model 2
        release_date: 2025-02-01
        properties:
          context_window: 16000
          max_tokens: 2000
          reasoning_model: false
""")

OVERRIDE_YAML = textwrap.dedent("""\
    base-config:
      company: OpenAI
      open_source: false
      supports:
        tools: true
        temperature: true
      metadata:
        available_for_everyone: true
      costs_per_million_token:
        input: 99.0
        output: 99.0
      default_parameters: {}

    overrides:
      openai/gpt-4o:
        label: GPT-4o Overridden
        properties:
          context_window: 999999
          max_tokens: 9999
          reasoning_model: false
""")


TRANSCRIPTION_YAML = textwrap.dedent("""\
    base-config:
      company: TestCo
      open_source: false
      supports:
        transcription: true
        temperature: false
      metadata:
        available_for_everyone: true
      transcription_streaming: true
      transcription_language:
        format: iso-639-1
        default: en
      default_parameters: {}

    transcription-models:
      testprovider/test-transcribe:
        label: Test Transcribe
        release_date: 2025-01-01
        transcription_cost:
          usd_per_minute: 0.01
          billing_basis: audio
""")


@pytest.fixture
def custom_yaml_file(tmp_path: Path) -> Path:
    f = tmp_path / "custom_models.yaml"
    f.write_text(CUSTOM_YAML)
    return f


@pytest.fixture(autouse=True)
def cleanup_test_models():
    yield
    registry = get_model_registry()
    for key in ["testprovider/test-model-1", "testprovider/test-model-2"]:
        registry.pop(key, None)
        get_transcription_registry().pop(key, None)
    get_transcription_registry().pop("testprovider/test-transcribe", None)


class TestLoadFromFile:
    async def test_models_added_to_registry(self, custom_yaml_file: Path):
        load_custom_model_configs(custom_yaml_file)

        registry = get_model_registry()
        assert "testprovider/test-model-1" in registry
        assert "testprovider/test-model-2" in registry

    async def test_model_fields_are_correct(self, custom_yaml_file: Path):
        load_custom_model_configs(custom_yaml_file)

        model = get_model_registry()["testprovider/test-model-1"]
        assert model.label == "Test Model 1"
        assert model.company == "TestCo"
        assert model.properties is not None
        assert model.properties.context_window == 8000
        assert model.properties.max_tokens == 1000
        assert model.provider_name == "testprovider"
        assert model.provider_endpoint == "test-model-1"
        assert model.full_key == "testprovider/test-model-1"

    async def test_transcription_models_added_to_transcription_registry(
        self, tmp_path: Path
    ):
        f = tmp_path / "transcription.yaml"
        f.write_text(TRANSCRIPTION_YAML)
        load_custom_model_configs(f)

        model = get_transcription_registry()["testprovider/test-transcribe"]
        assert model.transcription_cost is not None
        assert model.transcription_cost.usd_per_minute == 0.01
        assert "testprovider/test-transcribe" not in get_model_registry()

    async def test_override_moves_key_between_registries(
        self, custom_yaml_file: Path, tmp_path: Path
    ):
        load_custom_model_configs(custom_yaml_file)
        as_transcription = tmp_path / "as_transcription.yaml"
        as_transcription.write_text(
            TRANSCRIPTION_YAML.replace("test-transcribe", "test-model-1")
        )

        load_custom_model_configs(as_transcription)
        assert "testprovider/test-model-1" not in get_model_registry()
        assert get_registry_config("testprovider/test-model-1") is None
        assert get_transcription_registry_config("testprovider/test-model-1")

        load_custom_model_configs(custom_yaml_file)
        assert "testprovider/test-model-1" not in get_transcription_registry()
        assert get_registry_config("testprovider/test-model-1")

    async def test_empty_yaml_does_nothing(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")

        registry = get_model_registry()
        size_before = len(registry)
        load_custom_model_configs(f)
        assert len(registry) == size_before

    async def test_invalid_yaml_raises(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("key: [unclosed")

        with pytest.raises(Exception):
            load_custom_model_configs(f)


class TestLoadFromUrl:
    def _mock_response(self, content: str) -> MagicMock:
        mock = MagicMock()
        mock.read.return_value = content.encode("utf-8")
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    async def test_models_added_to_registry(self):
        with patch(
            "urllib.request.urlopen", return_value=self._mock_response(CUSTOM_YAML)
        ) as mock_urlopen:
            load_custom_model_configs("https://example.com/models.yaml")

        mock_urlopen.assert_called_once_with(
            "https://example.com/models.yaml", timeout=DEFAULT_URL_TIMEOUT_SECONDS
        )

        registry = get_model_registry()
        assert "testprovider/test-model-1" in registry
        assert "testprovider/test-model-2" in registry

    async def test_overrides_existing_model(self):
        registry = get_model_registry()
        original = registry["openai/gpt-4o"].model_copy()

        with patch(
            "urllib.request.urlopen", return_value=self._mock_response(OVERRIDE_YAML)
        ):
            load_custom_model_configs("https://example.com/override.yaml")

        model = registry["openai/gpt-4o"]
        assert model.label == "GPT-4o Overridden"
        assert model.costs_per_million_token is not None
        assert model.costs_per_million_token.input == 99.0
        assert model.properties is not None
        assert model.properties.context_window == 999999

        registry["openai/gpt-4o"] = original


class TestLoadLatestValsModelConfigs:
    async def test_loads_every_bundled_yaml_config(self):
        config_dir = Path(register_models_module.__file__).parent / "config"
        expected_files = {
            path.relative_to(config_dir).as_posix()
            for path in [
                *config_dir.glob("*.yaml"),
                *(config_dir / "voice").glob("*.yaml"),
            ]
        }

        with patch(
            "model_library.custom_register_models.load_custom_model_configs"
        ) as mock_load:
            load_latest_vals_model_configs(branch="test-branch")

        prefix = "https://raw.githubusercontent.com/vals-ai/model-library/test-branch/model_library/config/"
        loaded_files = [
            call.args[0].removeprefix(prefix) for call in mock_load.call_args_list
        ]
        assert loaded_files == sorted(loaded_files)
        assert set(loaded_files) == expected_files
        assert "arcee_models.yaml" in loaded_files
        assert "voice/deepgram_models.yaml" in loaded_files
        assert all(not filename.startswith("deprecated/") for filename in loaded_files)


class TestGatewayRegistryLoading:
    async def test_gateway_url_loads_registry_from_gateway(self):
        source_registry = get_model_registry()
        remote_model = source_registry["openai/gpt-4o"].model_dump(mode="json")
        payload = {"models": {"openai/gpt-4o": remote_model}}

        original_registry = register_models_module._registries
        register_models_module._registries = None

        class GatewaySettings:
            def get(self, name: str, default: str | None = None) -> str | None:
                match name:
                    case "MODEL_GATEWAY_URL":
                        return "https://gateway.test/"
                    case "MODEL_GATEWAY_API_KEY":
                        return "sk-gateway"
                    case _:
                        return default

        try:
            with (
                patch("model_library.model_library_settings", GatewaySettings()),
                patch(
                    "httpx.Client.get",
                    return_value=httpx.Response(
                        200,
                        json=payload,
                        request=httpx.Request(
                            "GET",
                            "https://gateway.test/registry?include_excluded_fields=true&include_transcription=true",
                        ),
                    ),
                ) as mock_get,
            ):
                registry = get_model_registry()

            assert list(registry) == ["openai/gpt-4o"]
            assert registry["openai/gpt-4o"].provider_endpoint == "gpt-4o"
            assert (
                registry["openai/gpt-4o"].rate_limit
                == source_registry["openai/gpt-4o"].rate_limit
            )
            assert (
                mock_get.call_args.args[0]
                == "https://gateway.test/registry?include_excluded_fields=true&include_transcription=true"
            )
            assert mock_get.call_args.kwargs["headers"] == {
                "Authorization": "Bearer sk-gateway"
            }
        finally:
            register_models_module._registries = original_registry

    async def test_gateway_registry_failure_does_not_fallback_to_local(self):
        original_registry = register_models_module._registries
        register_models_module._registries = None

        class GatewaySettings:
            def get(self, name: str, default: str | None = None) -> str | None:
                match name:
                    case "MODEL_GATEWAY_URL":
                        return "https://gateway.test/"
                    case "MODEL_GATEWAY_API_KEY":
                        return "sk-gateway"
                    case _:
                        return default

        try:
            with (
                patch("model_library.model_library_settings", GatewaySettings()),
                patch("httpx.Client.get", side_effect=OSError("gateway down")),
                pytest.raises(OSError, match="gateway down"),
            ):
                get_model_registry()

            assert register_models_module._registries is None
        finally:
            register_models_module._registries = original_registry

    async def test_gateway_registry_requires_api_key(self):
        original_registry = register_models_module._registries
        register_models_module._registries = None

        class GatewaySettings:
            def get(self, name: str, default: str | None = None) -> str | None:
                match name:
                    case "MODEL_GATEWAY_URL":
                        return "https://gateway.test/"
                    case "MODEL_GATEWAY_API_KEY":
                        return None
                    case _:
                        return default

        try:
            with (
                patch("model_library.model_library_settings", GatewaySettings()),
                patch("httpx.Client.get") as mock_get,
                pytest.raises(
                    ValueError,
                    match="MODEL_GATEWAY_API_KEY is required to load registry from gateway",
                ),
            ):
                get_model_registry()

            mock_get.assert_not_called()
            assert register_models_module._registries is None
        finally:
            register_models_module._registries = original_registry


class TestEnvVarAutoLoading:
    async def test_custom_config_loaded_at_init(self, custom_yaml_file: Path):
        """MODEL_LIBRARY_CUSTOM_CONFIG triggers load_custom_model_configs during registry init."""
        # save and reset the singleton so get_model_registry re-initializes
        original_registry = register_models_module._registries
        register_models_module._registries = None

        try:
            with patch("model_library.model_library_settings") as mock_settings:

                def get_setting(name: str, default: str | None = None) -> str | None:
                    if name == "MODEL_LIBRARY_CUSTOM_CONFIG":
                        return str(custom_yaml_file)
                    return default

                mock_settings.get.side_effect = get_setting
                registry = get_model_registry()

            assert "testprovider/test-model-1" in registry
            assert "testprovider/test-model-2" in registry
            mock_settings.get.assert_any_call("MODEL_GATEWAY_URL")
            mock_settings.get.assert_any_call("MODEL_LIBRARY_CUSTOM_CONFIG")
        finally:
            # restore original singleton
            register_models_module._registries = original_registry

    async def test_registry_not_published_until_custom_config_load_finishes(self):
        original_registry = register_models_module._registries
        custom_model = get_model_registry()["openai/gpt-4o"].model_copy(
            update={"full_key": "testprovider/test-model-1"}
        )
        register_models_module._registries = None

        class Settings:
            def get(self, name: str, default: str | None = None) -> str | None:
                if name == "MODEL_LIBRARY_CUSTOM_CONFIG":
                    return "custom.yaml"
                return default

        def load_custom_configs(source: str, *, registry):
            assert source == "custom.yaml"
            assert register_models_module._registries is None
            registry["testprovider/test-model-1"] = custom_model

        try:
            with (
                patch("model_library.model_library_settings", Settings()),
                patch(
                    "model_library.register_models._register_models", return_value={}
                ),
                patch(
                    "model_library.custom_register_models.load_custom_model_configs",
                    side_effect=load_custom_configs,
                ),
            ):
                registry = get_model_registry()

            assert "testprovider/test-model-1" in registry
        finally:
            register_models_module._registries = original_registry

    async def test_no_env_var_skips_custom_loading(self):
        """Without MODEL_LIBRARY_CUSTOM_CONFIG, no custom configs are loaded."""
        original_registry = register_models_module._registries
        register_models_module._registries = None

        try:
            with (
                patch("model_library.model_library_settings") as mock_settings,
                patch(
                    "model_library.custom_register_models.load_custom_model_configs"
                ) as mock_load,
            ):
                mock_settings.get.return_value = None
                get_model_registry()

            mock_load.assert_not_called()
        finally:
            register_models_module._registries = original_registry
