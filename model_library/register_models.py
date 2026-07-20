import importlib
import json
import pkgutil
import threading
from copy import deepcopy
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from time import monotonic
from urllib.parse import urljoin
from typing import Any, Callable, Type, TypeVar, cast, get_type_hints

import httpx
import yaml
from pydantic import ConfigDict, create_model, model_validator
from pydantic.fields import Field
from pydantic.main import BaseModel

from model_library import model_library_settings, providers
from model_library.base import LLM, LLMConfig, ProviderConfig
from model_library.utils import get_logger

T = TypeVar("T", bound=LLM)

logger = get_logger("register_models")

GATEWAY_REGISTRY_TIMEOUT_SECONDS = 30

"""
Model Registry structure
Do not set model defaults here, they should be set in the LLMConfig class
You can set metadata configs that are not passed into the LLMConfig class here, ex:
    available_for_everyone, deprecated, internal_only, available_as_evaluator, etc.
"""


class Supports(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    images: bool | None = None
    audio: bool | None = None
    videos: bool | None = None
    files: bool | None = None
    batch: bool | None = None
    temperature: bool | None = None
    tools: bool | None = None
    output_schema: bool | None = None

    def resolve(self) -> "Supports":
        """Fill in None fields with LLMConfig defaults."""
        fields = Supports.model_fields
        defaults = {
            name: LLMConfig.model_fields[f"supports_{name}"].default for name in fields
        }
        return Supports(
            **{
                name: val
                if (val := getattr(self, name)) is not None
                else defaults[name]
                for name in fields
            }
        )


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    deprecated: bool = False
    available_for_everyone: bool = True
    internal_only: bool = False
    available_as_evaluator: bool = False
    ignored_for_cost: bool = False


class Properties(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_window: int
    max_tokens: int
    training_cutoff: str | None = None
    reasoning_model: bool


class CacheCost(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    read: float | None = None
    write: float | None = None
    read_discount: float = 1
    write_markup: float = 1

    def get_costs(self, core_input_cost: float) -> tuple[float, float]:
        if self.read:
            read = self.read
        else:
            read = core_input_cost * self.read_discount

        if self.write:
            write = self.write
        else:
            write = core_input_cost * self.write_markup
        return read, write

    @model_validator(mode="after")
    def validate_costs(self):
        if not self.read and not self.read_discount:
            raise ValueError("Either read or read_discount must be set")
        return self


class ContextCost(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    threshold: float
    input: float
    output: float
    cache: CacheCost | None = None

    def get_costs(
        self, core_input_cost: float, core_output_cost: float, token_count: int
    ) -> tuple[float, float]:
        if token_count > self.threshold:
            return (
                self.input,
                self.output,
            )
        return core_input_cost, core_output_cost


class BatchCost(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: float | None = None
    output: float | None = None
    input_discount: float | None = None
    output_discount: float | None = None

    def get_costs(
        self, core_input_cost: float, core_output_cost: float
    ) -> tuple[float, float]:
        if self.input:
            input_cost = self.input
        else:
            assert self.input_discount
            input_cost = core_input_cost * self.input_discount

        if self.output:
            output_cost = self.output
        else:
            assert self.output_discount
            output_cost = core_output_cost * self.output_discount
        return input_cost, output_cost

    @model_validator(mode="after")
    def validate_costs(self):
        if not self.input and not self.input_discount:
            raise ValueError("Either input or input_discount must be set")
        if not self.output and not self.output_discount:
            raise ValueError("Either output or output_discount must be set")
        return self


class CostProperties(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: float
    output: float
    cache: CacheCost | None = None
    batch: BatchCost | None = None
    context: ContextCost | None = None


class BaseProviderProperties(BaseModel):
    """Static base class for dynamic ProviderProperties."""


class GatewayProviderProperties(BaseProviderProperties):
    """Raw provider metadata preserved by Gateway registry clients."""

    model_config = ConfigDict(extra="allow")


def all_subclasses(cls: type) -> list[type]:
    """Recursively find all subclasses of a class."""
    result: list[type] = []
    for subclass in cls.__subclasses__():
        result.append(subclass)
        result.extend(all_subclasses(subclass))
    return result


@cache
def get_dynamic_provider_properties_model() -> type[BaseProviderProperties]:
    field_definitions: dict[str, Any] = {}

    for cls in all_subclasses(ProviderConfig):
        hints = get_type_hints(cls)
        for name, typ in hints.items():
            if name not in field_definitions:
                # type + Field(default=None) tuple
                field_definitions[name] = (typ | None, Field(default=None))

    return create_model(
        "ProviderProperties",
        __base__=BaseProviderProperties,
        __module__=__name__,
        **field_definitions,
    )


class DefaultParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    reasoning_effort: str | bool | None = None
    compute_effort: str | int | bool | None = None


class RawModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    company: str
    label: str
    release_date: date | None = None
    open_source: bool
    documentation_url: str | None = None
    properties: Properties
    supports: Supports
    metadata: Metadata = Field(default_factory=Metadata)
    provider_properties: BaseProviderProperties = Field(
        default_factory=BaseProviderProperties
    )
    costs_per_million_token: CostProperties | None
    alternative_keys: list[str | dict[str, Any]] = Field(default_factory=list)
    default_parameters: DefaultParameters = Field(default_factory=DefaultParameters)
    provider_endpoint: str | None = None

    def model_dump(self, *args: object, **kwargs: object):
        data = super().model_dump(*args, **kwargs)
        # explicitly dump dynamic ProviderProperties instance
        data["provider_properties"] = self.provider_properties.model_dump(
            *args, **kwargs
        )

        default_kwargs = dict(kwargs)
        default_kwargs["exclude_unset"] = True
        data["default_parameters"] = self.default_parameters.model_dump(
            *args, **default_kwargs
        )
        return data


class ModelConfig(RawModelConfig):
    # post processing fields
    provider_endpoint: str  # pyright: ignore[reportIncompatibleVariableOverride, reportGeneralTypeIssues]
    provider_name: str
    full_key: str
    slug: str


ModelRegistry = dict[str, ModelConfig]


def model_config_from_json(data: dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig from a gateway registry JSON object."""
    provider_properties = data["provider_properties"]
    model = ModelConfig.model_validate_json(json.dumps(data))
    model.supports = model.supports.resolve()
    model.provider_properties = GatewayProviderProperties.model_validate(
        provider_properties
    )
    return model


def _gateway_registry_request(gateway_url: str) -> tuple[str, dict[str, str]]:
    from model_library import model_library_settings as current_settings

    api_key = current_settings.get("MODEL_GATEWAY_API_KEY")
    if not api_key:
        raise ValueError(
            "MODEL_GATEWAY_API_KEY is required to load registry from gateway"
        )

    registry_url = urljoin(gateway_url.rstrip("/") + "/", "registry")
    return registry_url, {"Authorization": f"Bearer {api_key}"}


def _model_registry_from_gateway_payload(payload: object) -> ModelRegistry:
    if not isinstance(payload, dict):
        raise ValueError("Gateway registry response must be a JSON object")

    payload_dict = cast(dict[str, object], payload)
    raw_models = payload_dict.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("Gateway registry response must include a models object")

    model_payloads = cast(dict[str, object], raw_models)
    return {
        key: model_config_from_json(cast(dict[str, Any], value))
        for key, value in model_payloads.items()
    }


def fetch_gateway_model_registry(gateway_url: str) -> ModelRegistry:
    """Fetch and parse a model registry snapshot from Gateway."""
    registry_url, headers = _gateway_registry_request(gateway_url)
    logger.info("Loading model registry from gateway: %s", registry_url)
    with httpx.Client(timeout=GATEWAY_REGISTRY_TIMEOUT_SECONDS) as client:
        response = client.get(registry_url, headers=headers)
        response.raise_for_status()
        payload = response.json()

    return _model_registry_from_gateway_payload(payload)


# Folder containing provider YAMLs
path_library = Path(__file__).parent / "config"


def deep_update(
    base: dict[str, Any], updates: dict[str, str | dict[str, Any]]
) -> dict[str, Any]:
    """Recursively update a dictionary, merging nested dictionaries instead of replacing them."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_update(base[key], value)  # pyright: ignore[reportAny]
        else:
            base[key] = value
    return base


def parse_yaml_blocks(
    model_blocks: dict[str, dict[str, dict[str, Any]]],
    registry: ModelRegistry,
) -> None:
    """Parse a single YAML document's model blocks and merge into registry."""
    ProviderProperties = get_dynamic_provider_properties_model()
    # start each provider block with its base-config defaults
    provider_base_config = model_blocks.get("base-config", {})
    for model_block, model_data in model_blocks.items():
        if model_block == "base-config":
            continue

        block_config = deepcopy(provider_base_config)
        if "base-config" in model_data:
            block_config = deep_update(block_config, model_data["base-config"])

        for model_name, model_config in model_data.items():
            if model_name == "base-config":
                continue

            # merge the per-model overrides
            current_model_config = deepcopy(block_config)
            current_model_config = deep_update(current_model_config, model_config)

            provider_properties = current_model_config.pop("provider_properties", {})

            # create model config object
            raw_model_obj: RawModelConfig = RawModelConfig.model_validate(
                current_model_config
            )
            raw_model_obj.supports = raw_model_obj.supports.resolve()

            provider_endpoint = (
                raw_model_obj.provider_endpoint or model_name.split("/", 1)[1]
            )
            # add provider metadata
            model_obj = ModelConfig.model_validate(
                {
                    **raw_model_obj.model_dump(),
                    "provider_name": model_name.split("/")[0],
                    "provider_endpoint": provider_endpoint,
                    "full_key": model_name,
                    "slug": model_name.replace("/", "_"),
                }
            )
            # load provider properties separately since the model was generated at runtime
            model_obj.provider_properties = ProviderProperties.model_validate(
                provider_properties
            )

            registry[model_name] = model_obj

            # add alternative keys
            alternative_keys = cast(
                list[str | dict[str, dict[str, Any]]],
                model_config.get("alternative_keys", []),
            )
            for key_item in alternative_keys:
                match key_item:
                    # check if we have config overrides
                    case str():
                        key = key_item
                        alt_config = {}
                    case dict():
                        key = list(key_item.keys())[0]
                        alt_config = key_item[key]

                alternative_provider_properties = deep_update(
                    deepcopy(provider_properties),
                    alt_config.get("provider_properties", {}),
                )
                copy = deepcopy(registry[model_name])
                provider_name, alternative_model = key.split("/", 1)

                # if same provider, keep endpoint, otherwise override
                if provider_name != copy.provider_name:
                    copy.provider_name = provider_name
                    copy.provider_endpoint = alternative_model

                if alt_config:
                    copy_dict = copy.model_dump()
                    copy_dict = deep_update(copy_dict, alt_config)
                    copy = ModelConfig.model_validate(copy_dict)

                # handle thinking labels
                if copy.properties.reasoning_model and "Nonthinking" in copy.label:
                    copy.label = copy.label.replace("Nonthinking", "Thinking")

                copy.slug = key.replace("/", "_")
                copy.full_key = key
                copy.alternative_keys = []
                copy.provider_properties = ProviderProperties.model_validate(
                    alternative_provider_properties
                )

                registry[key] = copy


def _register_models() -> ModelRegistry:
    logger.debug(f"Loading model registry from {path_library}")

    registry: ModelRegistry = {}

    # load each provider YAML
    yaml_files = list(Path(path_library).glob("*.yaml"))

    # include deprecated model configs (default: not included)
    include_deprecated = model_library_settings.get("MODEL_LIBRARY_INCLUDE_DEPRECATED")
    if isinstance(include_deprecated, str) and include_deprecated.lower() == "true":
        deprecated_dir = Path(path_library) / "deprecated"
        if deprecated_dir.exists():
            yaml_files.extend(deprecated_dir.glob("*.yaml"))

    sections = sorted(yaml_files, key=lambda x: "openai" in x.name.lower())
    for section in sections:
        try:
            with section.open() as file:
                model_blocks = cast(
                    dict[str, dict[str, dict[str, Any]]] | None, yaml.safe_load(file)
                )
        except OSError as error:
            logger.error(f"Error reading {section}: {error}")
            raise
        except yaml.YAMLError as error:
            logger.error(f"Error loading {section}: {error}")
            raise

        if not model_blocks:
            continue

        parse_yaml_blocks(model_blocks, registry)

    return registry


_provider_registry: dict[str, type[LLM]] = {}
_provider_registry_lock = threading.Lock()
_imported_providers = False


def register_provider(name: str) -> Callable[[Type[T]], Type[T]]:
    def decorator(cls: Type[T]) -> Type[T]:
        logger.debug(f"Registering provider {name}")

        if name in _provider_registry:
            raise ValueError(f"Provider {name} is already registered.")
        _provider_registry[name] = cls
        return cls

    return decorator


def _import_all_providers():
    """Import all provider modules. Any class with @register_provider will be automatically registered upon import"""

    package_name = providers.__name__

    # walk all submodules recursively
    for _, module_name, _ in pkgutil.walk_packages(
        providers.__path__, package_name + "."
    ):
        # skip private modules
        if module_name.split(".")[-1].startswith("_"):
            continue
        importlib.import_module(module_name)


def get_provider_registry() -> dict[str, type[LLM]]:
    """Return the provider registry, lazily loading all modules on first call."""
    global _imported_providers
    if not _imported_providers:
        with _provider_registry_lock:
            if not _imported_providers:
                _import_all_providers()
                _imported_providers = True

    return _provider_registry


_model_registry: ModelRegistry | None = None
_model_registry_refreshed_at: float | None = None
_model_registry_lock = threading.Lock()


def _load_model_registry() -> ModelRegistry:
    from model_library import model_library_settings as current_settings

    gateway_url = current_settings.get("MODEL_GATEWAY_URL")
    if gateway_url:
        return fetch_gateway_model_registry(gateway_url)

    get_provider_registry()
    registry = _register_models()
    custom_config = current_settings.get("MODEL_LIBRARY_CUSTOM_CONFIG")
    if custom_config:
        # Local import avoids a circular dependency:
        # custom_register_models imports register_models helpers.
        from model_library.custom_register_models import load_custom_model_configs

        logger.info(f"Loading custom config from {custom_config}")
        load_custom_model_configs(custom_config, registry=registry)
    return registry


def get_model_registry() -> ModelRegistry:
    """Thread-safe singleton access to model registry."""
    global _model_registry
    if _model_registry is None:
        with _model_registry_lock:
            if _model_registry is None:
                _model_registry = _load_model_registry()
    return _model_registry


def refresh_model_registry(
    *,
    refresh_ttl: timedelta,
    allow_stale_on_error: bool = False,
) -> None:
    """Refresh the shared model registry when its refresh TTL has expired."""
    global _model_registry, _model_registry_refreshed_at

    with _model_registry_lock:
        now = monotonic()
        current_registry = _model_registry
        if (
            current_registry is not None
            and _model_registry_refreshed_at is not None
            and now - _model_registry_refreshed_at < refresh_ttl.total_seconds()
        ):
            return

        try:
            registry = _load_model_registry()
        except httpx.HTTPError as error:
            refresh_error: Exception = error
        except OSError as error:
            refresh_error = error
        except ValueError as error:
            refresh_error = error
        except yaml.YAMLError as error:
            refresh_error = error
        else:
            _model_registry = registry
            _model_registry_refreshed_at = monotonic()
            return

        if (
            not allow_stale_on_error
            or current_registry is None
            or _model_registry_refreshed_at is None
        ):
            raise refresh_error
        logger.warning(
            "Failed to refresh model registry; using last known good snapshot",
            exc_info=refresh_error,
        )
        _model_registry_refreshed_at = monotonic()
