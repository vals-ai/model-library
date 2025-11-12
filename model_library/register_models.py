import threading
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_type_hints

import yaml
from pydantic import create_model, model_validator
from pydantic.fields import Field
from pydantic.main import BaseModel

from model_library.base import LLM, ProviderConfig
from model_library.providers.ai21labs import AI21LabsModel
from model_library.providers.alibaba import AlibabaModel
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.azure import AzureOpenAIModel
from model_library.providers.cohere import CohereModel
from model_library.providers.deepseek import DeepSeekModel
from model_library.providers.fireworks import FireworksModel
from model_library.providers.google.google import GoogleModel
from model_library.providers.inception import MercuryModel
from model_library.providers.kimi import KimiModel
from model_library.providers.mistral import MistralModel
from model_library.providers.openai import OpenAIModel
from model_library.providers.perplexity import PerplexityModel
from model_library.providers.together import TogetherModel
from model_library.providers.vals import DummyAIModel
from model_library.providers.xai import XAIModel
from model_library.providers.zai import ZAIModel
from model_library.utils import get_logger

MAPPING_PROVIDERS: dict[str, type[LLM]] = {
    "openai": OpenAIModel,
    "azure": AzureOpenAIModel,
    "anthropic": AnthropicModel,
    "together": TogetherModel,
    "mistralai": MistralModel,
    "grok": XAIModel,
    "fireworks": FireworksModel,
    "ai21labs": AI21LabsModel,
    "amazon": AmazonModel,
    "bedrock": AmazonModel,
    "cohere": CohereModel,
    "google": GoogleModel,
    "vals": DummyAIModel,
    "alibaba": AlibabaModel,
    "perplexity": PerplexityModel,
    "deepseek": DeepSeekModel,
    "zai": ZAIModel,
    "kimi": KimiModel,
    "inception": MercuryModel,
}

logger = get_logger(__name__)
# Folder containing provider YAMLs
path_library = Path(__file__).parent / "config"


"""
Model Registry structure
Do not set model defaults here, they should be set in the LLMConfig class
You can set metadata configs that are not passed into the LLMConfig class here, ex:
    available_for_everyone, deprecated, available_as_evaluator, etc.
"""


class Properties(BaseModel):
    context_window: int | None = None
    max_token_output: int | None = None
    training_cutoff: str | None = None
    reasoning_model: bool | None = None


class CacheCost(BaseModel):
    read: float | None = None
    write: float | None = None
    read_discount: float | None = None
    write_markup: float = 1

    def get_costs(
        self, core_input_cost: float, core_output_cost: float
    ) -> tuple[float, float]:
        if self.read:
            read = self.read
        else:
            assert self.read_discount
            read = core_input_cost * self.read_discount

        if self.write:
            write = self.write
        else:
            write = core_output_cost * self.write_markup
        return read, write

    @model_validator(mode="after")
    def validate_costs(self):
        if not self.read and not self.read_discount:
            raise ValueError("Either read or read_discount must be set")
        return self


class ContextCost(BaseModel):
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
    input: float | None = None
    output: float | None = None
    cache: CacheCost | None = None
    batch: BatchCost | None = None
    context: ContextCost | None = None


class ClassProperties(BaseModel):
    supports_images: bool | None = None
    supports_videos: bool | None = None
    supports_files: bool | None = None
    supports_batch_requests: bool | None = None
    supports_temperature: bool | None = None
    supports_tools: bool | None = None
    # vals specific
    deprecated: bool = False
    available_for_everyone: bool = True
    available_as_evaluator: bool = False
    ignored_for_cost: bool = False


"""
Each provider can have a set of provider-specific properties, we however want to accept
any possible property from a provider in the yaml, and validate later. So we join all
provider-specific properties into a single class.
"""


class BaseProviderProperties(BaseModel):
    """Static base class for dynamic ProviderProperties."""

    pass


def all_subclasses(cls: type) -> list[type]:
    """Recursively find all subclasses of a class."""
    result: list[type] = []
    for subclass in cls.__subclasses__():
        result.append(subclass)
        result.extend(all_subclasses(subclass))
    return result


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


ProviderProperties = get_dynamic_provider_properties_model()

if TYPE_CHECKING:
    ProviderPropertiesType = BaseProviderProperties
else:
    ProviderPropertiesType = ProviderProperties


class DefaultParameters(BaseModel):
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None


class RawModelConfig(BaseModel):
    company: str
    label: str
    description: str | None = None
    release_date: date | None = None
    open_source: bool
    documentation_url: str | None = None
    properties: Properties = Field(default_factory=Properties)
    class_properties: ClassProperties = Field(default_factory=ClassProperties)
    provider_properties: ProviderPropertiesType = Field(
        default_factory=ProviderProperties
    )
    costs_per_million_token: CostProperties = Field(default_factory=CostProperties)
    alternative_keys: list[str | dict[str, Any]] = Field(default_factory=list)
    default_parameters: DefaultParameters = Field(default_factory=DefaultParameters)


class ModelConfig(RawModelConfig):
    # post processing fields
    provider_name: str
    provider_endpoint: str
    full_key: str
    slug: str


ModelRegistry = dict[str, ModelConfig]


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


def _register_models() -> ModelRegistry:
    logger.debug(f"Loading model registry from {path_library}")

    registry: ModelRegistry = {}

    # load each provider YAML
    sections = Path(path_library).glob("*.yaml")
    sections = sorted(sections, key=lambda x: "openai" in x.name.lower())
    for section in sections:
        with open(section, "r") as file:
            try:
                model_blocks = cast(
                    dict[str, dict[str, dict[str, Any]]] | None, yaml.safe_load(file)
                )
            except yaml.YAMLError as e:
                logger.error(f"Error loading {section}: {e}")
                raise e

            if not model_blocks:
                continue

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
                    current_model_config = deep_update(
                        current_model_config, model_config
                    )

                    # create model config object
                    raw_model_obj: RawModelConfig = RawModelConfig.model_validate(
                        current_model_config, strict=True
                    )

                    provider_endpoint = (
                        current_model_config.get("provider_endpoint", None)
                        or model_name.split("/", 1)[1]
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
                        if (
                            copy.properties.reasoning_model
                            and "Nonthinking" in copy.label
                        ):
                            copy.label = copy.label.replace("Nonthinking", "Thinking")

                        copy.slug = key.replace("/", "_")
                        copy.full_key = key
                        copy.alternative_keys = []
                        registry[key] = copy

    return registry


_model_registry: ModelRegistry | None = None
_model_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Thread-safe singleton access to model registry."""
    global _model_registry
    if _model_registry is None:
        with _model_registry_lock:
            if _model_registry is None:
                _model_registry = _register_models()
    return _model_registry
