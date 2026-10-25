"""Build the customer catalog and credit policy from the Gateway snapshot."""

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_library.register_models import ModelConfig, RegistryEntry


class ValSmithPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    credits_per_task: Decimal = Field(
        default=Decimal("1"), gt=0, allow_inf_nan=False, strict=False
    )
    required_plan: Literal["individual", "enterprise"] = "individual"

    @field_validator("credits_per_task", mode="before")
    @classmethod
    def reject_float_rate(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("Use a decimal string for fractional credit rates")

        return value


VALSMITH_OVERRIDES: dict[str, ValSmithPolicy] = {
}


def _with_override(key: str, inherited: ValSmithPolicy) -> ValSmithPolicy:
    override = VALSMITH_OVERRIDES.get(key)
    if override is None:
        return inherited

    return ValSmithPolicy.model_validate(
        {**inherited.model_dump(), **override.model_dump(exclude_unset=True)}
    )


def resolve_valsmith_policies(
    registry: Mapping[str, RegistryEntry],
) -> dict[str, ValSmithPolicy]:
    policies = {
        key: _with_override(key, ValSmithPolicy())
        for key, model in registry.items()
        if isinstance(model, ModelConfig)
    }
    aliases: dict[str, ValSmithPolicy] = {}
    for key in sorted(policies):
        for alternative in registry[key].alternative_keys:
            alias = (
                alternative if isinstance(alternative, str) else next(iter(alternative))
            )
            if alias not in policies or registry[alias].alternative_keys:
                continue

            inherited = _with_override(alias, policies[key])
            previous = aliases.setdefault(alias, inherited)
            if previous != inherited:
                raise ValueError(f"Conflicting inherited ValSmith policies for {alias}")

    return {**policies, **aliases}


def _revision(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    return hashlib.sha256(encoded).hexdigest()


def _thinking_value(config: ModelConfig) -> str:
    for parameter in (
        config.default_parameters.reasoning_effort,
        config.default_parameters.compute_effort,
    ):
        if isinstance(parameter, bool):
            return "on" if parameter else "off"

        if parameter is not None:
            return str(parameter)

    return "on" if config.properties.reasoning_model else "off"


def _thinking_sort_key(value: str) -> tuple[int, str]:
    standard_order = ("off", "on", "minimal", "low", "medium", "high", "xhigh", "max")

    return (
        standard_order.index(value) if value in standard_order else len(standard_order),
        value,
    )


def _execution_signatures(config: ModelConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    properties = config.properties.model_dump(mode="json")
    parameters = config.default_parameters.model_dump(mode="json")
    execution = {
        "provider_name": config.provider_name,
        "provider_endpoint": config.provider_endpoint,
        "provider_properties": config.provider_properties.model_dump(mode="json"),
        "supports": config.supports.model_dump(mode="json"),
        "properties": properties,
        "default_parameters": parameters,
    }
    without_reasoning = {
        **execution,
        "properties": {
            key: value for key, value in properties.items() if key != "reasoning_model"
        },
        "default_parameters": {
            key: value
            for key, value in parameters.items()
            if key not in {"reasoning_effort", "compute_effort"}
        },
    }
    return execution, without_reasoning


def valsmith_catalog(
    registry: Mapping[str, RegistryEntry],
    payloads: Mapping[str, dict[str, Any]],
    policies: Mapping[str, ValSmithPolicy],
) -> dict[str, Any]:
    models = {
        key: config
        for key, config in registry.items()
        if key in payloads
        and isinstance(config, ModelConfig)
        and not config.metadata.deprecated
        and not config.metadata.internal_only
        and config.metadata.available_for_everyone
    }
    families = {key: key for key in models}
    effort = {key: _thinking_value(config) for key, config in models.items()}
    signatures = {key: _execution_signatures(config) for key, config in models.items()}
    representatives = {key: {effort[key]: key} for key in models}
    for key, config in sorted(models.items()):
        for alternative in config.alternative_keys:
            alias = (
                alternative if isinstance(alternative, str) else next(iter(alternative))
            )
            if (
                alias not in models
                or alias == key
                or models[alias].alternative_keys
                or families[alias] != alias
                or signatures[key][1] != signatures[alias][1]
            ):
                continue

            representative = representatives[key].get(effort[alias])
            if representative is not None and (
                signatures[representative][0] != signatures[alias][0]
                or policies[representative] != policies[alias]
            ):
                continue

            families[alias] = key
            if representative is None or (
                representative != key and alias < representative
            ):
                representatives[key][effort[alias]] = alias

    options = {
        key: [
            {
                "value": value,
                "label": value.replace("_", " ").replace("-", " ").title(),
                "model_key": members[value],
            }
            for value in sorted(members, key=_thinking_sort_key)
        ]
        for key, members in representatives.items()
        if families[key] == key and len(members) > 1
    }

    resolved = {
        key: {
            **payloads[key],
            "valsmith": policies[key].model_dump(mode="json"),
            "catalog": {
                "family_key": families[key],
                "thinking_options": options.get(key, []),
            },
        }
        for key in sorted(models)
    }
    pricing = {key: record["valsmith"] for key, record in resolved.items()}

    return {
        "models": resolved,
        "schema_version": 1,
        "catalog_revision": _revision(resolved),
        "pricing_revision": _revision(pricing),
    }
