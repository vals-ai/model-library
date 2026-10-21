"""The JSON schema AppConfig applies to every published pools document."""

import jsonschema
import pytest

from model_gateway.pool_config_schema import POOL_CONFIG_JSON_SCHEMA

_DOCUMENT = {
    "pools": {"anthropic/default": ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"]},
    "model_pools": [{"pool": "anthropic/default", "models": ["anthropic/claude-x"]}],
    "weights": {"ANTHROPIC_API_KEY_2": 1.0},
    "revision": 3,
}


@pytest.mark.unit
def test_schema_accepts_a_pools_document() -> None:
    jsonschema.validate(_DOCUMENT, POOL_CONFIG_JSON_SCHEMA)


@pytest.mark.unit
def test_schema_accepts_optional_descriptions() -> None:
    descriptions = {"ANTHROPIC_API_KEY": "our main key"}

    jsonschema.validate(
        {**_DOCUMENT, "descriptions": descriptions}, POOL_CONFIG_JSON_SCHEMA
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "change",
    [
        {"revision": -1},
        {"weights": {"ANTHROPIC_API_KEY": -1}},
        {"pools": {"anthropic/default": "ANTHROPIC_API_KEY"}},
        {"unexpected": True},
        {"descriptions": {"ANTHROPIC_API_KEY": 1}},
    ],
)
def test_schema_refuses_a_malformed_document(change: dict[str, object]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**_DOCUMENT, **change}, POOL_CONFIG_JSON_SCHEMA)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["pools", "model_pools", "weights", "revision"])
def test_schema_requires_every_field(field: str) -> None:
    body = {name: value for name, value in _DOCUMENT.items() if name != field}

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(body, POOL_CONFIG_JSON_SCHEMA)
