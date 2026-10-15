"""Convert Pydantic model schemas to the strict schema sent to the gateway.

Adapted from openai-python 2.54.0, openai/lib/_pydantic.py (Apache-2.0).
This local implementation accepts Pydantic model classes and omits SDK helpers.
See model_library/_licenses/openai-python.txt.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    return _ensure_strict_json_schema(schema, root=schema)


def _ensure_strict_json_schema(
    schema: dict[str, Any], *, root: dict[str, Any]
) -> dict[str, Any]:
    for definitions_name in ("$defs", "definitions"):
        definitions: dict[str, Any] | None = schema.get(definitions_name)
        if isinstance(definitions, dict):
            for definition in definitions.values():
                _ensure_strict_json_schema(definition, root=root)

    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    properties: dict[str, Any] | None = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
        for property_schema in properties.values():
            _ensure_strict_json_schema(property_schema, root=root)

    items: dict[str, Any] | None = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = _ensure_strict_json_schema(items, root=root)

    variants: list[dict[str, Any]] | None = schema.get("anyOf")
    if isinstance(variants, list):
        schema["anyOf"] = [
            _ensure_strict_json_schema(variant, root=root) for variant in variants
        ]

    variants = schema.get("allOf")
    if isinstance(variants, list):
        if len(variants) == 1:
            schema.update(_ensure_strict_json_schema(variants[0], root=root))
            schema.pop("allOf")
        else:
            schema["allOf"] = [
                _ensure_strict_json_schema(variant, root=root) for variant in variants
            ]

    if schema.get("default") is None:
        schema.pop("default", None)

    ref = schema.get("$ref")
    if ref and len(schema) > 1:
        resolved: Any = root
        for part in ref[2:].split("/"):
            resolved = resolved[part]
        schema.update({**resolved, **schema})
        schema.pop("$ref")
        return _ensure_strict_json_schema(schema, root=root)

    return schema
