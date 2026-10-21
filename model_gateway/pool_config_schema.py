"""The JSON schema AppConfig checks each published pools document against.

Keep this module dependency-free: the CDK stack imports it to attach the
validator.
"""

from __future__ import annotations

from typing import Any, Final

POOL_CONFIG_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pools", "model_pools", "weights", "revision"],
    "properties": {
        "pools": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "model_pools": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pool", "models"],
                "properties": {
                    "pool": {"type": "string"},
                    "models": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "weights": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "revision": {"type": "integer", "minimum": 0},
        "descriptions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
}
