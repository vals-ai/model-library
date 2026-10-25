"""Read the current gateway model registry for usage-ledger consumers."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Final, cast

import boto3  # pyright: ignore[reportMissingImports]

_REGISTRY_FETCH_TIMEOUT_SECONDS: Final = 10.0


def fetch_live_models() -> dict[str, dict[str, Any]] | None:
    base_url = os.environ.get("GATEWAY_REGISTRY_BASE_URL", "").strip()
    secret_name = os.environ.get("GATEWAY_REGISTRY_API_KEYS_SECRET_NAME", "").strip()
    if not base_url or not secret_name:
        return None
    try:
        api_key = _gateway_api_key(_secret_value(secret_name))
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/registry",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(
            request, timeout=_REGISTRY_FETCH_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = cast(dict[str, object], payload)["models"]
        if not isinstance(models, dict) or not models:
            raise ValueError("registry response must include a non-empty models map")
        return cast(dict[str, dict[str, Any]], models)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps({"gateway_registry_unavailable": f"{type(exc).__name__}: {exc}"})
        )
        return None


def _gateway_api_key(config_secret_value: str) -> str:
    config = cast(dict[str, object], json.loads(config_secret_value))
    api_keys_raw = config["api_keys"]
    if not isinstance(api_keys_raw, str):
        raise ValueError("gateway config secret api_keys field must be a JSON string")
    api_keys = cast(dict[str, str], json.loads(api_keys_raw))
    if not api_keys:
        raise ValueError("gateway config secret api_keys must not be empty")
    return api_keys[min(api_keys)]


def _secret_value(name: str) -> str:
    client_factory = cast(Any, boto3.client)
    secret = client_factory("secretsmanager").get_secret_value(SecretId=name)
    value = secret.get("SecretString")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Secret {name} must contain SecretString")
    return value
