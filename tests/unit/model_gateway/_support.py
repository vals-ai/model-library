import json
import os
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

import model_gateway.app as gateway_app
from model_gateway.key_assignments import AttemptKey

HEADERS = {"Authorization": "Bearer sk-test"}


class InMemoryAssignmentStore:
    """Single-process stand-in for the key assignments table."""

    def __init__(self) -> None:
        self._assignments: dict[tuple[AttemptKey, str], str] = {}

    async def get(self, attempt_key: AttemptKey, model: str) -> str | None:
        return self._assignments.get((attempt_key, model))

    async def put_if_absent(
        self, attempt_key: AttemptKey, model: str, key_name: str
    ) -> str:
        return self._assignments.setdefault((attempt_key, model), key_name)


def _load_json(value: Any) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AssertionError(f"Expected valid JSON: {exc}") from exc


def _make_client(
    *,
    client: tuple[str, int] = ("testclient", 50000),
    runtime_role: str = "combined",
    api_keys: dict[str, str] | None = None,
) -> TestClient:
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = json.dumps(api_keys or {"test": "sk-test"})
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.dict(os.environ, {"GATEWAY_RUNTIME_ROLE": runtime_role}),
    ):
        app = main.create_app()
        return TestClient(app, client=client)
