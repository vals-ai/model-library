import json
import os
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

import model_gateway.app as gateway_app

HEADERS = {"Authorization": "Bearer sk-test"}


def _load_json(value: Any) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AssertionError(f"Expected valid JSON: {exc}") from exc


def _make_client(
    *,
    client: tuple[str, int] = ("testclient", 50000),
    runtime_role: str = "combined",
) -> TestClient:
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = json.dumps({"test": "sk-test"})
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
