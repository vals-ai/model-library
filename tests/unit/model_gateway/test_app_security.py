from unittest.mock import patch

from fastapi.testclient import TestClient

import model_gateway.app as gateway_app
import model_gateway.routes.query as query_routes


class GatewaySettings:
    def get(self, name: str, default: str = "") -> str:
        return {
            "MODEL_GATEWAY_API_KEYS": "sk-test",
            "MODEL_GATEWAY_HMAC_SECRET": "test-secret",
        }.get(name, default)

    def unset(self, _key: str) -> None:
        pass


def test_query_trailing_slash_does_not_redirect_or_reach_model_lookup():
    with patch.object(gateway_app, "model_library_settings", GatewaySettings()):
        app = gateway_app.create_app()

    client = TestClient(app)
    request_body = {
        "model": "nonexistent/model",
        "inputs": [{"kind": "text", "text": "hi"}],
    }
    headers = {"Authorization": "Bearer sk-test"}

    with patch.object(query_routes, "get_query_llm") as get_query_llm:
        response = client.post(
            "/query/",
            json=request_body,
            headers=headers,
            follow_redirects=False,
        )

    assert response.status_code == 404
    assert "location" not in response.headers
    get_query_llm.assert_not_awaited()

    canonical_response = client.post("/query", json=request_body, headers=headers)
    assert canonical_response.status_code == 400
